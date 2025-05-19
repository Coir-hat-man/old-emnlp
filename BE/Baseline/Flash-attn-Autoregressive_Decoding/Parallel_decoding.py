import torch
import torch.nn as nn
# from modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
# from modeling_mistral_kv import MistralForCausalLM as KVMistralForCausalLM
# from llama_model_change import  LlamaForCausalLM
from llama_model import LlamaForCausalLM
from transformers import PreTrainedModel, PretrainedConfig
from utils import *
from medusa_choices import *
from transformers import AutoTokenizer, AutoConfig
import os
import warnings
import time
import os
import json
import argparse
import tqdm
import torch
import torch.nn.functional as F
# from Tree_node import MultiTokenGenerator
from quick_Tree_node import MultiTokenGenerator



class Parallel_decoding(nn.Module):

    def __init__(self,model_path,Node_num=300,medusa_choices=None,threshold=3.5,temperature=0,posterior_alpha=0.09,Device_index=None,choose_ad_num=8):
        # 调用父类的构造方法，传入配置参数
        super().__init__()
        model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.Model_device = torch.device(Device_index) if Device_index is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.model.to(self.Model_device)
        self.lm_head = model.lm_head.to(self.Model_device)
        # self.medusa_head = model.medusa_head.to(self.Model_device)
        self.medusa_layer_num = 5
        self.vocab_size = 32000
        self.temperature = temperature
        self.posterior_alpha = posterior_alpha
        self.choose_ad_num = choose_ad_num

        self.cache_len=[]
        self.last_turn_acc_num = None
        self.input_len = None#记录每个请求的最初始的长度
        self.draft_cache_lens = None
        self.target_cache_lens_for_draft = None
        self.output_ids = None
        self.batch_size = None
        self.gamma = None
        self.tree_mask = None
        self.diag_one = None
        self.father_index = None
        self.history_logp_sum = None
        self.all_spec = None

        self.spec_log_p = None

        self.acc_num = None
        self.acc_ids = None
        self.output_len = None
        self.draft_qlen = None
        self.pad_max_len = None
        self.Batch_node_allocate = None

        self.spec_num_id = None
        self.ALL_Node_num = Node_num
        self.threshold = threshold
        self.topk =10

        self.MT = None
        self.Maxrequestnum = None
        self.Max_gen_len   = None
        self.eos_token_id = AutoConfig.from_pretrained(model_path).eos_token_id
        self.Hidden_size = AutoConfig.from_pretrained(model_path).hidden_size
        
        self.request_list = None
        self.insert_request_num = None
        self.src_indices = None
        self.Hidden_bank = None
        self.free_flag = None
        self.batch_index = None
        self.finsh_index = None
        self.eos_flag = None
        self.insert_flag = False
        self.model_forward = False

        self.generate_time = 0
        self.count = None
        ####
        self.request_step=None
        self.request_generate_time=None

    def set_max_gen_len(self, max_gen_len):
        for layer in self.model.layers:
            layer.self_attn.max_len = max_gen_len
    def param_init(self,Maxrequestnum,max_gen_len):#初始化Kvcahe和相关参数
        if not hasattr(self, "range_tensor"):
            self.range_tensor = torch.arange(0, 1024)[None, :].to(self.Model_device)  # init
            self.reverse_range_tensor = torch.arange(-1024, -32 + 1).unsqueeze(0).to(self.Model_device)
            self.oned_range_tensor = torch.arange(0, 1024).to(self.Model_device)
            self.diag_matrix = torch.zeros((1024, 1024))[None, :, :].to(self.Model_device)
            self.diag_matrix[:, range(1024), range(1024)] = 1

        self.MT = MultiTokenGenerator(Node=self.ALL_Node_num)#初始化为预期分配Token
        #初始化Kvcahe
        self.set_max_gen_len(max_gen_len)
        self.model.init_kv_cache(Maxrequestnum,max_gen_len)
        self.Maxrequestnum = Maxrequestnum#最大请求次数
        self.Max_gen_len   = max_gen_len

        self.request_list =[0 for i in range(self.Maxrequestnum)]
        self.output_ids = torch.zeros((self.Maxrequestnum, max_gen_len),dtype=torch.long).fill_(self.eos_token_id).to(self.Model_device)
        self.input_len = torch.zeros((self.Maxrequestnum)).int().to(self.Model_device)
        self.cache_len = torch.zeros((self.Maxrequestnum)).int().to(self.Model_device)
        self.output_len = torch.zeros((self.Maxrequestnum)).int().to(self.Model_device)
        self.Hidden_bank = torch.zeros((self.Maxrequestnum, self.Hidden_size), dtype=torch.float16).to(self.Model_device)#存储每个请求最好的Hiddenstate
        # count : the output token number of model

        self.acc_ids = torch.zeros((self.Maxrequestnum,self.medusa_layer_num)).int()
        self.acc_num = torch.zeros((self.Maxrequestnum)).int()
        self.count = torch.zeros((self.Maxrequestnum)).int().to(self.Model_device)#记录每个请求已经生成的token数量，后续请求直接累加
        self.free_flag = torch.zeros((self.Maxrequestnum)).int()#标志空闲的Kvcahe，空闲为0，占用为1
        ####
        self.request_step=torch.zeros((self.Maxrequestnum)).int().to(self.Model_device)#记录每个请求经过了多少个step
        self.request_generate_time=torch.zeros((self.Maxrequestnum),dtype=torch.float16).to(self.Model_device)#记录每个请求的decode时间
#更新相关的参数
    def allocate_and_compact(self,request_list,prompt_len):#根据请求数量和Kvcahe占用动态分配空间和排序
        n = len(request_list)
        available_indices = torch.where(self.free_flag == 0)[0]
        selected = available_indices[:n]
        free_index =selected
        for i,index in enumerate(list(free_index)):
            self.request_list[index] = request_list[i]
        self.batch_index = selected.long()
        self.src_indices = free_index.squeeze(-1).long()
        self.insert_request_num = n
        self.free_flag[selected] = 1
        self.cache_len[self.src_indices] = prompt_len.int()
        self.input_len[self.src_indices] = prompt_len.int()
        self.output_len[self.src_indices] = 0
        ####
        self.request_step[self.src_indices] =0
        self.request_generate_time[self.src_indices] =0

    
    def remove_check(self,):#检测是否需要移动数据
        occupied = torch.nonzero(self.free_flag, as_tuple=True)[0]
        occupied_sorted = torch.sort(occupied)[0]
        m = occupied_sorted.size(0)
        
        # 检查是否需要紧缩操作
        if not torch.all(occupied_sorted == torch.arange(m, device=self.free_flag.device)):
            # 记录需要移动的位置
            moves = []
            for new_pos in range(m):
                old_pos = occupied_sorted[new_pos].item()
                if new_pos != old_pos:
                    moves.append([old_pos, new_pos])
            
            # 更新free_flag为紧缩后的状态
            self.free_flag.zero_()
            self.free_flag[:m] = 1
            self.model.batch_move(moves)
            sources, targets = zip(*moves)
            sources_t = torch.tensor(sources, dtype=torch.long, device=self.Model_device)
            targets_t = torch.tensor(targets, dtype=torch.long, device=self.Model_device)
            ####
            components = [
                self.output_ids,     # (max_batch, max_gen_len)
                self.input_len,      # (max_batch,)
                self.cache_len,      # (max_batch,)
                self.output_len,     # (max_batch,)
                self.request_step,  # (max_batch,)
                self.request_generate_time,# (max_batch,)
                self.Hidden_bank     # (max_batch, hidden_size)
            ]
            for tensor in components:
                tensor[targets_t] = tensor[sources_t]
            request_list =self.request_list
            for src, tgt in moves:
                self.request_list[tgt] = request_list[src]
        
    def prefill_target_model(self,input_ids,prefill_type,V_tree_mask=None,prompt_len=None):#进行目标模型prefill 时候的代码
        
        if prefill_type == "prompt_prefill":
            hidden_states = self.model.forward(input_ids.to(self.Model_device), exec_type="prompt_prefill",batch_index=self.batch_index)["last_hidden_state"]
            hidden_states = hidden_states[range(self.insert_request_num), self.input_len[self.src_indices]-1, ...]##修改-1
            self.Hidden_bank[self.src_indices]= hidden_states
            # print(self.batch_index)
        elif prefill_type == "draft_prefill":
            hidden_states = self.model.forward(input_ids.to(self.Model_device),exec_type="draft_prefill", tree_mask=V_tree_mask.to(self.Model_device),cache_lens=self.cache_len[:self.batch_size],draft_qlen=self.draft_qlen)["last_hidden_state"]
            valid_hiddens = [hidden_states[i, :b, :] for i, b in enumerate(self.Batch_node_allocate)]
            valid_hiddens= torch.cat(valid_hiddens, dim=0).unsqueeze(0)
            hidden_states = valid_hiddens

        elif prefill_type =="decoding":#自回归解码
            hidden_states = self.model.forward(input_ids.to(self.Model_device),exec_type="decoding",cache_lens=self.cache_len[:self.batch_size])["last_hidden_state"].squeeze(1)

        return hidden_states
    
    

    def update_inference_inputs(self,remove_kvcache_index=None):
        if remove_kvcache_index is not None:
            self.model.draft_move(self.cache_len[:self.batch_size], remove_kvcache_index)  # 移动Kvcahe
            self.finsh_index = []#完成请求的index
            self.eos_flag = False
            for i in range(self.batch_size):
                self.output_ids[i:i+1, self.output_len[i]:self.output_len[i] + self.acc_num[i]] = self.acc_ids[i:i+1, :self.acc_num[i]]

                if self.eos_token_id in self.acc_ids[i:i+1, :self.acc_num[i]] or self.cache_len[i]>=self.Max_gen_len-100:#防止Kvcahe溢出
                    self.finsh_index.append(i)
                    self.eos_flag = True
                # if self.cache_len[i]>=self.Max_gen_len-100:
                #     print("out of cache!")
            self.cache_len[:self.batch_size] += self.acc_num #更新cachelen
            self.output_len[:self.batch_size] += self.acc_num #更新outputlen
            self.count[:self.batch_size]+= self.acc_num
            ####
            self.request_step[:self.batch_size]+=1
        else:
            self.finsh_index = []
            self.eos_flag = False
            for i in range(self.batch_size):
                self.output_ids[i:i+1, self.output_len[i]:self.output_len[i] + 1] = self.acc_ids[i:i+1]
                if self.eos_token_id in self.acc_ids[i:i+1] or self.cache_len[i]>=self.Max_gen_len-100:#防止Kvcahe溢出:
                    self.finsh_index.append(i)
                    self.eos_flag = True
            self.cache_len[:self.batch_size] += 1 #更新cachelen
            self.output_len[:self.batch_size] += 1 #更新outputlen
            self.count[:self.batch_size]+= 1
            ####
            self.request_step[:self.batch_size]+=1

    def process_requests(self, requests):#将请求打包成模型输入
        prompt_list=[]
        request_list =[]
        for i in range(len(requests)):
            request =requests[i]
            request.start_time = time.time()
            prompt_list.append(request.prompt)
            request_list.append(request)
            
        inputs = self.tokenizer(prompt_list, return_tensors="pt", padding='longest', truncation=True, padding_side="right")
        cache_len = inputs['attention_mask'].sum(dim=-1).to(self.Model_device)
        input_ids = inputs['input_ids'].to(self.Model_device)
        return request_list,input_ids, cache_len
    
    def tree_spec_generate(self, requests):
        self.model_forward = True
        with torch.no_grad(): 
            if requests:#如果不是空请求就进行Prefill
                request_list,input_ids, prompt_length = self.process_requests(requests)
                self.allocate_and_compact(request_list,prompt_length)
                # prefill LLM
                prefill_begin_time = time.time()
                self.prefill_target_model(input_ids,prefill_type="prompt_prefill",prompt_len=prompt_length)
                torch.cuda.synchronize()
                prefill_end_time = time.time()
                #print(f"prefill time: {(prefill_end_time- prefill_begin_time) * 1000} ms")
            self.remove_check()#检测是否需要移动数据
            self.batch_size = sum(self.free_flag)
            hidden_states = self.Hidden_bank[:self.batch_size]
            for out_index in range(0, self.Max_gen_len):
                decode_begin_time = time.time()
                start_gerate_time = time.time()
                llm_logit = self.lm_head(hidden_states)
                acc_ids = torch.argmax(llm_logit, dim=-1)#top1解码
                self.acc_ids= acc_ids.unsqueeze(1)
                hidden_states = self.prefill_target_model(self.acc_ids,"decoding")
                self.update_inference_inputs()
                ####
                generate_time = time.time() - start_gerate_time
                self.request_generate_time[:self.batch_size]+=generate_time
                if self.eos_flag == True:

                    self.Hidden_bank[:self.batch_size] = hidden_states.squeeze(1)
                    finsh_request_list = []
                    for index in self.finsh_index:

                        request = self.request_list[index]
                        output_len = int(self.output_len[index])
                        ####
                        step=int(self.request_step[index])
                        g_time=float(self.request_generate_time[index])
                        request.steps = step
                        request.generate_time=g_time
                        request.output_len = output_len
                        output_ids = self.output_ids[index:index+1,:self.output_len[index]]
                        request.output_ids = output_ids.tolist()
                        finsh_request_list.append(request)
                        self.free_flag[index] = 0
                    return finsh_request_list
                elif  self.insert_flag == True:#请求没完成但是有请求需要插入
                    self.Hidden_bank[:self.batch_size] = hidden_states.squeeze(1)
                    return None
                torch.cuda.synchronize()
                decode_end_time = time.time()
                # print(f"decode time: {(decode_end_time - decode_begin_time) * 1000} ms")



