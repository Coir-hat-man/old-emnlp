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
def print_memory_usage(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1024**3  # 转换为GB
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[{prefix}] Current allocated: {allocated:.4f}GB, "
          f"Peak allocated: {max_allocated:.4f}GB, "
          f"Reserved: {reserved:.4f}GB")
class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 2.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        medusa_num_heads=5,
        medusa_num_layers=1,
        base_model_name_or_path="/home/zhouyh/model/medusa_vicuna_7b",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path

class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class MedusaModelABC(nn.Module):
    """The Medusa Language Model Head.

    This module creates a series of prediction heads (based on the 'medusa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    def __init__(
        self,
        config,
    ):
        """
        Args:
            config (PretrainedConfig): The configuration of the MedusaModel.
        """
        super().__init__(config)
        # For compatibility with the old APIs

        medusa_num_heads = config.medusa_num_heads
        medusa_num_layers = config.medusa_num_layers
        base_model_name_or_path = config._name_or_path
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.medusa = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        # Create a list of Medusa heads
        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * medusa_num_layers),
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                )
                for _ in range(medusa_num_heads)
            ]
        )
    # Add a link named base_model to self
    @property
    def base_model(self):
        return self
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the medusa_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
                config=config,
            )
        except:
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            base_model_config.medusa_num_heads = 5 # TODO: fix the uploaded config (only include 2 heads)
            base_model_config.medusa_num_layers = config.medusa_num_layers
            model = super().from_pretrained(
                config.base_model_name_or_path,
                *args,
                **kwargs,
                config=base_model_config,
            )
            medusa_head_path = os.path.join(pretrained_model_name_or_path, "medusa_lm_head.pt")
            if os.path.exists(medusa_head_path):
                filename = medusa_head_path
            else:
                filename = hf_hub_download(pretrained_model_name_or_path, "medusa_lm_head.pt")
            medusa_head_state_dict = torch.load(filename, map_location=model.device)
            model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)
            return model
    def get_tokenizer(self):
        return self.tokenizer
    def Medusa_head(self,hidden_states):
        #hidden_states batch X 1 X hidden_dim
        medusa_logits = []
        # TODO: Consider parallelizing this loop for efficiency?
        for i in range(self.medusa):
            medusa_logits.append(self.medusa_head[i](hidden_states))
        return torch.stack(medusa_logits)

class MedusaModelLlama(MedusaModelABC, LlamaForCausalLM):
    pass

class MedusaModel():
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the medusa_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        except:
            # MEDUSA-v0.1 load
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            config.model_type = base_model_config.model_type

        if config.model_type == "llama":
            return MedusaModelLlama.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        elif config.model_type == "mistral":
            return MedusaModelMistral.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        else:
            raise ValueError("Only support llama and mistral for now!!")


class Parallel_decoding(MedusaModel):

    def __init__(self,model_path,Node_num=None,threshold=3.5,temperature=0,posterior_alpha=0.09,Device_index=None):
        # 调用父类的构造方法，传入配置参数
        super().__init__()
        model = MedusaModel.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.Model_device = torch.device(Device_index) if Device_index is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.model.to(self.Model_device)
        self.lm_head = model.lm_head.to(self.Model_device)
        self.medusa_head = model.medusa_head.to(self.Model_device)
        self.medusa_layer_num = 5
        self.vocab_size = 32000
        self.temperature = temperature
        self.posterior_alpha = posterior_alpha
        self.Node_num = Node_num

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

        self.MT = MultiTokenGenerator()#初始化为预期分配Token
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
    
    
    def verfiy(self,llm_logitslist,candidateslist,retrieve_indices_list):  # select best candidates
        acc_nums=[]
        acc_ids=[]
        best_node_index =[]
        remove_kvcache_index = []#记录需要移动的位置
        #根据temperature来判断
        for i in range(self.batch_size):
            llm_logit= llm_logitslist[i]
            candidate= candidateslist[i]
            best_candidate, accept_length = evaluate_posterior(
                llm_logit, candidate, self.temperature
            )
            remove_kvcache_index.append(retrieve_indices_list[i][best_candidate:best_candidate+1,])
            # acc_id=candidate[None, best_candidate, : ]
            acc_id=candidate[None, best_candidate, : accept_length + 1]
            pad_size = self.medusa_layer_num - accept_length -1
            acc_id= F.pad(acc_id, (0, pad_size), mode='constant', value=0)
            best_node_index.append(int(retrieve_indices_list[i][best_candidate,accept_length]))
            acc_ids.append(acc_id)
            acc_nums.append(accept_length+1)#注意这里得加一,index不代表实际接收token数
        #更新下一轮的输入
        self.acc_ids=torch.stack(acc_ids).squeeze(1)
        self.acc_num=torch.Tensor(acc_nums).int().to(self.Model_device)
        start_indices = [sum(self.Batch_node_allocate[:i]) for i in range(len(self.Batch_node_allocate))]
        bestnodes_index = [start + index for start, index in zip(start_indices, best_node_index)]##修改
        return bestnodes_index,remove_kvcache_index

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

    def draft_produce(self,hidden_states,MT,prune=True):  # produce draft sequence based on parallel decoding          
        #hidden_states batch X 1 X hidden_dim
        medusa_logits = []
        for i in range(self.medusa_layer_num):
            medusa_logits.append(self.medusa_head[i](hidden_states))
        medusa_logits =torch.stack(medusa_logits).squeeze(2).permute(1, 0, 2)#batch X layer X hidden_dim
        logits = self.lm_head(hidden_states)


        cart_candidateslist, tree_candidateslist , tree_attn_masklist ,retrieve_indiceslist,node_num=MT.quick_node_prune(medusa_logits,logits.squeeze(),[1 for _ in range(self.batch_size)],self.Node_num)
        self.Batch_node_allocate = node_num
        self.draft_qlen  = node_num
        self.pad_max_len  = max(self.draft_qlen)#记录最大的q_len,后面进行pad          
        batch_tree_candidates = []
        batch_tree_mask = []
        for i in range(self.batch_size):
            pad_tree_candidtates = torch.zeros((1,self.pad_max_len)).long()
            pad_tree_mask = medusa_logits.new_ones((1,self.pad_max_len,self.pad_max_len))
            tree_candidates = tree_candidateslist[i] 
            medusa_attn_mask =tree_attn_masklist[i]
            pad_tree_candidtates[:,:tree_candidates.size(0)] = tree_candidates
            pad_tree_mask[:,:tree_candidates.size(0),:tree_candidates.size(0)] = medusa_attn_mask
            pad_tree_mask = torch.tril(pad_tree_mask)

            batch_tree_candidates.append(pad_tree_candidtates)
            batch_tree_mask.append(pad_tree_mask)


        batch_tree_candidates = torch.stack(batch_tree_candidates).squeeze(1)
        batch_tree_mask = torch.stack(batch_tree_mask).squeeze(1).squeeze(1)#[batch_size,seq_len,seq_len]
        batch_candidates_list = cart_candidateslist
        batch_retrieve_indices_list = retrieve_indiceslist
        return batch_candidates_list,batch_tree_candidates,batch_tree_mask ,batch_retrieve_indices_list#后面两个是pad的torch tensor矩阵 ，前面是list

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
                self.prefill_target_model(input_ids,prefill_type="prompt_prefill",prompt_len=prompt_length)
                prefill_end_time = time.time()
            self.remove_check()#检测是否需要移动数据
            self.batch_size = sum(self.free_flag)
            hidden_states = self.Hidden_bank[:self.batch_size]
            for out_index in range(0, self.Max_gen_len):
                start_gerate_time = time.time()    
                batch_candidates_list,batch_tree_candidates,batch_tree_mask ,batch_retrieve_indices_list=self.draft_produce(hidden_states,self.MT)
                
                hidden_states_node =  self.prefill_target_model(batch_tree_candidates,
                                                        prefill_type="draft_prefill",
                                                        V_tree_mask=batch_tree_mask)
                llm_logits = self.lm_head(hidden_states_node)
                
                llm_logit_batch_list = []#将他从一维度抽取出来

                index = 0
                for i in range(self.batch_size):
                    llm_logit = llm_logits[:,index:index+self.Batch_node_allocate[i],:].squeeze(0)
                    llm_logit=  llm_logit[batch_retrieve_indices_list[i]]#用路径去取 变为path*layer*hidden
                    llm_logit_batch_list.append(llm_logit)
                    index += self.Batch_node_allocate[i]##修改
                best_node_index,remove_kvcache_index=self.verfiy(llm_logit_batch_list,batch_candidates_list,batch_retrieve_indices_list)
                hidden_states = torch.stack([hidden_states_node[:,best_node_index[i]:best_node_index[i]+1,] for i in range(self.batch_size)]).squeeze(2).squeeze(2).to(self.Model_device)
                self.update_inference_inputs(remove_kvcache_index)
                ####
                generate_time = time.time() - start_gerate_time
                self.request_generate_time[:self.batch_size]+=generate_time
                # print(self.acc_num)
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
