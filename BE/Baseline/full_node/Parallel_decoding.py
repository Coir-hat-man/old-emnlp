import torch
import torch.nn as nn
# from modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
# from modeling_mistral_kv import MistralForCausalLM as KVMistralForCausalLM
# from llama_model_change import  LlamaForCausalLM
from llama_model import LlamaForCausalLM
from transformers import PreTrainedModel, PretrainedConfig
from utils import *
from choices import *
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

from cnets import Model
from configs import EConfig
import time

# from Tree_node import MultiTokenGenerator
# from quick_Tree_node import MultiTokenGenerator
def print_memory_usage(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1024**3  # 转换为GB
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[{prefix}] Current allocated: {allocated:.4f}GB, "
          f"Peak allocated: {max_allocated:.4f}GB, "
          f"Reserved: {reserved:.4f}GB")
class EaModel(nn.Module):

    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            ea_model_path,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        self.eos_token_id = AutoConfig.from_pretrained(self.base_model_name_or_path).eos_token_id
        self.hidden_size = AutoConfig.from_pretrained(self.base_model_name_or_path).hidden_size
        config = EConfig.from_pretrained(ea_model_path)
        self.ea_layer = Model(config)
        self.cache_lens = None
        self.logits_processor = None
        self.new_token = None
        self.hidden_bank = None
        self.finsh_request_list = []

        low_memory=False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device

        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.device=device
        self.ea_layer.init_tree()
        
        # 请求处理使用
        self.max_request_num = None #请求使用
        self.max_gen_len = None
        self.input_len = None
        self.cache_len = None
        self.output_len = None
        
        # 用于buff的情况
        self.tree_buffers =  None
        self.tree_choices =  None
        
        self.insert_request_num = None
        
        self.cache_len=[]
        self.last_turn_acc_num = None
        self.input_len = None#记录每个请求的最初始的长度
        self.draft_cache_lens = None
        self.target_cache_lens_for_draft = None
        self.output_ids = None
        self.batch_size = None
        self.gamma = None
        self.tree_mask = None
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
        self.topk = 10

        self.request_list = None
        self.insert_request_num = None
        self.src_indices = None
        self.hidden_bank = None
        self.free_flag = None
        self.batch_index = None
        self.finsh_index = None
        self.eos_flag = None
        self.insert_flag = False
        self.model_forward = False

        self.generate_time = 0
        self.count = None
        
        self.out_idx = 0
        
                

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            base_model_path=None,
            ea_model_path=None,
            **kwargs,
    ):

        base_model = LlamaForCausalLM.from_pretrained(
            base_model_path, **kwargs
        )

        configpath=os.path.join(ea_model_path,"config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")
        model = cls(
            base_model,
            base_model_path,
            configpath
        )
        load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
        if not os.path.exists(load_model_path):
            load_model_path=hf_hub_download(ea_model_path, "pytorch_model.bin")
        ea_layer_state_dict = torch.load(load_model_path,
                                         map_location=base_model.device)
        model.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        
        # for i, layer in enumerate(model.ea_layer):
        #     print(f"\n==== ea_layer Layer {i} ====")
        #     for name, param in layer.named_parameters():
        #         print(f"{name}:\n{param.data}")
        

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
            init=True,
            cache_lens = None,
            tree_buffers = None,
            exec_type=None,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            # print(f"past_key_values.shape in eamodel is {past_key_values[0][0].data.shape}, past_key_values[0][1] is {past_key_values[0][0].current_length}")
            outputs = self.base_model.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids,
                cache_lens=cache_lens,
                tree_buffers=tree_buffers,
                exec_type=exec_type,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0].clone()
        if init:
            if self.logits_processor is not None:
                # logits = orig[:, -1]
                selected_logits = orig[torch.arange(orig.size(0), device=orig.device), self.cache_lens-1]
                logits = self.logits_processor(None, selected_logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                token = torch.multinomial(probabilities, 1)
            else:
                selected_logits = orig[torch.arange(orig.size(0), device=orig.device), self.cache_lens-1]  # shape: (batch_size, vocab_size)

                # argmax to get token id
                token = torch.argmax(selected_logits, dim=-1)  # shape: (batch_size,)
                token = token[:, None]  # shape: (batch_size, 1)

            input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

            ea_logits = self.ea_layer.topK_genrate(hidden_states, input_ids, self.base_model.lm_head, self.logits_processor,attention_mask=attention_mask)
            # print(f"ea_logits: {ea_logits}")
            if output_orig:
                return ea_logits, outputs, orig, hidden_states, token
            return ea_logits, hidden_states, token
        else:
            if output_orig:
                return outputs, orig, hidden_states

class Parallel_decoding(EaModel):

    def __init__(self,base_model_path, EAGLE_model_path, node_choices=None,threshold=3.5,temperature=0,posterior_alpha=0.09,Device_index=None):
        # 设置设备
        self.Model_device = f"cuda:{Device_index}" if Device_index is not None else "cuda"

        # 加载基础模型，只加载一次，并放在显存友好的位置
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",  # 自动放到合适的GPU
        )

        # 传给父类
        config_path = os.path.join(EAGLE_model_path, "config.json")
        if not os.path.exists(config_path):
            config_path = hf_hub_download(EAGLE_model_path, "config.json")

        super().__init__(base_model, base_model_path, config_path)

        # 加载 EA layer 权重（只加载一次！）
        model_path = os.path.join(EAGLE_model_path, "pytorch_model.bin")
        if not os.path.exists(model_path):
            model_path = hf_hub_download(EAGLE_model_path, "pytorch_model.bin")

        ea_layer_state_dict = torch.load(model_path, map_location="cpu")
        self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        self.ea_layer.to(torch.float16).to(self.Model_device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.Model_device = torch.device(Device_index) if Device_index is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.base_model = base_model.to(self.Model_device)
        self.ea_layer = self.ea_layer.to(self.Model_device)
        self.lm_head = base_model.lm_head.to(self.Model_device)
        self.vocab_size = 32000
        self.temperature = temperature
        self.posterior_alpha = posterior_alpha
        self.Node_choice = node_choices
        
        # print(f"self.base_model: {self.base_model}")
        # print("--"* 30)
        
        # q_proj_weight = self.base_model.model.layers[0].self_attn.q_proj.weight
        # print("Layer 0 Q projection weights:\n", q_proj_weight)
        
        # print(f"\n--- Parameters of self.base_model.lm_head ({type(self.base_model.lm_head )}) ---")
        # for name, param in self.base_model.lm_head.named_parameters():
        #     if param.requires_grad:
        #         print(f"Name: {name}")
        #         print(f"Shape: {param.shape}")
        #         print(f"Device: {param.device}")
        #         print(f"Requires Grad: {param.requires_grad}")
        #         print(f"Data (first few elements if large, or full if small):")
        #         # 为了避免打印过多数据，可以只打印一部分
        #         if param.numel() > 20: # 如果参数元素超过20个
        #             print(param.flatten()[:10].tolist(), "...") # 打印前10个扁平化后的元素
        #         else:
        #             print(param.data)
        #         print("-" * 20)

        self.cache_len = None
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
        self.eos_token_id = AutoConfig.from_pretrained(base_model_path).eos_token_id
        self.Hidden_size = AutoConfig.from_pretrained(base_model_path).hidden_size
        
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
        self.max_predict_num = 7

        self.generate_time = 0
        self.count = None
        
        # eagle add
        self.tree_logits_list = None
        self.tree_logits_prob_list = None
        self.tree_buffers = None # 用于存储token tree的相关信息
        self.tokens = None
        self.draft_hidden = None
        self.draft_input_ids = None
        self.max_draft_len = None
        ####
        self.request_step=None
        self.request_generate_time=None

    def set_max_gen_len(self, max_gen_len):
        for layer in self.base_model.model.layers:
            layer.self_attn.max_len = max_gen_len
            
    def param_init(self,Maxrequestnum,max_gen_len):#初始化Kvcahe和相关参数
        if not hasattr(self, "range_tensor"):
            self.range_tensor = torch.arange(0, 1024)[None, :].to(self.Model_device)  # init
            self.reverse_range_tensor = torch.arange(-1024, -32 + 1).unsqueeze(0).to(self.Model_device)
            self.oned_range_tensor = torch.arange(0, 1024).to(self.Model_device)
            self.diag_matrix = torch.zeros((1024, 1024))[None, :, :].to(self.Model_device)
            self.diag_matrix[:, range(1024), range(1024)] = 1

        # self.MT = MultiTokenGenerator()#初始化为预期分配Token
        #初始化Kvcahe
        self.set_max_gen_len(max_gen_len)
        self.base_model.model.init_kv_cache(Maxrequestnum,max_gen_len)
        self.Maxrequestnum = Maxrequestnum#最大请求次数
        self.Max_gen_len   = max_gen_len

        self.request_list =[0 for i in range(self.Maxrequestnum)]
        self.output_ids = torch.zeros((self.Maxrequestnum, max_gen_len),dtype=torch.long).fill_(self.eos_token_id).to(self.Model_device)
        self.input_len = torch.zeros((self.Maxrequestnum)).int().to(self.Model_device)
        self.cache_len = torch.zeros((self.Maxrequestnum)).int().to(self.Model_device)
        self.output_len = torch.zeros((self.Maxrequestnum)).int().to(self.Model_device)
        self.Hidden_bank = torch.zeros((self.Maxrequestnum, self.Hidden_size), dtype=torch.float16).to(self.Model_device)#存储每个请求最好的Hiddenstate
        # count : the output token number of model

        self.acc_ids = torch.zeros((self.Maxrequestnum, self.max_predict_num)).int()
        self.acc_num = torch.zeros((self.Maxrequestnum)).int().to(self.Model_device)
        self.count = torch.zeros((self.Maxrequestnum)).int().to(self.Model_device)#记录每个请求已经生成的token数量，后续请求直接累加
        self.free_flag = torch.zeros((self.Maxrequestnum)).int()#标志空闲的Kvcahe，空闲为0，占用为1
        
        
        # eagle add
        self.ea_layer.reset_kv(self.Maxrequestnum)
        #存储每个请求最好的tree_logits[batch_size, tree_node_num, TOPK]
        tree_node_num = 11 # TODO: 11 is the number of nodes with branches in the Eagle static tree, need to be changed
        self.tree_logits_list = torch.zeros((self.Maxrequestnum, tree_node_num, TOPK), dtype=torch.int64).to(self.Model_device)
        self.tree_logits_prob_list = torch.zeros((self.Maxrequestnum, tree_node_num, TOPK), dtype=torch.float16).to(self.Model_device)
        self.tokens = self.tokens = torch.zeros((self.Maxrequestnum, 1), dtype=torch.long).to(self.Model_device)# init tokens which are used to generate tree
        max_token_depth = 6
        self.draft_hidden = torch.zeros((self.Maxrequestnum, max_token_depth, self.Hidden_size), dtype=torch.float16).to(self.Model_device)
        self.draft_hidden_index = torch.zeros((self.Maxrequestnum, max_token_depth), dtype=torch.long).to(self.Model_device)
        self.draft_input_ids = torch.zeros((self.Maxrequestnum, max_token_depth + 1), dtype=torch.long).to(self.Model_device)
        
        
        tree_choices = mc_sim_7b_63
        
        tree_buffers = generate_tree_buffers(
            tree_choices, device=self.base_model.model.layers[-1].self_attn.q_proj.weight.device
        )
        tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
            self.base_model.lm_head.weight.device)
        tree_buffers["tree_position_ids"]=tree_buffers["tree_position_ids"].to(self.base_model.device)
        self.tree_buffers = tree_buffers

        self.tree_buffers["retrieve_indices_batch"]=tree_buffers["retrieve_indices"].expand(self.Maxrequestnum,-1,-1)
        # print(f"self.tree_buffers['node_index']: {self.tree_buffers['node_index'].shape} {self.tree_buffers['node_index']}")
        
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
        # print(f"allocate_and_compact self.batch_index: {self.batch_index} self.src_indices: {self.src_indices}")
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
            self.base_model.model.batch_move(moves)
            self.ea_layer.batch_move(moves)  ## ea_layer change kv cache
            sources, targets = zip(*moves)
            sources_t = torch.tensor(sources, dtype=torch.long, device=self.Model_device)
            targets_t = torch.tensor(targets, dtype=torch.long, device=self.Model_device)
            ####
            components = [
                self.output_ids,     # (max_batch, max_gen_len)
                self.input_len,      # (max_batch,)
                self.cache_len,      # (max_batch,)
                self.output_len,     # (max_batch,)
                self.Hidden_bank,     # (max_batch, hidden_size)
                self.ea_layer.draft_cache_lens,
                self.request_step,  # (max_batch,)
                self.request_generate_time,# (max_batch,)
            ]

            for tensor in components:
                tensor[targets_t] = tensor[sources_t]
            request_list =self.request_list
            for src, tgt in moves:
                self.request_list[tgt] = request_list[src]
        
    def prefill_target_model(self,input_ids,prefill_type,V_tree_mask=None,prompt_len=None):#进行目标模型prefill 时候的代码
        
        if prefill_type == "prompt_prefill":
            hidden_states = self.base_model.model.forward(input_ids.to(self.Model_device), exec_type="prompt_prefill", batch_index=self.batch_index)["last_hidden_state"]

            output = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), self.cache_len[self.src_indices]-1]
            
            selected_logits = self.lm_head(output)

            token = torch.argmax(selected_logits, dim=-1)  # shape: (batch_size,)
            token = token[:, None]  # shape: (batch_size, 1)
            draft_input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
            return hidden_states, draft_input_ids, input_ids, token
            ## 得确定一下这些token是不是都是长度为1？还是长度不相同，长度不相同的话，得考虑如何进一步改善了
        
        elif prefill_type == "draft_prefill":
            # print(f"input_ids: {input_ids.shape}, V_tree_mask: {V_tree_mask.shape} {V_tree_mask[0].shape} self.batch_size: {self.batch_size}, draft_qlen: {self.draft_qlen}")
            hidden_states = self.base_model.model.forward(input_ids.to(self.Model_device), exec_type="draft_prefill", \
                    tree_mask=V_tree_mask.to(self.Model_device),\
                    cache_lens=self.cache_len[:self.batch_size], draft_qlen=self.draft_qlen)["last_hidden_state"]
            valid_hiddens = [hidden_states[i, :b, :] for i, b in enumerate(self.Batch_node_allocate)]
            valid_hiddens= torch.cat(valid_hiddens, dim=0).unsqueeze(0)
            hidden_states = valid_hiddens
            return hidden_states

        elif prefill_type =="decoding":#自回归解码
            hidden_states = self.base_model.model.forward(input_ids.to(self.Model_device),exec_type="decoding",cache_lens=self.cache_len[:self.batch_size])["last_hidden_state"].squeeze(1)

        return hidden_states
   
    # ### 05.15  16.45
    def verify(self, llm_logits, batch_candidates_list, batch_tree_candidates, batch_retrieve_indices_list, hidden_states_node):
        device = llm_logits.device
        batch_size = self.batch_size
        
        # Compute node ranges directly
        indices = torch.cumsum(
            torch.cat([torch.zeros(1, device=device), 
                    torch.tensor(self.Batch_node_allocate[:-1], device=device)]), 
            dim=0
        ).long()
        node_ranges = torch.stack([indices, indices + torch.tensor(self.Batch_node_allocate, device=device)], dim=1)
        
        # Expand logits once
        llm_logits_expanded = llm_logits.squeeze(0)  # [total_nodes, seq_len, vocab]
        
        # Normalize indices more efficiently
        normalized_indices = [idx.view(-1, 1) if idx.dim() == 1 else 
                            idx.squeeze(-1) if idx.dim() == 3 else idx 
                            for idx in batch_retrieve_indices_list]
        
        remove_kvcache_index = []
        max_accept_len = 0
        
        # Process each batch item
        for i in range(batch_size):
            # start, end = node_ranges[i]
            for_test_begin_time = time.time()
            start = node_ranges[i][0].item()
            end = node_ranges[i][1].item()
            batch_logits = llm_logits_expanded[start:end]
            retrieve_idx = normalized_indices[i]
            candidates = batch_candidates_list[i].to(device)
            
            # Handle dimension constraints
            valid_candidates = min(retrieve_idx.size(0), candidates.size(0))
            valid_seq_len = min(retrieve_idx.size(1), candidates.size(1))
            
            # Direct indexing for candidate logits
            candidate_logits = batch_logits[retrieve_idx[:valid_candidates, :valid_seq_len]]
            
            # Compute accept lengths in one step
            predicted_tokens = torch.argmax(candidate_logits[:, :valid_seq_len-1], dim=-1)
            posterior_mask = (candidates[:valid_candidates, 1:valid_seq_len] == predicted_tokens)
            batch_accept_lengths = torch.cumprod(posterior_mask, dim=-1).sum(dim=-1)
            
            # Find best candidate directly
            accept_length, best_candidate = torch.max(batch_accept_lengths, dim=0)
            accept_length = accept_length.item()
            best_candidate = best_candidate.item()
            
            # Store best indices
            best_remove_idx = retrieve_idx[best_candidate][:accept_length+1]
            remove_kvcache_index.append(best_remove_idx.unsqueeze(0) if best_remove_idx.dim() == 1 else best_remove_idx)
            
            # torch.cuda.synchronize()
            # print(f"index time: {(time.time() - for_test_begin_time) * 1000} ms")
            
            # Update hidden states directly without intermediate variable
            # hidden_len = min(hidden_states_node[0][start:end][retrieve_idx][best_candidate].size(0), 
            #                 self.draft_hidden.size(1))
            # hidden_begin_time = time.time()
            self.draft_hidden[i] = hidden_states_node[0][start:end][retrieve_idx][best_candidate]
            # torch.cuda.synchronize()
            # hidden_end_time = time.time()
            # print(f"hidden_states_node time: {(hidden_end_time - hidden_begin_time) * 1000} ms")
            
            # Update IDs in one operation
            self.draft_input_ids[i, :accept_length+1] = candidates[best_candidate, :accept_length+1]
            self.draft_input_ids[i, accept_length+1] = torch.argmax(candidate_logits[best_candidate, accept_length])
            
            # Update accumulated states directly
            self.acc_ids[i, :accept_length+2] = self.draft_input_ids[i, :accept_length+2]
            self.acc_num[i] = accept_length
            self.tokens[i] = self.draft_input_ids[i, accept_length+1]
            
            max_accept_len = max(max_accept_len, accept_length + 1)
            # torch.cuda.synchronize()
            # for_test_end_time = time.time()
            # print(f"for iteration time:  {(for_test_end_time - for_test_begin_time) * 1000} ms")
        
        self.max_draft_len = max_accept_len
        return remove_kvcache_index

    def update_inference_inputs(self,remove_kvcache_index=None):
        if remove_kvcache_index is not None:
            self.base_model.model.draft_move(self.cache_len[:self.batch_size], remove_kvcache_index)  # 移动Kvcahe
            self.finsh_index = []#完成请求的index
            self.eos_flag = False
            for i in range(self.batch_size):
                self.output_ids[i:i+1, self.output_len[i]:self.output_len[i] + self.acc_num[i] + 1] = self.acc_ids[i:i+1, :self.acc_num[i] +1 ]

                if self.eos_token_id in self.acc_ids[i:i+1, :self.acc_num[i] + 1] or self.cache_len[i]>=self.Max_gen_len-100:#防止Kvcahe溢出
                    self.finsh_index.append(i)
                    self.eos_flag = True

            #print(f"self.acc_num: {self.acc_num[:self.batch_size] + 1}")
            self.cache_len[:self.batch_size] += self.acc_num[:self.batch_size] + 1 #更新cachelen
            self.output_len[:self.batch_size] += self.acc_num[:self.batch_size] + 1 #更新outputlen
            self.count[:self.batch_size]+= self.acc_num[:self.batch_size] + 1
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

    def draft_produce(self, hidden_states, input_ids, exec_type="prefill", position_ids=None, \
                        len_posi=None, cache_len = None):  # produce draft sequence based on parallel decoding          
        if exec_type == 'prefill':
            tree_logits = self.ea_layer.topK_genrate(hidden_states,
                                                input_ids=input_ids,
                                                head=self.base_model.lm_head, logits_processor=self.logits_processor, batch_size=self.batch_size,
                                                batch_index=self.batch_index, cache_lens=self.cache_len[self.batch_index])
            self.tree_logits_list[self.batch_index] = tree_logits[0]
            self.tree_logits_prob_list[self.batch_index] = tree_logits[1]
        
        elif exec_type == 'tree_decoding':
            tree_logits = self.ea_layer.topK_genrate(hidden_states,
                                                input_ids=input_ids, batch_size=self.batch_size,
                                                head=self.base_model.lm_head, logits_processor=self.logits_processor, 
                                                accept_length=self.acc_num[:self.batch_size], exec_type=exec_type)
            self.tree_logits_list[:self.batch_size] = tree_logits[0]
            self.tree_logits_prob_list[:self.batch_size] = tree_logits[1]

    def draft_prune(self, sample_token):
        # generate_candidates_begin_time = time.time()
        cart_candidateslist, tree_candidateslist , tree_attn_masklist ,retrieve_indiceslist,node_num = generate_candidates(
                self.tree_logits_list[:self.batch_size],
                self.tree_logits_prob_list[:self.batch_size],
                self.tree_buffers["tree_indices"],
                self.tree_buffers["retrieve_indices"],
                sample_token,
                self.logits_processor,
                self.tree_buffers["node_index"],
            )  
        # torch.cuda.synchronize()
        # generate_candidates_end_time = time.time()
        # print(f"generate_candidates time: {(generate_candidates_end_time - generate_candidates_begin_time) * 1000} ms")
        self.Batch_node_allocate = node_num
        self.draft_qlen  = node_num
        self.pad_max_len  = max(self.draft_qlen)#记录最大的q_len,后面进行pad          
        batch_tree_candidates = []
        batch_tree_mask = []
        for i in range(self.batch_size):
            pad_tree_candidtates = torch.zeros((1,self.pad_max_len)).long()
            pad_tree_mask = self.tree_logits_list[:self.batch_size].new_ones((1,self.pad_max_len,self.pad_max_len))
            tree_candidates = tree_candidateslist[i] 
            medusa_attn_mask = tree_attn_masklist[i]
            pad_tree_candidtates[:,:tree_candidates.size(0)] = tree_candidates
            pad_tree_mask[:,:tree_candidates.size(0),:tree_candidates.size(0)] = medusa_attn_mask
            pad_tree_mask = torch.tril(pad_tree_mask)

            batch_tree_candidates.append(pad_tree_candidtates)
            batch_tree_mask.append(pad_tree_mask)


        batch_tree_candidates = torch.stack(batch_tree_candidates).squeeze(1)
        batch_tree_mask = torch.stack(batch_tree_mask).squeeze(1).squeeze(1)#[batch_size,seq_len,seq_len]
        batch_candidates_list = cart_candidateslist
        batch_retrieve_indices_list = retrieve_indiceslist

        return batch_candidates_list, batch_tree_candidates, batch_tree_mask, batch_retrieve_indices_list
    
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
                self.batch_size = sum(self.free_flag)
                # prefill LLM
                hidden_states, draft_input_ids, input_ids, token = self.prefill_target_model(input_ids,prefill_type="prompt_prefill",prompt_len=prompt_length)
                self.draft_produce(hidden_states, draft_input_ids, cache_len=prompt_length)
                self.tokens[self.batch_index] = token
                prefill_end_time = time.time()
            self.remove_check()#检测是否需要移动数据
            self.batch_size = sum(self.free_flag)
            hidden_states = self.Hidden_bank[:self.batch_size]
            # print(f"requests: {requests}")

            for out_index in range(0, self.Max_gen_len):
                # print(f"tree token: {token}")
                torch.cuda.synchronize()
                start_gerate_time = time.time()    
                batch_candidates_list, batch_tree_candidates, batch_tree_mask, batch_retrieve_indices_list= self.draft_prune(self.tokens[:self.batch_size])
                
                # torch.cuda.synchronize()
                # end_draft_time = time.time()
                # print(f"draft_prune time: {(end_draft_time - start_gerate_time) * 1000} ms")
                
                hidden_states_node =  self.prefill_target_model(batch_tree_candidates,
                                                        prefill_type="draft_prefill",
                                                        V_tree_mask=batch_tree_mask)
                torch.cuda.synchronize()
                end_target_time = time.time()
                llm_logits = self.lm_head(hidden_states_node)
                
                
                # print(f"draft produce logits: {logits}")
                verify_begin_time = time.time()
                remove_kvcache_index = \
                    self.verify(llm_logits, batch_candidates_list, batch_tree_candidates, batch_retrieve_indices_list, hidden_states_node)
                # torch.cuda.synchronize()
                # verify_end_time = time.time()
                # print(f"verify time: {(verify_end_time - verify_begin_time) * 1000} ms")
                                
                self.update_inference_inputs(remove_kvcache_index)

                self.draft_produce(self.draft_hidden[:self.batch_size, :self.max_draft_len, :], \
                    self.draft_input_ids[:self.batch_size, :self.max_draft_len + 1], exec_type="tree_decoding")

                torch.cuda.synchronize()
                end_generate_time = time.time()
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
                        self.batch_size = sum(self.free_flag)
                        # print(f"self.batch_size: {self.batch_size}, test self.free_flag: {self.free_flag}")
                    return finsh_request_list
                
                elif  self.insert_flag == True:#请求没完成但是有请求需要插入
                    # print(f"self.insert_flag: {self.insert_flag}")
                    self.Hidden_bank[:self.batch_size] = hidden_states.squeeze(1)
                    return None



