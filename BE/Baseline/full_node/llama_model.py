import math
from typing import List, Optional, Tuple, Union
import time
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import numpy as np
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
)
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaMLP,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
)

from flash_attn import flash_attn_func, flash_attn_with_kvcache
from triton_tree_attn import attention as tree_attention
if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-llama/Llama-2-7b-hf"
_CONFIG_FOR_DOC = "LlamaConfig"

def print_memory_usage(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1024**3  # 转换为GB
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[{prefix}] Current allocated: {allocated:.2f}GB, "
          f"Peak allocated: {max_allocated:.2f}GB, "
          f"Reserved: {reserved:.2f}GB")
    

class TensorCompressor:
    def __init__(self, input_tensor, draft_qlen):
        self.draft_qlen = draft_qlen
        self.bsz, self.q_len, self.hidden_size = input_tensor.size()
        self.device = input_tensor.device
        self.dtype = input_tensor.dtype
        self.cumulative_lengths = np.cumsum([0] + draft_qlen)
        self.total_valid = self.cumulative_lengths[-1]

    def compress(self, input_tensor):
        # 使用向量化拼接代替逐元素复制
        slices = []
        for i in range(self.bsz):
            valid_len = self.draft_qlen[i]
            if valid_len > 0:
                slices.append(input_tensor[i, :valid_len])
        if not slices:
            return torch.empty((0, self.hidden_size), device=self.device, dtype=self.dtype)
        return torch.cat(slices, dim=0)
    def restore(self, input_tensor):
        output = torch.zeros(
            (self.bsz * self.q_len, self.hidden_size),
            device=self.device,
            dtype=self.dtype
        )
        for i in range(self.bsz):
            valid_len = self.draft_qlen[i]
            if valid_len > 0:
                start = self.cumulative_lengths[i]
                end = self.cumulative_lengths[i+1]
                output[i*self.q_len : i*self.q_len+valid_len] = input_tensor[start:end]
        return output
    # def restore(self, input_tensor):
    #     padded_chunks = []
    #     for i in range(self.bsz):
    #         valid_len = self.draft_qlen[i]
    #         start = self.cumulative_lengths[i]
    #         end = self.cumulative_lengths[i+1]
    #         chunk = input_tensor[start:end]  # 取出当前样本的有效数据
    #         if valid_len < self.q_len:
    #             # 计算需要填充的长度并创建零张量
    #             pad_len = self.q_len - valid_len
    #             pad_tensor = torch.zeros((pad_len, self.hidden_size), 
    #                                 device=input_tensor.device, 
    #                                 dtype=input_tensor.dtype)
    #             # 在序列维度拼接填充张量
    #             chunk = torch.cat([chunk, pad_tensor], dim=0)
    #         padded_chunks.append(chunk)
    #     # 将所有填充后的块拼接成最终输出
    #     return torch.cat(padded_chunks, dim=0)
    
# class TensorCompressor:
#     def __init__(self, input_tensor, draft_qlen):
#         self.draft_qlen = draft_qlen
#         self.bsz, self.q_len,self.hidden_size = input_tensor.size()
#         self.device = input_tensor.device
#         self.dtype = input_tensor.dtype
#         self.cumulative_lengths = np.cumsum([0] + draft_qlen)
#         self.total_valid = self.cumulative_lengths[-1]

#     def compress(self, input_tensor):
#         compressed = torch.zeros(
#             (self.total_valid, self.hidden_size),
#             device=self.device,
#             dtype=self.dtype
#         )
#         for i in range(self.bsz):
#             valid_len = self.draft_qlen[i]
#             if valid_len > 0:
#                 start = self.cumulative_lengths[i]
#                 end = self.cumulative_lengths[i+1]
#                 compressed[start:end] = input_tensor[i, :valid_len]
#         return compressed

#     def restore(self, input_tensor):
#         output = torch.zeros(
#             (self.bsz * self.q_len, self.hidden_size),
#             device=self.device,
#             dtype=self.dtype
#         )
#         for i in range(self.bsz):
#             valid_len = self.draft_qlen[i]
#             if valid_len > 0:
#                 start = self.cumulative_lengths[i]
#                 end = self.cumulative_lengths[i+1]
#                 output[i*self.q_len : i*self.q_len+valid_len] = input_tensor[start:end]
#         return output
    
class KVCacheManager(nn.Module):
    def __init__(self, config, max_request_num, max_gen_len):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.max_request_num = max_request_num
        self.max_gen_len = max_gen_len
        
        # 合并所有层的缓存到单个Tensor (层数, 请求数, 生成长度, 头数, 头维度)
        self.K_cache = nn.Parameter(
            torch.zeros(
                (self.num_layers, max_request_num, max_gen_len,
                 config.num_key_value_heads, config.head_dim),
                dtype=torch.float16,
                device='cuda'
            ), requires_grad=False
        )
        self.V_cache = nn.Parameter(
            torch.zeros_like(self.K_cache),
            requires_grad=False
        )

    def find_adjacent(self,tensor):#检查是否有连续的序列
        if tensor.nelement() <= 1:
            return None
        start = None
        continuous_sequences = []

        for i in range(tensor.nelement() - 1):
            if tensor[i + 1] - tensor[i] == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    continuous_sequences.append([start, i])
                    start = None

        # 检查最后一个序列
        if start is not None:
            continuous_sequences.append([start, tensor.nelement() - 1])

        if continuous_sequences:
            return continuous_sequences
        else:
            return None
        
    def get_cache_view(self, layer_idx, batch_ids=None):
        """获取指定层的缓存视图，支持批次索引"""
        layer_K = self.K_cache[layer_idx]  # [max_request, seq, head, dim]
        layer_V = self.V_cache[layer_idx]
        return (layer_K if batch_ids is None else layer_K[batch_ids],
                layer_V if batch_ids is None else layer_V[batch_ids])

    def update_cache(self, layer_idx, new_K, new_V):
        # """原位更新指定层缓存"""
        self.K_cache.data[layer_idx].copy_(new_K.detach())
        self.V_cache.data[layer_idx].copy_(new_V.detach())
        # self.K_cache[layer_idx] = new_K.detach()
        # self.V_cache[layer_idx] = new_V.detach()


    def tokencache_move(self, cachelen, accept_indices_list):
        """
        压缩缓存，将接受的token移动到连续位置（批量处理版本）
        Args:
            cachelen: torch.Tensor [num_requests] 各请求当前有效缓存长度
            accept_indices_list: List[torch.Tensor] 各请求被接受的token相对偏移列表
        """
        num_requests = len(accept_indices_list)
        req_indices_list = []  # 记录请求ID
        src_pos_list = []      # 记录源位置
        tgt_pos_list = []      # 记录目标位置

        # 收集所有需要移动的请求的索引信息
        for req_id in range(num_requests):
            current_len = cachelen[req_id].item()
            accepted = accept_indices_list[req_id].squeeze(0).to(device='cuda')
            src_pos = current_len + accepted
            num_accept = src_pos.size(0)
            
            # 生成目标位置序列
            tgt_pos = torch.arange(
                current_len, 
                current_len + num_accept, 
                device='cuda', 
                dtype=torch.long
            )
            
            # 收集当前请求的索引信息
            req_indices_list.append(
                torch.full((num_accept,), req_id, device='cuda', dtype=torch.long)
            )
            src_pos_list.append(src_pos)
            tgt_pos_list.append(tgt_pos)

        # 合并所有请求的索引信息
        req_indices = torch.cat(req_indices_list)
        src_positions = torch.cat(src_pos_list)
        tgt_positions = torch.cat(tgt_pos_list)

        # 批量拷贝数据
        with torch.no_grad():
            src_k = self.K_cache.data[:, req_indices, src_positions, :, :].clone()
            src_v = self.V_cache.data[:, req_indices, src_positions, :, :].clone()
            
            # 批量写入目标位置
            self.K_cache.data[:, req_indices, tgt_positions, :, :] = src_k
            self.V_cache.data[:, req_indices, tgt_positions, :, :] = src_v

            
            # # 批量写入目标位置
            # self.K_cache.data[:, req_indices, tgt_positions, :, :] = self.K_cache.data[:, req_indices, src_positions, :, :]
            # self.V_cache.data[:, req_indices, tgt_positions, :, :] = self.V_cache.data[:, req_indices, src_positions, :, :]

    def batchcache_move(self, indices):
        # 解压索引对为源列表和目标列表
        sources, targets = zip(*indices) if indices else ((), ())
        
        if not sources:  # 如果没有索引，直接返回
            return
        
        # 转换为PyTorch张量
        sources_tensor = torch.tensor(sources, dtype=torch.long, device='cuda')
        targets_tensor = torch.tensor(targets, dtype=torch.long, device='cuda')
        
        # 转换为Python列表以便处理连续块
        sources_list = sources_tensor.cpu().tolist()
        targets_list = targets_tensor.cpu().tolist()
        
        n = len(sources_list)
        blocks = []
        i = 0
        
        # 识别所有连续的块
        while i < n:
            start = i
            j = i + 1
            # 扩展当前块直到不满足连续条件
            while j < n:
                if sources_list[j] == sources_list[j-1] + 1 and targets_list[j] == targets_list[j-1] + 1:
                    j += 1
                else:
                    break
            blocks.append((start, j - 1))
            i = j
        
        # 对每个块执行高效的切片操作
        for start_idx, end_idx in blocks:
            src_start = sources_list[start_idx]
            src_end = sources_list[end_idx] + 1  # 切片为左闭右开
            tgt_start = targets_list[start_idx]
            tgt_end = targets_list[end_idx] + 1
            
            # 执行切片复制
            self.K_cache[:, tgt_start:tgt_end] = self.K_cache[:, src_start:src_end]
            self.V_cache[:, tgt_start:tgt_end] = self.V_cache[:, src_start:src_end]

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # Modification here to support tree decoding
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        self.max_len = 512
        self.log_ratio = math.log(0.7)
        self.prefix_lens = None
        self.layer_idx = layer_idx
        self.last_layer = (config.num_hidden_layers == self.layer_idx + 1)
        self.softmax_scale = 1 / (128 ** 0.5)
        self.range_indices = None
        self.batch_size =None
        self.max_seq_len = None

    def forward(
        self,
        hidden_states,
        position_embeddings,
        cache_lens=None,
        flex_attn=None,
        tree_mask=None,
        exec_type=None,
        induction_head=False,
        Tensor_manager = None,
        kv_cache_manager =None,
        batch_index=None
    ):

        kv_cache = None
        if exec_type == "prompt_prefill":
            y = self.prefill(hidden_states, position_embeddings,kv_cache_manager,batch_index)
        elif exec_type == "draft_prefill":
            y = self.tree_decoding(hidden_states, position_embeddings, cache_lens, tree_mask,Tensor_manager=Tensor_manager, kv_cache_manager=kv_cache_manager)
        elif exec_type == "decoding":
            y = self.decoding(hidden_states, position_embeddings, cache_lens, kv_cache_manager)
        return y, kv_cache
     

    def prefill(
            self,
            hidden_states,
            position_embeddings,
            kv_cache_manager,
            batch_index
            ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        attn_output = flash_attn_func(query_states, key_states, value_states, causal=True)
        
        K_cache, V_cache = kv_cache_manager.get_cache_view(self.layer_idx)

        K_cache[batch_index, :q_len] = key_states
        V_cache[batch_index, :q_len] = value_states

        kv_cache_manager.update_cache(self.layer_idx, K_cache, V_cache)


        self.range_indices = torch.arange(1024, device=K_cache
                                          .device)
        attn_output = self.o_proj(attn_output.view(bsz, q_len, -1))
        return attn_output
    def triton_tree_part_fwd(self, query_states, key_states, value_states, tree_mask, prefix_lse, bsz, q_len):

        current_out, current_lse = tree_attention(
            query_states.permute(0, 2, 1, 3), 
            key_states.permute(0, 2, 1, 3), 
            value_states.permute(0, 2, 1, 3), 
            tree_mask
        )
        weight = torch.nn.functional.sigmoid(prefix_lse - current_lse)
        current_out = current_out.transpose(1, 2)
        weight = weight.transpose(1, 2).unsqueeze(-1)
        return current_out, weight, key_states,value_states
    
    def tree_decoding(
            self,
            hidden_states,
            position_embeddings,
            cache_lens,
            tree_mask=None,
            Tensor_manager=None,
            kv_cache_manager=None
            ):
        '''
        tree_mask: bsz fseq fseq (flatten_seqlen)
        '''

        query_states = Tensor_manager.restore(self.q_proj(hidden_states)).view((Tensor_manager.bsz, Tensor_manager.q_len, self.num_heads, self.head_dim))
        key_states = Tensor_manager.restore(self.k_proj(hidden_states)).view((Tensor_manager.bsz, Tensor_manager.q_len, self.num_heads, self.head_dim))
        value_states = Tensor_manager.restore(self.v_proj(hidden_states)).view((Tensor_manager.bsz, Tensor_manager.q_len, self.num_heads, self.head_dim))

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        K_cache, V_cache = kv_cache_manager.get_cache_view(self.layer_idx)

        prefix_o, prefix_lse = flash_attn_with_kvcache(query_states, K_cache[:Tensor_manager.bsz], V_cache[:Tensor_manager.bsz], cache_seqlens=cache_lens, return_softmax_lse=True)
        
        range_indices = cache_lens.unsqueeze(-1) + self.range_indices[:tree_mask.size(-1)].unsqueeze(0)  # 计算范围
        bsz_indices = self.range_indices[:Tensor_manager.bsz].unsqueeze(-1)
        K_cache[bsz_indices, range_indices] = key_states
        V_cache[bsz_indices, range_indices] = value_states
        kv_cache_manager.update_cache(self.layer_idx, K_cache, V_cache)
        
        current_out, weight,key_states,value_states = self.tree_part_fwd(query_states, key_states, value_states, tree_mask,prefix_lse, Tensor_manager.bsz, Tensor_manager.q_len)
        #current_out, weight,key_states,value_states = self.triton_tree_part_fwd(query_states, key_states, value_states, tree_mask,prefix_lse, Tensor_manager.bsz, Tensor_manager.q_len)
        attn_output = prefix_o * weight + current_out * (1 - weight)
        attn_output = attn_output.view(Tensor_manager.bsz, Tensor_manager.q_len, self.hidden_size).to(hidden_states.dtype)
        attn_output = Tensor_manager.compress(attn_output)
        attn_output = self.o_proj(attn_output)

        return attn_output
    
    def decoding(#自回归解码时候
            self,
            hidden_states,
            position_embeddings,
            cache_lens,
            kv_cache_manager
            ):

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)
        K_cache, V_cache = kv_cache_manager.get_cache_view(self.layer_idx)
        attn_output = flash_attn_with_kvcache(query_states, K_cache[:bsz], V_cache[:bsz], key_states, value_states, 
                                             causal=True, cache_seqlens=cache_lens)
        bsz_indices = self.range_indices[:bsz].unsqueeze(-1)
        range_indices = cache_lens.unsqueeze(-1) + 1
        K_cache[bsz_indices, range_indices] = key_states
        V_cache[bsz_indices, range_indices] = value_states
        kv_cache_manager.update_cache(self.layer_idx, K_cache, V_cache)
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


    @torch.compile
    def tree_part_fwd(self, query_states, key_states, value_states, tree_mask, prefix_lse, bsz, q_len):
        key_states = key_states.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=2)
        value_states = value_states.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=2)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.permute(0, 2, 3, 1)
        value_states = value_states.transpose(1, 2)
        if self.last_layer:
            attn_score = torch.matmul(query_states * self.softmax_scale, key_states)
        else:
            attn_score = torch.matmul(query_states, key_states) * self.softmax_scale
        attn_score = attn_score.to(torch.float16)
        attn_score_tree_mask = tree_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attn_score = attn_score.masked_fill(attn_score_tree_mask == 0, -float('inf'))
        attn_weight = torch.softmax(attn_score, dim=-1).to(query_states.dtype)
        current_out = torch.matmul(attn_weight, value_states).permute(0, 2, 1, 3)
        current_lse = attn_score.logsumexp(dim=-1, keepdim=True).transpose(1, 2)
        # if torch._dynamo.is_compiling():
        #     prefix_lse = prefix_lse.reshape(bsz, self.num_heads, q_len, -1).transpose(1, 2)
        #     print("here")
        # else:
        prefix_lse = prefix_lse.view(bsz, self.num_heads, q_len, -1).transpose(1, 2)
        weight = torch.nn.functional.sigmoid(prefix_lse - current_lse).to(query_states.dtype)
        return current_out, weight,key_states,value_states



LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaAttention,
    "sdpa": LlamaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        cache_lens=None,
        flex_attn=None,
        exec_type=None,
        tree_mask=None,
        induction_head=False,
        Tensor_manager = None,
        kv_cache_manager=None,
        batch_index=None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        if exec_type == "draft_prefill":
            
            residual = hidden_states

            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, kv_cache = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                cache_lens=cache_lens,
                flex_attn=flex_attn,
                exec_type=exec_type,
                tree_mask=tree_mask,
                induction_head=induction_head,
                Tensor_manager=Tensor_manager,
                kv_cache_manager=kv_cache_manager
            )
            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

            outputs = (hidden_states, kv_cache)

            return outputs
        elif exec_type == "prompt_prefill":
            residual = hidden_states

            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, kv_cache = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                cache_lens=cache_lens,
                flex_attn=flex_attn,
                exec_type=exec_type,
                tree_mask=tree_mask,
                induction_head=induction_head,
                kv_cache_manager=kv_cache_manager,
                batch_index=batch_index
            )
            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

            outputs = (hidden_states, kv_cache)

            return outputs
        elif exec_type == "decoding":#选择自回归解码
            residual = hidden_states

            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, kv_cache = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                cache_lens=cache_lens,
                flex_attn=flex_attn,
                exec_type=exec_type,
                tree_mask=tree_mask,
                induction_head=induction_head,
                kv_cache_manager=kv_cache_manager
            )
            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

            outputs = (hidden_states, kv_cache)

            return outputs




class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.kv_cache_manager = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def init_kv_cache(self, max_request_num,max_gen_len):
        """显存按需初始化"""
        if self.kv_cache_manager is None:
            self.kv_cache_manager = KVCacheManager(self.config, max_request_num,max_gen_len)
    def draft_move(self,cachelen, accept_indices_list):#草稿模型的接受token的移动

        self.kv_cache_manager.tokencache_move(cachelen,accept_indices_list)

    def batch_move(self,indices_list):
        self.kv_cache_manager.batchcache_move(indices_list)


    def forward(
        self,
        input_ids,
        position_ids=None,
        position_embeddings=None,
        inputs_embeds=None,
        cache_lens=None,
        flex_attn=None,
        exec_type=None,
        tree_mask=None,
        induction_head=False,
        draft_qlen=None,
        batch_index=None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if position_ids is None:
            if tree_mask is None:
                position_ids = torch.arange(0, input_ids.size(1))[None, :].to(input_ids.device)
            else:
                position_ids = tree_mask.sum(dim=-1) - 1

            if cache_lens is not None:
                position_ids = position_ids + cache_lens[:, None]
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        if position_embeddings is None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if exec_type == "draft_prefill":
            batch_size,q_len,hidden_size = hidden_states.size()
            TC = TensorCompressor(hidden_states,draft_qlen)#初始化
            hidden_states = TC.compress(hidden_states)
            for decoder_layer in self.layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings,
                    cache_lens,
                    flex_attn,
                    exec_type,
                    tree_mask,
                    induction_head,
                    Tensor_manager=TC,
                    kv_cache_manager=self.kv_cache_manager,
                )
                hidden_states = layer_outputs[0]

            hidden_states = self.norm(hidden_states)
            hidden_states = TC.restore(hidden_states).view(batch_size,q_len,hidden_size)#最后进行复原
        elif exec_type == "prompt_prefill":
            for decoder_layer in self.layers:

                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings,
                    cache_lens,
                    flex_attn,
                    exec_type,
                    tree_mask,
                    induction_head,
                    kv_cache_manager=self.kv_cache_manager,
                    batch_index=batch_index
                )

                
                hidden_states = layer_outputs[0]
            hidden_states = self.norm(hidden_states)

        elif exec_type == "decoding":
            for decoder_layer in self.layers:

                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings,
                    cache_lens,
                    flex_attn,
                    exec_type,
                    tree_mask,
                    induction_head,
                    kv_cache_manager=self.kv_cache_manager,
                )

                
                hidden_states = layer_outputs[0]
            hidden_states = self.norm(hidden_states)


        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None,
        )

class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def set_max_gen_len(self, max_gen_len):
        for layer in self.model.layers:
            layer.self_attn.max_len = max_gen_len

    def init_kv_cache(self,max_request_num,max_gen_len):
        self.model.init_kv_cache(max_request_num,max_gen_len)
    
    def draft_move(self,cachelen, accept_indices_list):#草稿模型的接受token的移动
        self.model.draft_move(cachelen,accept_indices_list)

    def batch_move(self,indices_list):
        self.model.batch_move(indices_list)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids,
        position_ids=None,
        position_embeddings=None,
        inputs_embeds=None,
        labels=None,
        cache_lens=None,
        exec_type=None,
        induction_head=False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids, 
            inputs_embeds=inputs_embeds,
            cache_lens=cache_lens,
            exec_type=exec_type,
            induction_head=induction_head,
        )

        hidden_states = outputs[0]
        last_kv = outputs[1]

        loss = None
        if labels is not None:
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            loss_fn = LigerFusedLinearCrossEntropyLoss()
            hidden_dim = hidden_states.size(-1)
            loss = loss_fn(self.lm_head.weight, hidden_states[:, 1:].reshape(-1, hidden_dim), labels[:, :-1].view(-1))
        else:
            logits = self.lm_head(hidden_states[:, -128:, :]).float()

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=last_kv,
            hidden_states=None,
            attentions=None,
        )
