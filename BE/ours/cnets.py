from typing import List, Optional, Tuple

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
import time

try:
    from .configs import EConfig
    from .utils_c import *
    from .choices import *
except:
    from configs import EConfig
    from utils_c import *
    from choices import *

from transformers.models.llama.modeling_llama import (
    rotate_half,
)

from flash_attn import flash_attn_func, flash_attn_with_kvcache
import numpy as np

top_k = 10


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
    
class KVCacheManager(nn.Module):
    def __init__(self, num_hidden_layers, num_key_value_heads, head_dim, max_request_num, max_gen_len):
        super().__init__()
        self.num_layers = num_hidden_layers
        self.max_request_num = max_request_num
        self.max_gen_len = max_gen_len
        
        # 合并所有层的缓存到单个Tensor (层数, 请求数, 生成长度, 头数, 头维度)
        self.K_cache = nn.Parameter(
            torch.zeros(
                (self.num_layers, max_request_num, max_gen_len,
                 num_key_value_heads, head_dim),
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

    def batchcache_move(self, indices):
        # 解压索引对为源列表和目标列表
        sources, targets = zip(*indices) if indices else ((), ())
        # print(f"batchcache_move sources: {sources} targets: {targets}")
        
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
            # print(f"tgt_start:tgt_end: {tgt_start}:{tgt_end}, src_start:src_end: {src_start}:{src_end}")
            self.K_cache[:, tgt_start:tgt_end] = self.K_cache[:, src_start:src_end].clone()
            self.V_cache[:, tgt_start:tgt_end] = self.V_cache[:, src_start:src_end].clone()

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    # print(f"cos: {cos.shape}, sin: {sin.shape}, q: {q.shape}, k: {k.shape}")
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    # print(f"cos: {cos.shape}, sin: {sin.shape}, q: {q.shape}, k: {k.shape}")
    cos = cos[position_ids].unsqueeze(2)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(2)  # [bs, 1, seq_len, dim]
    # print(f"cos: {cos.shape}, sin: {sin.shape}, q: {q.shape}, k: {k.shape}")
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # print(f"seq_len: {seq_len}, self.max_seq_len_cached: {self.max_seq_len_cached}")
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        self.softmax_scale = 1.0 / (self.head_dim ** 0.5)
        
        self.layer_idx = 0

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(
        self,
        hidden_states,
        position_embeddings,
        cache_lens=None,
        tree_mask=None,
        batch_size=None,
        q_len=None,
        exec_type=None,
        past_key_values=None,
        position_ids=None,
        kv_cache_manager =None,
        batch_index=None,
        update_cache=False,
    ):

        kv_cache = None
        if exec_type == "prefill":
            y, kv_cache = self.prefill(hidden_states, position_embeddings, position_ids, kv_cache_manager, batch_index)
        elif exec_type == "tree_decoding":
            y, kv_cache = self.tree_decoding(hidden_states, position_embeddings, cache_lens,  batch_size, q_len, tree_mask,\
                        position_ids=position_ids, past_key_values=past_key_values, kv_cache_manager=kv_cache_manager, update_cache=update_cache)
        elif exec_type == "decoding":
            y, kv_cache = self.decoding(hidden_states, position_embeddings, cache_lens, position_ids=position_ids, \
                kv_cache_manager=kv_cache_manager, update_cache=update_cache)
        return y, kv_cache
     

    def prefill(
            self,
            hidden_states,
            position_embeddings,
            position_ids,
            kv_cache_manager,
            batch_index
            ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # print(f"test query_states: {query_states}")

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        # print(f"cos: {cos.shape}, sin: {sin.shape}, query_states: {query_states.shape}, key_states: {key_states.shape}")
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        attn_output = flash_attn_func(query_states, key_states, value_states, causal=True)
        
        K_cache, V_cache = kv_cache_manager.get_cache_view(self.layer_idx)

        K_cache[batch_index, :q_len] = key_states
        V_cache[batch_index, :q_len] = value_states

        kv_cache_manager.update_cache(self.layer_idx, K_cache, V_cache)

        self.range_indices = torch.arange(1024, device=K_cache.device)
        
        attn_output = self.o_proj(attn_output.view(bsz, q_len, -1))
        
        return attn_output, None
    
    def tree_decoding(
            self,
            hidden_states,
            position_embeddings,
            cache_lens,
            batch_size = None,
            q_len = None,
            tree_mask=None,
            position_ids=None,
            past_key_values=None,
            kv_cache_manager=None,
            update_cache=False,
            ):
        '''
        tree_mask: bsz fseq fseq (flatten_seqlen)
        '''

        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        K_cache, V_cache = kv_cache_manager.get_cache_view(self.layer_idx)

        prefix_o, prefix_lse = flash_attn_with_kvcache(query_states, K_cache[:batch_size], V_cache[:batch_size], cache_seqlens=cache_lens, return_softmax_lse=True)
        # prefix_o prefix_lse 这一块可以考虑优化一下,因为cache_lens都是一致的,这一部份的计算似乎是重复的,可以提出来.  
        # 哈哈哈,想错了,query_states是不同的, 所以不能提出来
        
        if past_key_values is not None:
            # print(f"past_key_values: {past_key_values}")
            # print(f"past_key_values: {past_key_values[0].shape}, past_key_values: {past_key_values[1].shape}")
            # print(f"key_states: {key_states.shape}, value_states: {value_states.shape}")
            key_states = torch.cat([past_key_values[0], key_states], dim=1)
            value_states = torch.cat([past_key_values[1], value_states], dim=1)
        else:
            key_states = key_states
            value_states = value_states
        
        current_out, weight = self.tree_part_fwd(query_states, key_states, value_states, tree_mask, prefix_lse, batch_size, q_len)
        #current_out, weight,key_states,value_states = self.triton_tree_part_fwd(query_states, key_states, value_states, tree_mask,prefix_lse, batch_size, Tensor_manager.q_len)
        attn_output = prefix_o * weight + current_out * (1 - weight)
        attn_output = attn_output.view(batch_size, q_len, self.hidden_size).to(hidden_states.dtype)
        attn_output = self.o_proj(attn_output)
        # print(f"return key_states: {key_states.shape}, value_states: {value_states.shape}")
        if update_cache:
            bsz_indices = self.range_indices[:batch_size].unsqueeze(-1)
            # print(f"cache_lens: {cache_lens}, self.range_indices: {self.range_indices[:tree_mask.size(-1)]}, tree_mask: {tree_mask.size(-1)}")
            range_indices = cache_lens.unsqueeze(-1) + self.range_indices[:tree_mask.size(-1)].unsqueeze(0)
            # print(f"bsz_indices: {bsz_indices}, range_indices: {range_indices} key_states: {key_states.shape}")
            K_cache[bsz_indices, range_indices] = key_states
            V_cache[bsz_indices, range_indices] = value_states
            kv_cache_manager.update_cache(self.layer_idx, K_cache, V_cache)  ## 这里进一步明确一下更新问题
            return attn_output, None

        return attn_output, (key_states, value_states)
    
    def decoding(  #自回归解码时候
            self,
            hidden_states,
            position_embeddings,
            cache_lens,
            position_ids=None,
            kv_cache_manager=None
            ):

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
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


    # @torch.compile
    def tree_part_fwd(self, query_states, key_states, value_states, tree_mask, prefix_lse, bsz, q_len):
        # print(f"tree mask: {tree_mask.shape}, query_states: {query_states.shape}, key_states: {key_states.shape}, value_states: {value_states.shape}")
        key_states = key_states.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=2)
        value_states = value_states.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=2)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.permute(0, 2, 3, 1)
        value_states = value_states.transpose(1, 2)
        attn_score = torch.matmul(query_states, key_states) * self.softmax_scale
        attn_score = attn_score.to(torch.float16)
        attn_score_tree_mask = tree_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attn_score = attn_score.masked_fill(attn_score_tree_mask == 0, -float('inf'))
        attn_weight = torch.softmax(attn_score, dim=-1).to(query_states.dtype)
        current_out = torch.matmul(attn_weight, value_states).permute(0, 2, 1, 3)
        current_lse = attn_score.logsumexp(dim=-1, keepdim=True).transpose(1, 2)
        prefix_lse = prefix_lse.view(bsz, self.num_heads, q_len, -1).transpose(1, 2)
        weight = torch.nn.functional.sigmoid(prefix_lse - current_lse).to(query_states.dtype)
        return current_out, weight

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.index = index
        if self.index != 0:
            self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        cache_lens=None,
        batch_size=None,
        q_len=None,
        exec_type=None,
        tree_mask=None,
        position_ids=None,
        past_key_values=None,
        kv_cache_manager=None,
        batch_index=None,
        update_cache=False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
            
        # print(f"test hidden_states: {hidden_states}")
        
        if exec_type == "tree_decoding":
            
            # decoder_begin_time = time.time()
            # Self Attention
            # attn_begin_time = time.time()
            hidden_states, kv_cache = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                cache_lens=cache_lens,
                batch_size=batch_size,
                q_len=q_len,
                tree_mask=tree_mask,
                exec_type=exec_type,
                position_ids=position_ids,
                past_key_values=past_key_values,
                kv_cache_manager=kv_cache_manager,
                update_cache=update_cache,
            )
            # torch.cuda.synchronize()
            # attn_end_time = time.time()
            hidden_states = residual + hidden_states

            # Fully Connected
            # full_begin_time = time.time()
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            # torch.cuda.synchronize()
            # full_end_time = time.time()

            outputs = (hidden_states, kv_cache)
            # torch.cuda.synchronize()
            # decoder_end_time = time.time()
            # print(f"shape: {hidden_states.shape} decoder time: {(decoder_end_time - decoder_begin_time) * 1000}")
            # print(f'attn time: {(attn_end_time - attn_begin_time) *1000}ms, full time: {(full_end_time- full_begin_time)*1000}ms')

            return outputs
        elif exec_type == "prefill":
            
            # Self Attention
            hidden_states, kv_cache = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                cache_lens=cache_lens,
                batch_size=batch_size,
                q_len=q_len,
                exec_type=exec_type,
                tree_mask=tree_mask,
                position_ids=position_ids,
                kv_cache_manager=kv_cache_manager,
                batch_index=batch_index,
                update_cache=update_cache,
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
            
            # Self Attention
            hidden_states, kv_cache = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                cache_lens=cache_lens,
                batch_size=batch_size,
                q_len=q_len,
                exec_type=exec_type,
                tree_mask=tree_mask,
                position_ids=position_ids,
                kv_cache_manager=kv_cache_manager,
                past_key_values=past_key_values,
                update_cache=update_cache,
            )
            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

            outputs = (hidden_states, kv_cache)

            return outputs

class Model(nn.Module):
    def __init__(self, config, load_emb=False, path=None):
        super().__init__()

        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList([LlamaDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        self.kv_cache_manager = None
        
        
        self.cache_lens = None
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
        


        self.count = None

    def init_tree(self, bs=1):
        self.tree = mc_sim_7b_63
        #self.tree_buffer = generate_tree_buffers(self.tree, self.embed_tokens.weight.device, bs=bs)
        self.tree_buffer = generate_tree_buffers(self.tree, self.embed_tokens.weight.device)
        print(f"self.tree_buffer: {self.tree_buffer}")

    def reset(self):
        self.tree_mask = None
        
    def para_init(self):
        pass
    
    def forward(
            self,
            hidden_states,
            input_ids,
            position_ids=None,
            position_embeddings=None,
            cache_lens=None,
            exec_type=None,
            tree_mask=None,
            past_key_values=None,
            draft_qlen=None,
            batch_index=None,
            update_cache=False,
    ):
        with torch.no_grad():
            input_ids[input_ids == -1] = 0
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        # print(f"inputs_embeds: {inputs_embeds.shape}, hidden_states: {hidden_states.shape}, input_ids: {input_ids.shape}")
        hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))
        # print(f"before position_ids: {position_ids}")
        if position_ids is None:
            if tree_mask is None:
                position_ids = torch.arange(0, input_ids.size(1))[None, :].to(input_ids.device)
            else:
                position_ids = tree_mask.sum(dim=-1) - 1

            if cache_lens is not None:
                position_ids = position_ids + cache_lens[:, None]

        # print(f"after position_ids: {position_ids}")
        # print(f"cache_lens: {cache_lens}, position_ids: {position_ids}")
        next_decoder_cache = None
        if exec_type == "tree_decoding":
            batch_size, q_len, hidden_size = hidden_states.size()
            for decoder_layer in self.layers:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    cache_lens=cache_lens,
                    batch_size=batch_size,
                    q_len=q_len,
                    exec_type=exec_type,
                    tree_mask=tree_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    kv_cache_manager=self.kv_cache_manager,
                    update_cache=update_cache,
                )
                hidden_states = layer_outputs[0]
                next_decoder_cache = layer_outputs[1]

        elif exec_type == "prefill":
            for decoder_layer in self.layers:

                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    cache_lens=cache_lens,
                    exec_type=exec_type,
                    tree_mask=tree_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    kv_cache_manager=self.kv_cache_manager,
                    batch_index=batch_index
                )
                hidden_states = layer_outputs[0]
                next_decoder_cache = layer_outputs[1]

        elif exec_type == "decoding":
            for decoder_layer in self.layers:

                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    cache_lens=cache_lens,
                    batch_size=batch_size,
                    q_len=q_len,
                    exec_type=exec_type,
                    tree_mask=tree_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    kv_cache_manager=self.kv_cache_manager,
                    update_cache=update_cache,
                )
                
                hidden_states = layer_outputs[0]
                next_decoder_cache = layer_outputs[1]

        return hidden_states, next_decoder_cache

    def reset_kv(self, max_request_num=15, max_cache_len=1024):
        self.stable_kv = None
        self.init_kv_cache(1, 32, 128, max_request_num, 1024)
        self.draft_cache_lens = torch.zeros((max_request_num)).int().to("cuda")
        
    def init_kv_cache(self, num_hidden_layers, num_key_value_heads, head_dim, max_request_num, max_gen_len):
        """显存按需初始化"""
        if self.kv_cache_manager is None:
            self.kv_cache_manager = KVCacheManager(num_hidden_layers, num_key_value_heads, head_dim, max_request_num, max_gen_len)
            self.position_embeddings = self.rotary_emb(self.kv_cache_manager.K_cache, seq_len=1024) # 用于位置编码，直接给最大的就好，后面他们会根据position_ids进行取值
            
    def draft_move(self,cachelen, accept_indices_list):#草稿模型的接受token的移动
        self.kv_cache_manager.tokencache_move(cachelen,accept_indices_list)
    
    def batch_move(self,indices_list):
        self.kv_cache_manager.batchcache_move(indices_list)

    @torch.no_grad()
    def repeat_hidden(self, hidden_state, repeat_num):
        new_hidden = []
        for id, i in enumerate(repeat_num):
            new_hidden.append(hidden_state[:, id:id + 1].repeat(1, i, 1))
        return torch.cat(new_hidden, dim=1)

    def sample(self, logits, logits_processor, k=1, replacement=False):
        bs, seq_len, _ = logits.shape
        logits = logits.view(-1, logits.shape[-1])
        logits = logits_processor(None, logits)

        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        sampled_indices = torch.multinomial(probabilities, k, replacement=False)
        sampled_probs = torch.gather(probabilities, -1, sampled_indices)

        cumulative_sum = torch.cumsum(sampled_probs, dim=-1)
        cumulative_sum = torch.cat(
            (torch.zeros(cumulative_sum.shape[0], 1, device=cumulative_sum.device), cumulative_sum[:, :-1]), dim=-1)

        sampled_probs = sampled_probs / (1 - cumulative_sum)
        sampled_probs[torch.isinf(sampled_probs)] = -1
        sampled_probs[torch.isnan(sampled_probs)] = -1

        sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)

        sampled_indices = sampled_indices.view(bs, seq_len, -1)
        sampled_probs = sampled_probs.view(bs, seq_len, -1)
        probabilities = probabilities.view(bs, seq_len, -1)

        return sampled_indices, sampled_probs, probabilities

    def generate_tree_mask(self, input_ids, batch_size, device, valid_lengths):
        """
        Generate a tree_mask for tree decoding based on input_ids length and valid_lengths.
        Args:
            input_ids: The input IDs tensor.
            batch_size: The batch size for the current input.
            device: The device to place the generated mask on.
            valid_lengths: A tensor indicating the valid lengths for each request in the batch.
        Returns:
            A tensor representing the tree_mask.
        """
        seq_len = input_ids.size(1)
        # 创建基础的下三角掩码
        tree_mask = torch.tril(torch.ones((batch_size, seq_len, seq_len), device=device, dtype=torch.float32))
        # print(f"valid_len: {valid_lengths}, seq_len: {seq_len}")
        # 对于每个样本，将超出有效长度的位置设置为0
        for i, valid_len in enumerate(valid_lengths + 1):
            if valid_len < seq_len:
                # 将超出有效长度的位置掩码设为0
                tree_mask[i, :, valid_len:] = 0  # 屏蔽超出有效长度的列
                tree_mask[i, valid_len:, :] = 0  # 屏蔽超出有效长度的行

        return tree_mask
    
    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, head, logits_processor, position_ids=None, cache_lens=None, batch_size=None,\
                            batch_index=None, max_length=4,  use_cache=True, accept_length = None, len_posi=None, exec_type="prefill"):
        # begin_time = time.time()
        
        bs = input_ids.shape[0]
        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)

        if accept_length is not None:
            tree_mask = self.generate_tree_mask(input_ids, bs, hidden_states.device, accept_length)

        ss_token, ss_prob, ss_op = [], [], []

        self.reset()
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print(f"draft move time: {(end_time - begin_time) * 1000}ms")
        if exec_type == "prefill":
            # print(f"prefill self.draft_cache_lens: {self.draft_cache_lens}")
            self.draft_cache_lens[batch_index] = cache_lens
            # print(f"after prefill self.draft_cache_lens: {self.draft_cache_lens}")
            position_ids = torch.arange(input_ids.size(1), device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(bs, -1)

            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, position_ids=position_ids,
                        position_embeddings=self.position_embeddings, cache_lens=cache_lens, exec_type=exec_type, batch_index=batch_index)
            
            draft_cache_lens = self.draft_cache_lens[batch_index]

        else:
            # print(f"tree_decoding self.draft_cache_lens: {self.draft_cache_lens}")
            self.draft_cache_lens[:batch_size] = self.draft_cache_lens[:batch_size] + accept_length + 1
            # print(f"after tree_decoding self.draft_cache_lens: {self.draft_cache_lens}")
            position_ids = self.draft_cache_lens[:batch_size][:, None]
            out_hidden, past_key_values = self(hidden_states=hidden_states, 
                                                input_ids=input_ids, 
                                                position_ids=position_ids, 
                                                position_embeddings=self.position_embeddings,
                                                cache_lens=self.draft_cache_lens[:batch_size],
                                                exec_type="tree_decoding",
                                                tree_mask=tree_mask,
                                                past_key_values=None,
                                                batch_index=None,
                                                update_cache=True,
                                                )
            draft_cache_lens = self.draft_cache_lens[:batch_size]

        # torch.cuda.synchronize()
        # prefill_end_time = time.time()
        # print(f"prefill time: {(prefill_end_time - end_time) * 1000}ms")
        
        if accept_length is None:
            last_nopadding = draft_cache_lens - 1
        else:
            last_nopadding=accept_length
        
        ab=tuple(range(bs))
        

        out_hidden = out_hidden[ab,last_nopadding][:,None]
        last_headout = head(out_hidden)
        # torch.cuda.synchronize()
        # head_end_time = time.time()
        for i in range(len(self.tree_buffer['tree_indices'])):
            if logits_processor is not None:
                topk_index, topk_prob, op = self.sample(last_headout, logits_processor, k=top_k, )
            else:
                probs = torch.softmax(last_headout, dim=-1)
                topk_prob, topk_index = torch.topk(probs, top_k, dim=-1)  # 在概率上取topk

                op = None

            ss_token.append(topk_index)
            ss_prob.append(topk_prob)
            ss_op.append(op)
            input_ids = topk_index.view(bs,-1)[:, self.tree_buffer['tree_indices'][i]]

            hidden_states = out_hidden
            hidden_states = self.repeat_hidden(hidden_states, self.tree_buffer["repeat_nums"][i])
            self.tree_mask = self.tree_buffer['attn_mask'][i].squeeze(1).squeeze(0).expand(bs, -1, -1)
            position_ids = draft_cache_lens[:, None] + self.tree_buffer["position_ids"][i][None,:]
            # torch.cuda.synchronize()
            # self_begin_time = time.time()
            out_hidden, past_key_values = self(hidden_states=hidden_states, 
                                                input_ids=input_ids, 
                                                position_ids=position_ids, 
                                                position_embeddings=self.position_embeddings,
                                                cache_lens=draft_cache_lens,
                                                exec_type="tree_decoding",
                                                tree_mask=self.tree_mask,
                                                past_key_values=past_key_values,
                                                batch_index=None,
                                                )
            # torch.cuda.synchronize()
            # self_end_time = time.time()
            # print(f"self time: {(self_end_time - self_begin_time) * 1000}ms")            

            last_headout = head(out_hidden)
        # torch.cuda.synchronize()
        # forward_end_time = time.time()
        # print(f"forward time: {(forward_end_time - head_end_time) * 1000}ms")
        if logits_processor is not None:
            topk_index, topk_prob, op = self.sample(last_headout, logits_processor, k=top_k, )
        else:
            probs = torch.softmax(last_headout, dim=-1)
            topk_prob, topk_index = torch.topk(probs, top_k, dim=-1)  # 在概率上取topk
            op = None
        ss_token.append(topk_index)
        ss_prob.append(topk_prob)
        ss_op.append(op)

        return (torch.cat(ss_token,dim=1), torch.cat(ss_prob,dim=1), ss_op)