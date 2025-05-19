import copy
import random

# typing
from typing import List, Tuple
import time
import torch
import itertools

# TODO
# from transformers import LlamaTokenizer
# tokenizer=LlamaTokenizer.from_pretrained("/home/lyh/weights/hf/vicuna_v13/7B/")

TOPK = 10  # topk for sparse tree

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


class TensorCompressor:
    def __init__(self, input_tensor, draft_qlen):
        self.draft_qlen = draft_qlen
        self.bsz, self.q_len,self.hidden_size = input_tensor.size()
        self.device = input_tensor.device
        self.dtype = input_tensor.dtype
        self.cumulative_lengths = np.cumsum([0] + draft_qlen)
        self.total_valid = self.cumulative_lengths[-1]

    def compress(self, input_tensor):
        compressed = torch.zeros(
            (self.total_valid, self.hidden_size),
            device=self.device,
            dtype=self.dtype
        )
        for i in range(self.bsz):
            valid_len = self.draft_qlen[i]
            if valid_len > 0:
                start = self.cumulative_lengths[i]
                end = self.cumulative_lengths[i+1]
                compressed[start:end] = input_tensor[i, :valid_len]
        return compressed

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


def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
        return processor_list


# test_processor = prepare_logits_processor(
#         0.0, 0.0, -1, 1
#     )


def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def generate_tree_buffers(tree_choices, device="cuda"):
    sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
    tree_len = len(sorted_tree_choices) + 1
    
    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_tree_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth

    tree_attn_mask = torch.eye(tree_len, tree_len)
    tree_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            # retrieve ancestor position
            if len(cur_tree_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_tree_choice) - 1):
                ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
            tree_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    tree_indices = torch.zeros(tree_len, dtype=torch.long)
    p_indices = [0 for _ in range(tree_len - 1)]
    b_indices = [[] for _ in range(tree_len - 1)]
    tree_indices[0] = 0
    start = 0
    bias = 0
    for i in range(len(depth_counts)):
        inlayer_bias = 0
        b = []
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            cur_parent = cur_tree_choice[:-1]
            if j != 0:
                if cur_parent != parent:
                    bias += 1
                    inlayer_bias += 1
                    parent = cur_parent
                    b = []
            else:
                parent = cur_parent
            tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
            p_indices[start + j] = inlayer_bias
            if len(b) > 0:
                b_indices[start + j] = copy.deepcopy(b)
            else:
                b_indices[start + j] = []
            b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
        start += depth_counts[i]

    p_indices = [-1] + p_indices
    tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_tree_choices)):
        cur_tree_choice = sorted_tree_choices[-i - 1]
        retrieve_indice = []
        if cur_tree_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_tree_choice)):
                retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                retrieve_paths.append(cur_tree_choice[:c + 1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                 dim=1)

    maxitem = retrieve_indices.max().item() + 5

    def custom_sort(lst):
        # sort_keys=[len(list)]
        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys

    retrieve_indices = retrieve_indices.tolist()
    retrieve_indices = sorted(retrieve_indices, key=custom_sort)
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    
    # add node_index
    coordinates=[]
    for number in range(tree_len):
        positions = torch.where(retrieve_indices == number)
        if positions[0].numel() > 0:
            coordinates.append([positions[0][0].item(), positions[1][0].item()])
    node_index = torch.tensor(coordinates, dtype=torch.long)

    p_indices = torch.tensor(p_indices)
    p_indices_new = p_indices[retrieve_indices]
    p_indices_new = p_indices_new.tolist()

    b_indices = [[]] + b_indices
    b_indices_new = []
    for ib in range(retrieve_indices.shape[0]):
        iblist = []
        for jb in range(retrieve_indices.shape[1]):
            index = retrieve_indices[ib, jb]
            if index == -1:
                iblist.append([])
            else:
                b = b_indices[index]
                if len(b) > 0:
                    bt = []
                    for bi in b:
                        bt.append(torch.where(tree_indices == bi)[0].item())
                    iblist.append(torch.tensor(bt, device=device))
                else:
                    iblist.append(b)
        b_indices_new.append(iblist)

    # Aggregate the generated buffers into a dictionary
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in tree_buffers.items()
    }
    tree_buffers["p_indices"] = p_indices_new
    tree_buffers["b_indices"] = b_indices_new
    tree_buffers["node_index"] = node_index
    return tree_buffers


def initialize_tree(input_ids, model, tree_attn_mask, past_key_values, attention_mask=None, tree_buffers=None):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    tree_logits, outputs, logits, hidden_state, sample_token = model(
        input_ids, past_key_values=past_key_values, output_orig=True,
        attention_mask=attention_mask, position_ids=position_ids, tree_buffers=tree_buffers, exec_type="prefill"
    )
    return tree_logits, logits, hidden_state, sample_token

medusa_choices_init = [[0],[1],[2],[3],[0,0],[0,1],[0,2],[1,0],[1,1],[2,0],[2,1],[3,0]
                        ,[0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,1],[0,2,0],[0,2,1],[1,0,0],
                        [0,0,0,0],[0,0,0,1],[0,0,0,2],[0,0,0,0,0],[0,0,0,0,1]]

def generate_medusa_buffers(
        slect_node_index,
        tree_candidates
    ):

    slect_node_index = sorted(slect_node_index)
    tree_candidates =tree_candidates[:,slect_node_index]
    

    medusa_choices = [medusa_choices_init[i-1] for i in slect_node_index if i > 0]
    sorted_medusa_choices =medusa_choices
    medusa_len = len(sorted_medusa_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_medusa_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    
    # Create the attention mask for Medusa
    medusa_attn_mask = torch.eye(medusa_len, medusa_len)
    medusa_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            # retrieve ancestor position
            if len(cur_medusa_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_medusa_choice) - 1):
                ancestor_idx.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]) + 1)
            medusa_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    # Generate tree indices for the Medusa structure
    medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long)
    medusa_tree_indices[0] = 0
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            medusa_tree_indices[start + j + 1] = cur_medusa_choice[-1] + TOPK * i + 1
        start += depth_counts[i]

    # Generate retrieval indices for Medusa structure verification
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_medusa_choices)):
        cur_medusa_choice = sorted_medusa_choices[-i-1]
        retrieve_indice = []
        if cur_medusa_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_medusa_choice)):
                retrieve_indice.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]))
                retrieve_paths.append(cur_medusa_choice[:c+1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = 5
    retrieve_indices = [pad_path(path, max_length,-2) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1)
    pad_tree_candidates =  torch.cat([tree_candidates, torch.zeros((1,1), dtype=torch.long, device=tree_candidates.device)], dim=1)

    cart_candidates  = pad_tree_candidates[:,retrieve_indices]

    return  cart_candidates, tree_candidates , medusa_attn_mask ,retrieve_indices

scale_vector = torch.tensor([1, 1, 1, 1, 1, 1]).to(torch.float16).to("cuda:0")
# scale_vector = torch.tensor([1, 1.05, 1.1, 1.15, 1.2, 1.25]).to(torch.float16).to("cuda:0")
# def generate_candidates(tree_logits, candidates_tree_prob, tree_indices, retrieve_indices, sample_token, logits_processor, node_index=None, Node_num = 256):
#     # torch.cuda.synchronize()
#     # tree_candidate_begin_time = time.time()
#     bs = sample_token.shape[0]
#     sample_token = sample_token.to(tree_indices.device)

#     # candidates_logit = sample_token[0]
#     candidates_logit = sample_token

#     candidates_tree_logits = tree_logits

#     candidates = torch.cat([candidates_logit, candidates_tree_logits.view(bs, -1)], dim=-1)

#     tree_candidates = candidates[:, tree_indices]

#     tree_candidates_ext = torch.cat(
#         [tree_candidates, torch.zeros((bs, 1), dtype=torch.long, device=tree_candidates.device)-1], dim=-1)

#     cart_candidates = tree_candidates_ext[:, retrieve_indices]
    
#     candidates_tree_prob = candidates_tree_prob
#     candidates_prob = torch.cat(
#         [torch.ones((bs, 1), device=candidates_tree_prob.device, dtype=torch.float32),
#             candidates_tree_prob.view(bs, -1)],
#         dim=-1)

#     tree_candidates_prob = candidates_prob[:, tree_indices]
#     tree_candidates_prob_ext = torch.cat(
#         [tree_candidates_prob, torch.ones((bs, 1), dtype=torch.float32, device=tree_candidates_prob.device)],
#         dim=-1)
#     cart_candidates_prob = tree_candidates_prob_ext[:, retrieve_indices]
#     cart_candidates_probs_after = torch.cumprod(cart_candidates_prob, dim=-1)
#     cart_candidates_probs_after = cart_candidates_probs_after * scale_vector
    
#     Node_probs =  cart_candidates_probs_after[:, node_index[:, 0], node_index[:, 1]]
    
    
#     Freeze_Node =[]
#     Freeze_Node_num =0
#     Freeze_mask =torch.zeros_like(Node_probs, dtype=torch.bool)
#     bsz = tree_logits.size(0)
#     # torch.cuda.synchronize()
#     # tree_candidate_end_time = time.time()
#     # print(f"tree_candidate time: {(tree_candidate_end_time - tree_candidate_begin_time) * 1000} ms")
#     # node_produce_begin_time = time.time()
#     for i in range(bsz):#计算每个请求冻结的node 
#         Nodelist= [[0, 1], [0, 2]]
#         Nodelist = list(set(element for sublist in Nodelist for element in sublist))
#         Freeze_Node.append(Nodelist)
#         Freeze_mask[i,Nodelist] = True
#         Freeze_Node_num +=len(Nodelist)

#     #更新实际分配的Token数

#     if bsz*26 <= Node_num:#防止越界,给最多的Node数
#         Node_num =26*bsz
    
#     if Node_num - Freeze_Node_num > 3:#超出直接给冻结的Node即可
#         Prune_Node = Node_num - Freeze_Node_num
#         Node_probs[Freeze_mask] = float('-inf')
#         batch_size, num_nodes = Node_probs.shape
        
#         topk_values, topk_flat_indices = torch.topk(Node_probs.view(-1), Prune_Node)

#         topk_batch_indices = topk_flat_indices // num_nodes
#         topk_node_indices = topk_flat_indices % num_nodes
#         for i,j in zip(topk_batch_indices.tolist(), topk_node_indices.tolist()):
#             Freeze_Node[i]=Freeze_Node[i]+[j]
#     # torch.cuda.synchronize()
#     # node_produce_end_time = time.time()
#     # print(f"node_produce time: {(node_produce_end_time - node_produce_begin_time) * 1000} ms")
#     Node_num = []
#     cart_candidateslist=[] 
#     tree_candidateslist=[] 
#     tree_attn_masklist =[]
#     retrieve_indiceslist=[]
#     generate_medusa_buffers_begin_time = time.time()
#     for i in range(bsz):
#         # generate_medusa_buffers_single_begin_time = time.time()
#         cart_candidates, tree_candidates_1, medusa_attn_mask ,retrieve_indices=generate_medusa_buffers(Freeze_Node[i],tree_candidates[i:i+1])
#         # torch.cuda.synchronize()
#         # generate_medusa_buffers_single_end_time = time.time()
#         # print(f"generate_medusa_buffers_single time: {(generate_medusa_buffers_single_end_time - generate_medusa_buffers_single_begin_time) * 1000} ms")
#         cart_candidateslist.append(cart_candidates.squeeze(0))
#         tree_candidateslist.append(tree_candidates_1.squeeze(0))
#         tree_attn_masklist.append(medusa_attn_mask)
#         retrieve_indiceslist.append(retrieve_indices)
#         Node_num.append(tree_candidates_1.size(1))
#     # torch.cuda.synchronize()
#     # generate_medusa_buffers_end_time = time.time()
#     # print(f"generate_medusa_buffers time: {(generate_medusa_buffers_end_time - generate_medusa_buffers_begin_time) * 1000} ms")
#     # print(f"cart_candidateslist time: {(generate_medusa_buffers_end_time - tree_candidate_begin_time) * 1000} ms")
    
    
#     return cart_candidateslist, tree_candidateslist, tree_attn_masklist, retrieve_indiceslist, Node_num
def generate_candidates(tree_logits, candidates_tree_prob, tree_indices, retrieve_indices, sample_token, logits_processor, node_index=None, Node_num = 256):

    bs = sample_token.shape[0]
    candidates_logit = sample_token

    candidates_tree_logits = tree_logits

    candidates = torch.cat([candidates_logit, candidates_tree_logits.view(bs, -1)], dim=-1)

    tree_candidates = candidates[:, tree_indices]
    Freeze_Node=[[i for i in range(26)]  for _  in range(bs)]
    Node_num = []
    cart_candidateslist=[] 
    tree_candidateslist=[] 
    tree_attn_masklist =[]
    retrieve_indiceslist=[]
    generate_medusa_buffers_begin_time = time.time()
    for i in range(bs):
        cart_candidates, tree_candidates_1, medusa_attn_mask ,retrieve_indices=generate_medusa_buffers(Freeze_Node[i],tree_candidates[i:i+1])
        cart_candidateslist.append(cart_candidates.squeeze(0))
        tree_candidateslist.append(tree_candidates_1.squeeze(0))
        tree_attn_masklist.append(medusa_attn_mask)
        retrieve_indiceslist.append(retrieve_indices)
        Node_num.append(tree_candidates_1.size(1))

    return cart_candidateslist, tree_candidateslist, tree_attn_masklist, retrieve_indiceslist, Node_num
def evaluate_posterior(
        logits, candidates, tree_candidates
):
    posterior_mask = (
            candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
    ).int()

    candidates_accept_length = (torch.cumprod(posterior_mask, dim=-1)).sum(dim=-1)
    accept_length = candidates_accept_length.max(dim=0).values
    best_candidate = torch.argmax(candidates_accept_length).to(torch.long)

    logits_batch = logits[best_candidate, accept_length, :]
    accept_length = accept_length

    return best_candidate.tolist(), accept_length, logits_batch
# def evaluate_posterior(
#         logits, candidates, tree_candidates
# ):
#     """
#     Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

#     Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
#     probabilities to select the best candidate.

#     Args:
#     - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
#     - candidates (torch.Tensor): Candidate token sequences.
#     - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
#     - posterior_threshold (float): Threshold for posterior probability.
#     - posterior_alpha (float): Scaling factor for the threshold.

#     Returns:
#     - best_candidate (torch.Tensor): Index of the chosen best candidate.
#     - accept_length (int): Length of the accepted candidate sequence.
#     """
#     # Greedy decoding based on temperature value
#     bs = tree_candidates.size(0)
#     # Find the tokens that match the maximum logits for each position in the sequence
#     # print(f"evaluate_posterior candidates: {candidates.shape} logits: {logits.shape}")
#     posterior_mask = (
#             candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
#     ).int()
#     # posterior_mask = (
#     #         candidates[:, :, 1:].to(logits.device) == torch.argmax(logits[:, :, :-1], dim=-1)
#     # ).int()
#     candidates_accept_length = (torch.cumprod(posterior_mask, dim=-1)).sum(dim=-1)
#     # print(f"candidates_accept_length: {candidates_accept_length.shape} \n{candidates_accept_length}")
#     accept_length = candidates_accept_length.max(dim=0).values
#     # print(f"accept_length: {accept_length}")
#     best_candidate = torch.argmax(candidates_accept_length).to(torch.long)

#     # print(f"logits: {logits.shape} \n{logits}")
#     # print(f"best_candidate: {best_candidate}")
#     # print(f"accept_length: {accept_length}")
#     logits_batch = logits[best_candidate, accept_length, :]
#     # print(f"logits_batch: {logits_batch.shape} \n{logits_batch}")
#     accept_length = accept_length

#     return best_candidate.tolist(), accept_length, logits_batch
