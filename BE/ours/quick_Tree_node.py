import torch
import torch.nn.functional as F # 假设 MultiTokenGenerator 中用到了
import time

# --- 从 MultiTokenGenerator 借鉴/修改的辅助函数 ---
def _pad_path(path, max_length, pad_value=-1):
    """辅助函数：填充路径以使其达到最大长度。"""
    # 确保 path 是列表，而不是元组
    path_list = list(path) if isinstance(path, tuple) else path
    return path_list + [pad_value] * (max_length - len(path_list))

# @torch.jit.script # 尝试JIT，但路径元组和列表操作可能不完全兼容，需要调整
def regenerate_buffers_for_pruned_nodes_optimized(
    selected_node_original_indices: list,
    all_original_static_paths: list, # list of tuples
    pruned_node_tokens_for_item: torch.Tensor, # (1, num_selected_nodes)
    # top_k_branches_per_node: int, # 可能不再直接需要，因为我们不重新生成 tree_indices
    # max_depth_for_retrieve: int   # 可能不再直接需要，因为max_length由实际路径决定
    ):
    """
    优化版：根据选定的原始节点索引，重新生成 tree_attn_mask 和 retrieve_indices。
    主要优化：使用字典进行路径查找。
    """
    device = pruned_node_tokens_for_item.device

    # 1. 获取选中节点的路径定义，并排序 (这是新的 "medusa_choices")
    speculative_selected_paths_tuples = []
    for original_idx in selected_node_original_indices:
        if original_idx == 0: # 根节点单独处理或根据约定
            continue
        path = all_original_static_paths[original_idx]
        if not path: # 路径为空元组，通常是根的子节点
            continue
        speculative_selected_paths_tuples.append(path)

    # 按长度和值排序路径元组
    sorted_selected_paths_tuples = sorted(speculative_selected_paths_tuples, key=lambda x: (len(x), x))

    # 创建从路径元组到其在排序后列表中的索引的映射 (0-indexed for speculative)
    path_to_new_spec_idx = {path: i for i, path in enumerate(sorted_selected_paths_tuples)}

    num_selected_speculative_nodes = len(sorted_selected_paths_tuples)
    num_total_selected_nodes = pruned_node_tokens_for_item.size(1) # 包括根节点 (如果存在)

    # 2. 计算深度计数 (depth_counts)
    depth_counts = []
    prev_depth = 0
    if num_selected_speculative_nodes > 0:
        for path in sorted_selected_paths_tuples:
            depth = len(path)
            if depth != prev_depth:
                # 填充可能由于剪枝导致的深度间隙
                for _ in range(depth - prev_depth - 1):
                    depth_counts.append(0)
                depth_counts.append(0)
            depth_counts[depth - 1] += 1 # depth is 1-based, list index is 0-based
            prev_depth = depth
    
    # 3. 创建注意力掩码 (medusa_attn_mask)
    medusa_attn_mask_pruned = torch.eye(num_total_selected_nodes, num_total_selected_nodes, device=device, dtype=torch.bool) # 使用bool以节省内存，后续可转float
    if num_total_selected_nodes > 0 : # 确保至少有根节点
      medusa_attn_mask_pruned[:, 0] = True # 所有选定节点关注初始token (节点0)

    start_idx_in_sorted_paths = 0 # 对应 path_to_new_spec_idx 的值
    for i in range(len(depth_counts)): # i is depth - 1
        for j in range(depth_counts[i]):
            # current_path_tuple = sorted_selected_paths_tuples[start_idx_in_sorted_paths + j]
            # current_node_new_idx_in_spec = start_idx_in_sorted_paths + j
            # current_node_total_idx = current_node_new_idx_in_spec + 1 # +1 因为掩码索引0是根

            # 从 path_to_new_spec_idx 反向查找路径可能更清晰，或者直接用索引
            current_node_new_idx_in_spec = start_idx_in_sorted_paths + j
            current_path_tuple = sorted_selected_paths_tuples[current_node_new_idx_in_spec]
            current_node_total_idx = current_node_new_idx_in_spec + 1


            if len(current_path_tuple) == 1: # 深度为1的节点，父节点是根 (索引0)
                # 已经通过 eye 和 medusa_attn_mask_pruned[:, 0] = True 设置
                continue

            ancestor_indices_in_mask = []
            # 查找所有祖先路径在新的剪枝集中的索引
            for c_len in range(len(current_path_tuple) - 1): # 遍历所有前缀路径作为祖先
                ancestor_p_tuple = current_path_tuple[:c_len+1]
                # 使用字典查找祖先在新列表中的索引
                anc_new_idx_spec = path_to_new_spec_idx.get(ancestor_p_tuple)
                if anc_new_idx_spec is not None:
                    ancestor_indices_in_mask.append(anc_new_idx_spec + 1) # +1 for root offset
                # else:
                    # 如果祖先不在选定路径中 (这理论上不应发生，如果剪枝保证了树的连通性)
                    # print(f"Warning: Ancestor {ancestor_p_tuple} of {current_path_tuple} not found in selected paths.")
            
            if ancestor_indices_in_mask:
                 medusa_attn_mask_pruned[current_node_total_idx, ancestor_indices_in_mask] = True
            # else:
                 # 如果没有找到任何有效祖先（除了根），它只关注根，这已设置
                 # print(f"Warning: No valid speculative ancestors found for {current_path_tuple}. Attends only to root.")

        start_idx_in_sorted_paths += depth_counts[i]

    # 4. 生成检索索引 (retrieve_indices)
    retrieve_indices_nest_pruned_list = [] # 使用列表存储变长路径
    
    # 确保根路径总是存在 (如果至少有一个节点)
    if num_total_selected_nodes > 0:
        retrieve_indices_nest_pruned_list.append([0]) # 根路径

    processed_leaf_paths_for_retrieve = set() # 用于确保每个叶子路径只添加一次其主干

    for i in range(len(sorted_selected_paths_tuples)):
        current_path_tuple = sorted_selected_paths_tuples[-i-1] # 从最长的路径开始处理
        
        # 检查此路径是否已经是某个已处理路径的子路径，或者它本身是否是叶子节点
        # 我们希望每个 "verification path" (从根到一个节点) 都是唯一的
        # 原始逻辑是收集所有节点到根的路径，并去重。
        # 这里我们为 sorted_selected_paths_tuples 中的每个路径构建到根的路径

        current_retrieve_path_indices = [0] # Start with root node
        for sub_path_len in range(1, len(current_path_tuple) + 1):
            sub_path_tuple = current_path_tuple[:sub_path_len]
            node_idx_in_spec = path_to_new_spec_idx.get(sub_path_tuple)
            if node_idx_in_spec is not None:
                current_retrieve_path_indices.append(node_idx_in_spec + 1)
            else:
                # 子路径不在选定节点中，这是预料之外的，除非 selected_nodes 不构成有效子树
                # print(f"Error: Sub-path {sub_path_tuple} for {current_path_tuple} not found.")
                current_retrieve_path_indices = [] # 标记为无效路径
                break
        
        if current_retrieve_path_indices:
            # 避免重复添加完全相同的路径
            # tuple(current_retrieve_path_indices) 可以用来检查重复，但列表本身可能就够了
            # 如果需要严格的 MultiTokenGenerator.retrieve_indices 语义，可能需要更仔细的去重
            if tuple(current_retrieve_path_indices) not in processed_leaf_paths_for_retrieve:
                 retrieve_indices_nest_pruned_list.append(current_retrieve_path_indices)
                 processed_leaf_paths_for_retrieve.add(tuple(current_retrieve_path_indices))


    # 去重 retrieve_indices_nest_pruned_list (如果上面逻辑可能产生重复)
    # 通常，从每个选定节点回溯到根的路径是唯一的。
    # 如果 sorted_selected_paths_tuples 本身就代表了所有需要验证的“叶子”或重要中间节点，
    # 那么上面的逻辑应该没问题。

    max_len_retrieve = 0
    if retrieve_indices_nest_pruned_list:
        max_len_retrieve = max(len(p) for p in retrieve_indices_nest_pruned_list)
    elif num_total_selected_nodes > 0 : # 只有根节点
        max_len_retrieve = 1 # 路径 [0]
    # else: num_total_selected_nodes == 0, max_len_retrieve is 0

    # padding_idx_for_retrieve 指向填充token的位置
    # tree_candidates_ext 在原始代码中追加了一个值为-1的token。
    # 所以，retrieve_indices的填充值应该是 num_total_selected_nodes (这个新附加token的索引)
    padding_idx_for_retrieve = num_total_selected_nodes 

    retrieve_indices_pruned_tensor_list = []
    if max_len_retrieve > 0: # 只有当有路径时才填充
        for path_list in retrieve_indices_nest_pruned_list:
            retrieve_indices_pruned_tensor_list.append(
                torch.tensor(_pad_path(path_list, max_len_retrieve, pad_value=padding_idx_for_retrieve), dtype=torch.long, device=device)
            )
    
    if retrieve_indices_pruned_tensor_list:
        retrieve_indices_pruned = torch.stack(retrieve_indices_pruned_tensor_list)
    else: # 没有路径或没有节点
        retrieve_indices_pruned = torch.empty((0, max_len_retrieve if max_len_retrieve > 0 else 1), dtype=torch.long, device=device)

    # 5. 生成 cart_candidates
    padding_token_value = -1 # 哨兵值
    padded_pruned_node_tokens = torch.cat(
        [pruned_node_tokens_for_item,
         torch.full((1, 1), padding_token_value, dtype=torch.long, device=device)],
        dim=1
    )
    
    if retrieve_indices_pruned.numel() == 0 or retrieve_indices_pruned.shape[1] == 0 :
        cart_candidates_pruned = torch.empty((retrieve_indices_pruned.shape[0], retrieve_indices_pruned.shape[1]), dtype=torch.long, device=device)
    else:
        # retrieve_indices_pruned 中的索引必须在 padded_pruned_node_tokens 的范围内
        # 最大索引是 num_total_selected_nodes (对应填充token)
        # 确保 retrieve_indices_pruned 中的值 < padded_pruned_node_tokens.size(1)
        if retrieve_indices_pruned.numel() > 0 and retrieve_indices_pruned.max() >= padded_pruned_node_tokens.size(1):
            # print(f"Warning: Max index in retrieve_indices_pruned ({retrieve_indices_pruned.max()}) is out of bounds for padded_pruned_node_tokens size ({padded_pruned_node_tokens.size(1)}). Clamping.")
            retrieve_indices_pruned = torch.clamp(retrieve_indices_pruned, max=padded_pruned_node_tokens.size(1) - 1)

        cart_candidates_pruned = padded_pruned_node_tokens.gather(1, retrieve_indices_pruned)


    return cart_candidates_pruned, pruned_node_tokens_for_item, medusa_attn_mask_pruned.float(), retrieve_indices_pruned

def generate_candidates_with_pruning(
    tree_logits_list_batch,       # Tensor (bs, num_spec_slots) - chosen tokens
    tree_logits_prob_list_batch,  # Tensor (bs, num_spec_slots) - probs of chosen tokens
    original_tree_indices,        # Tensor (num_total_original_nodes_inc_root,)
    # original_retrieve_indices,    # 不再直接需要，因为我们会重新生成
    sample_token,                 # Tensor (bs, 1) or (bs,)
    logits_processor,             # Logits processor (可能在此函数中未使用)
    num_nodes_to_keep_total: int, 
    all_original_static_paths: list, 
    # top_k_branches_per_node: int, # 不再直接需要
    # max_depth_for_retrieve: int   # 不再直接需要
    ):
    """
    使用优化后的 regenerate_buffers_for_pruned_nodes_optimized。
    """
    if isinstance(tree_logits_list_batch, list):
        bs = len(tree_logits_list_batch)
        if bs == 0: return [], [], [], []
        # 假设可以安全地堆叠，如果不行，则需要在 draft_prune 中循环
        tree_logits_stacked = torch.stack(tree_logits_list_batch, dim=0)
        tree_probs_stacked = torch.stack(tree_logits_prob_list_batch, dim=0)
    else: #已经是张量
        bs = tree_logits_list_batch.shape[0]
        if bs == 0: return [], [], [], []
        tree_logits_stacked = tree_logits_list_batch
        tree_probs_stacked = tree_logits_prob_list_batch
    
    device = sample_token.device
    print(f"sample_token: {sample_token.shape}, {sample_token}")
    sample_token = sample_token.view(bs, -1) 

    # 1. 获取 `full_tree_nodes_tokens` 和 `full_tree_nodes_probs`
    all_tokens_flat = torch.cat([sample_token, tree_logits_stacked.view(bs, -1)], dim=-1)
    prob_sample_token = torch.ones((bs, 1), device=device, dtype=torch.float32)
    all_probs_flat = torch.cat([prob_sample_token, tree_probs_stacked.view(bs, -1)], dim=-1)

    full_tree_nodes_tokens = all_tokens_flat.gather(1, original_tree_indices.unsqueeze(0).expand(bs, -1))
    full_tree_nodes_probs = all_probs_flat.gather(1, original_tree_indices.unsqueeze(0).expand(bs, -1))

    # 2. 节点剪枝 (向量化尝试，如果剪枝后节点数固定)
    # 如果 num_nodes_to_keep_total 导致每个batch项选择的节点数不同，则仍需循环。
    # 为保持通用性，这里保留循环，但可以考虑针对固定数量进行优化。
    
    pruned_cart_candidates_list = []
    pruned_tree_candidates_list = [] 
    pruned_tree_mask_list = []
    pruned_retrieve_indices_list = []

    for b in range(bs):
        current_probs = full_tree_nodes_probs[b] # (num_total_original_nodes,)
        speculative_node_probs_item = current_probs[1:]
        
        num_spec_nodes_to_keep = max(0, num_nodes_to_keep_total - 1)
        
        if speculative_node_probs_item.numel() == 0 or num_spec_nodes_to_keep == 0:
            selected_spec_indices_original_offset = torch.empty((0,), dtype=torch.long, device=device)
        elif speculative_node_probs_item.numel() <= num_spec_nodes_to_keep:
            selected_spec_indices_original_offset = torch.arange(speculative_node_probs_item.numel(), device=device) + 1
        else:
            _, top_k_spec_indices = torch.topk(speculative_node_probs_item, k=num_spec_nodes_to_keep)
            selected_spec_indices_original_offset = top_k_spec_indices + 1 

        final_selected_indices_item_original = torch.cat([
            torch.tensor([0], dtype=torch.long, device=device),
            selected_spec_indices_original_offset
        ])
        final_selected_indices_item_original_sorted = torch.sort(final_selected_indices_item_original).values
        
        # 3. Gather pruned tree candidates (tokens)
        current_pruned_tokens_item = full_tree_nodes_tokens[b:b+1, final_selected_indices_item_original_sorted]
        
        # 4. Regenerate buffers for this item using the optimized function
        print(f"final_selected_indices_item_original_sorted.tolist(): {final_selected_indices_item_original_sorted.tolist()}")
        print(f"all_original_static_paths: {all_original_static_paths}")
        print(f"current_pruned_tokens_item: {current_pruned_tokens_item}")
        cart_cand_b, tree_cand_b, mask_b, retr_idx_b = regenerate_buffers_for_pruned_nodes_optimized(
            final_selected_indices_item_original_sorted.tolist(), # 传递原始索引列表
            all_original_static_paths,
            current_pruned_tokens_item
        )
        pruned_cart_candidates_list.append(cart_cand_b.squeeze(0)) 
        pruned_tree_candidates_list.append(tree_cand_b.squeeze(0))
        pruned_tree_mask_list.append(mask_b)
        pruned_retrieve_indices_list.append(retr_idx_b)
    
    return (pruned_cart_candidates_list,
            pruned_tree_candidates_list,
            pruned_tree_mask_list,
            pruned_retrieve_indices_list)