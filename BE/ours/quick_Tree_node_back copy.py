import torch
from utils import *
import time
class MultiTokenGenerator:
    def __init__(self, topk=10,Node=64,DEPTH=4):
        self.topk = topk
        self.interval = [0]
        self.NODE = Node  # number of nodes in the tree
        self.DEPTH = DEPTH  # depth of the tree
        self.static_tree = sorted([(0,), (0, 0), (1,), (0, 1), (0, 0, 0), (1, 0), (2,), (0, 2), (0, 0, 1), (0, 3), (3,), (0, 1, 0), (2, 0), (4,), (0, 0, 2), (0, 4), (1, 1), (1, 0, 0), (0, 0, 0, 0), (5,), (0, 0, 3), (0, 5), (0, 2, 0), (3, 0), (0, 1, 1), (0, 6), (6,), (0, 7), (0, 0, 4), (4, 0), (1, 2), (0, 8), (7,), (0, 3, 0), (0, 0, 0, 1), (0, 0, 5), (2, 1), (0, 0, 6), (1, 0, 1), (0, 0, 1, 0), (2, 0, 0), (5, 0), (0, 9), (0, 1, 2), (8,), (0, 4, 0), (0, 2, 1), (1, 3), (0, 0, 7), (0, 0, 0, 2), (0, 0, 8), (1, 1, 0), (0, 1, 0, 0), (6, 0), (9,), (0, 1, 3), (0, 0, 0, 3), (1, 0, 2), (0, 5, 0), (3, 1), (0, 0, 2, 0), (7, 0), (1, 4)], key=lambda x: (len(x), x))
        self.tree_indices = None
        self.retrieve_indices = None
        self.node_index = None
        self.node_path = None
        self.freeze_num =2#选取top2进行冻结
        self.init()#初始化

    def get_paths(self,tree_tensor, depth, topk):
        depth_idx = depth - 1
        column = tree_tensor[:, depth_idx]
        
        # 过滤掉无效的-1值
        mask = column != -1
        valid_values = column[mask]
        
        # 获取唯一值并按升序排序
        unique_values = torch.unique(valid_values)
        sorted_unique, _ = torch.sort(unique_values)
        
        # 取前topk个唯一值
        topk = min(topk, len(sorted_unique))  # 确保不超过实际数量
        selected_values = sorted_unique[:topk]
        
        # 收集每个值首次出现的行索引
        selected_rows = []
        for value in selected_values:
            # 使用nonzero找到第一个出现的索引
            row_idx = (column == value).nonzero()[0, 0].item()
            selected_rows.append(row_idx)
        
        # 提取路径的前depth个元素
        paths = [tree_tensor[row, :depth].tolist() for row in selected_rows]
    
        return paths
    
    def init(self,device="cuda"):

        # Sort the medusa_choices based on their lengths and then their values
        sorted_medusa_choices = self.static_tree
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
        # Generate tree indices for the Medusa structure
        medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long)
        medusa_tree_indices[0] = 0
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_medusa_choice = sorted_medusa_choices[start + j]
                medusa_tree_indices[start + j + 1] = cur_medusa_choice[-1] + self.topk * i + 1
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
                    retrieve_indice.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1])+1)
                    retrieve_paths.append(cur_medusa_choice[:c+1])
            retrieve_indices_nest.append([0]+retrieve_indice)

        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)

        self.node_path=[self.get_paths(retrieve_indices,2,self.freeze_num)]
        for i in range(2,self.DEPTH+2):
           self.node_path.append(self.get_paths(retrieve_indices,i,self.freeze_num))

        coordinates = []
        for number in range(medusa_len):
            positions = torch.where(retrieve_indices == number)
            if positions[0].numel() > 0:
                coordinates.append([positions[0][0].item(), positions[1][0].item()])
        self.node_index = torch.tensor(coordinates, dtype=torch.long)
        # Aggregate the generated buffers into a dictionary
        self.tree_indices = medusa_tree_indices
        print(f"self.tree_indices: {self.tree_indices}")
        self.retrieve_indices = retrieve_indices

    def generate_medusa_buffers(self,
        slect_node_index,
        tree_candidates
    ):

        slect_node_index = sorted(slect_node_index)
        tree_candidates =tree_candidates[:,slect_node_index]
        medusa_choices = [self.static_tree[i-1] for i in slect_node_index if i > 0]

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
        max_length = self.DEPTH
        retrieve_indices = [pad_path(path, max_length,-2) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1)
        pad_tree_candidates =  torch.cat([tree_candidates, torch.zeros((1,1), dtype=torch.long, device=tree_candidates.device)], dim=1)
        cart_candidates  = pad_tree_candidates[:,retrieve_indices]

        return  cart_candidates, tree_candidates , medusa_attn_mask ,retrieve_indices
    
    def quick_node_prune(self,
        medusa_logits,
        logits,
        predict_len,
        Node_num,
    ):
        self.NODE = Node_num#更新实际分配的Token数
        bsz = medusa_logits.size(0)

        if bsz*64 <= self.NODE:#防止越界,给最多的Node数
            self.NODE =64*bsz
            
        top_probs, top_indices = torch.topk(medusa_logits, self.topk, dim=-1)
        top_probs = top_probs.float()
        top_probs = F.softmax(top_probs, dim=-1)

        logits = logits.float()
        logits = F.softmax(logits, dim=-1)
        # Greedy decoding: Select the most probable candidate from the original logits.

        candidates_prob ,candidates_logit = torch.topk(logits,1,dim=-1)
        if bsz==1:
            candidates_prob = candidates_prob.unsqueeze(0)
            candidates_logit = candidates_logit.unsqueeze(0)
        # Combine the selected candidate from the original logits with the topk medusa logits.
        allcandidates_probs = torch.cat([candidates_prob, top_probs[:,:self.DEPTH].view(bsz,-1)], dim=-1)
        allcandidates_logits = torch.cat([candidates_logit, top_indices[:,:self.DEPTH].view(bsz,-1)], dim=-1)
        
        
        # Map the combined candidates to the tree indices to get tree candidates.
        tree_candidates_probs = allcandidates_probs[:,self.tree_indices]
        tree_candidates_indices = allcandidates_logits[:,self.tree_indices]
        
        # Extend the tree candidates by appending a zero.
        tree_candidates_ext_probs = torch.cat([tree_candidates_probs, torch.zeros((bsz,1), dtype=torch.long, device=tree_candidates_probs.device)], dim=-1)
        # Retrieve the cartesian candidates using the retrieve indices.
        cart_candidates_probs = tree_candidates_ext_probs[:,self.retrieve_indices]
        # Calculate the cumulative product of the cartesian candidates.
        cart_candidates_probs = torch.cumprod(cart_candidates_probs, dim=-1)
        Node_probs =  cart_candidates_probs[:, self.node_index[:, 0], self.node_index[:, 1]]
        
        Freeze_Node =[]
        Freeze_Node_num =0
        Freeze_mask =torch.zeros_like(Node_probs, dtype=torch.bool)
        for i in range(bsz):#计算每个请求冻结的node
            if predict_len[i] == 1:
                Nodelist= self.node_path[predict_len[i]]
                Nodelist = list(set(element for sublist in Nodelist for element in sublist))
                Freeze_Node.append(Nodelist)
                Freeze_mask[i,Nodelist] = True
                Freeze_Node_num +=len(Nodelist)
            else:
                Nodelist= self.node_path[predict_len[i]-1]
                Nodelist = list(set(element for sublist in Nodelist for element in sublist))
                Freeze_Node.append(Nodelist)
                Freeze_mask[i,Nodelist] = True
                Freeze_Node_num +=len(Nodelist)

        #剪枝部分
        if self.NODE - Freeze_Node_num > 3:#超出直接给冻结的Node即可
            Prune_Node = self.NODE - Freeze_Node_num
            Node_probs[Freeze_mask] = float('-inf')
            batch_size, num_nodes = Node_probs.shape
            # 摊开概率矩阵并获取topk的索引
            topk_values, topk_flat_indices = torch.topk(Node_probs.view(-1), Prune_Node)
            
            # 计算原始的batch索引和node索引
            topk_batch_indices = topk_flat_indices // num_nodes
            topk_node_indices = topk_flat_indices % num_nodes
            for i,j in zip(topk_batch_indices.tolist(), topk_node_indices.tolist()):
                Freeze_Node[i]=Freeze_Node[i]+[j]
        Node_num = []
        cart_candidateslist=[] 
        tree_candidateslist=[] 
        tree_attn_masklist =[]
        retrieve_indiceslist=[]
        for i in range(bsz):
            cart_candidates, tree_candidates , medusa_attn_mask ,retrieve_indices=self.generate_medusa_buffers(Freeze_Node[i],tree_candidates_indices[i:i+1])
            cart_candidateslist.append(cart_candidates.squeeze(0))
            tree_candidateslist.append(tree_candidates.squeeze(0))
            tree_attn_masklist.append(medusa_attn_mask)
            retrieve_indiceslist.append(retrieve_indices)
            Node_num.append(tree_candidates.size(1))
        return cart_candidateslist, tree_candidateslist , tree_attn_masklist, retrieve_indiceslist, Node_num
    
    
    # def quick_node_prune(self,
    #     medusa_logits,
    #     logits,
    #     predict_len,
    #     Node_num,
    # ):
    #     self.NODE = Node_num#更新实际分配的Token数
    #     bsz = medusa_logits.size(0)

    #     if bsz*64 <= self.NODE:#防止越界,给最多的Node数
    #         self.NODE =64*bsz
            
    #     top_probs, top_indices = torch.topk(medusa_logits, self.topk, dim=-1)
    #     top_probs = top_probs.float()
    #     top_probs = F.softmax(top_probs, dim=-1)

    #     logits = logits.float()
    #     logits = F.softmax(logits, dim=-1)
    #     # Greedy decoding: Select the most probable candidate from the original logits.

    #     candidates_prob ,candidates_logit = torch.topk(logits,1,dim=-1)
    #     if bsz==1:
    #         candidates_prob = candidates_prob.unsqueeze(0)
    #         candidates_logit = candidates_logit.unsqueeze(0)
    #     # Combine the selected candidate from the original logits with the topk medusa logits.
    #     allcandidates_probs = torch.cat([candidates_prob, top_probs[:,:self.DEPTH].view(bsz,-1)], dim=-1)
    #     allcandidates_logits = torch.cat([candidates_logit, top_indices[:,:self.DEPTH].view(bsz,-1)], dim=-1)
        
        
    #     # Map the combined candidates to the tree indices to get tree candidates.
    #     tree_candidates_probs = allcandidates_probs[:,self.tree_indices]
    #     tree_candidates_indices = allcandidates_logits[:,self.tree_indices]
        
    #     # Extend the tree candidates by appending a zero.
    #     tree_candidates_ext_probs = torch.cat([tree_candidates_probs, torch.zeros((bsz,1), dtype=torch.long, device=tree_candidates_probs.device)], dim=-1)
    #     # Retrieve the cartesian candidates using the retrieve indices.
    #     cart_candidates_probs = tree_candidates_ext_probs[:,self.retrieve_indices]
    #     # Calculate the cumulative product of the cartesian candidates.
    #     cart_candidates_probs = torch.cumprod(cart_candidates_probs, dim=-1)
    #     Node_probs =  cart_candidates_probs[:, self.node_index[:, 0], self.node_index[:, 1]]
        
    #     Freeze_Node =[]
    #     Freeze_Node_num =0
    #     Freeze_mask =torch.zeros_like(Node_probs, dtype=torch.bool)
    #     for i in range(bsz):#计算每个请求冻结的node
    #         if predict_len[i] == 1:
    #             Nodelist= self.node_path[predict_len[i]]
    #             Nodelist = list(set(element for sublist in Nodelist for element in sublist))
    #             Freeze_Node.append(Nodelist)
    #             Freeze_mask[i,Nodelist] = True
    #             Freeze_Node_num +=len(Nodelist)
    #         else:
    #             Nodelist= self.node_path[predict_len[i]-1]
    #             Nodelist = list(set(element for sublist in Nodelist for element in sublist))
    #             Freeze_Node.append(Nodelist)
    #             Freeze_mask[i,Nodelist] = True
    #             Freeze_Node_num +=len(Nodelist)

    #     #剪枝部分
    #     if self.NODE - Freeze_Node_num > 3:#超出直接给冻结的Node即可
    #         Prune_Node = self.NODE - Freeze_Node_num
    #         Node_probs[Freeze_mask] = float('-inf')
    #         batch_size, num_nodes = Node_probs.shape
    #         # 摊开概率矩阵并获取topk的索引
    #         topk_values, topk_flat_indices = torch.topk(Node_probs.view(-1), Prune_Node)
            
    #         # 计算原始的batch索引和node索引
    #         topk_batch_indices = topk_flat_indices // num_nodes
    #         topk_node_indices = topk_flat_indices % num_nodes
    #         for i,j in zip(topk_batch_indices.tolist(), topk_node_indices.tolist()):
    #             Freeze_Node[i]=Freeze_Node[i]+[j]
    #     Node_num = []
    #     cart_candidateslist=[] 
    #     tree_candidateslist=[] 
    #     tree_attn_masklist =[]
    #     retrieve_indiceslist=[]
    #     for i in range(bsz):
    #         cart_candidates, tree_candidates , medusa_attn_mask ,retrieve_indices=self.generate_medusa_buffers(Freeze_Node[i],tree_candidates_indices[i:i+1])
    #         cart_candidateslist.append(cart_candidates.squeeze(0))
    #         tree_candidateslist.append(tree_candidates.squeeze(0))
    #         tree_attn_masklist.append(medusa_attn_mask)
    #         retrieve_indiceslist.append(retrieve_indices)
    #         Node_num.append(tree_candidates.size(1))
    #     return cart_candidateslist, tree_candidateslist , tree_attn_masklist, retrieve_indiceslist, Node_num
    

# predict_len = [1, 3, 5,5,3,1,1,3,5,5]
# logits=torch.randn(5,32000).to('cuda')
# medusa_logits=torch.randn(5,5,32000).to('cuda')
# MT=MultiTokenGenerator(Node=200)
# for _ in range(100):
#     MT.quick_node_prune(medusa_logits=medusa_logits, logits=logits,predict_len=predict_len)
# start_time =time.time()
# MT.quick_node_prune(medusa_logits=medusa_logits, logits=logits,predict_len=predict_len)
# end_time=time.time()
# print(end_time-start_time)

