import torch
from utils import *
import time

class MultiTokenGenerator:
    def __init__(self, topk=10,DEPTH=4,medusa_choices=None):
        self.topk = topk
        self.DEPTH = DEPTH  # depth of the tree
        self.tree_indices = None
        self.static_tree = medusa_choices
        self.retrieve_indices = None
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
        sorted_medusa_choices = sorted(self.static_tree, key=lambda x: (len(x), x))
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
        # Node2=[]
        # Node3=[]
        # Node4=[]
        # Node5=[]
        # for i in range(len(retrieve_indices_nest)):
        #     if len(retrieve_indices_nest[i])>=2:
        #         Node2.append(retrieve_indices_nest[i])
        #     elif len(retrieve_indices_nest[i])>=3:
        #         Node3.append(retrieve_indices_nest[i])
        #     elif len(retrieve_indices_nest[i])>=4:
        #         Node4.append(retrieve_indices_nest[i])
        #     elif len(retrieve_indices_nest[i])>=5:
        #         Node5.append(retrieve_indices_nest[i])
        # self.node_path = [[0], Node2,Node3,Node4,Node5]
        # print(self.node_path)

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
        self.retrieve_indices = retrieve_indices

    def generate_medusa_buffers(self,
        medusa_choices,
        tree_candidates
    ):
        tree_candidates=tree_candidates
        sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
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
    
    def quick_node(self,
        medusa_logits,
        logits,
        medusa_choices,
    ):

        bsz = medusa_logits.size(0)    
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
        
        cart_candidateslist=[] 
        tree_candidateslist=[] 
        tree_attn_masklist =[]
        retrieve_indiceslist=[]
        nodenum=[]
        for i in range(bsz):
            cart_candidates, tree_candidates , medusa_attn_mask ,retrieve_indices=self.generate_medusa_buffers(medusa_choices,tree_candidates_indices[i:i+1])
            cart_candidateslist.append(cart_candidates.squeeze(0))
            tree_candidateslist.append(tree_candidates.squeeze(0))
            tree_attn_masklist.append(medusa_attn_mask)
            retrieve_indiceslist.append(retrieve_indices)
            nodenum.append(len(medusa_choices)+1)
        return cart_candidateslist, tree_candidateslist , tree_attn_masklist ,retrieve_indiceslist,nodenum
    


