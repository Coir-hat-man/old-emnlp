import torch
from utils import *
import time
class MultiTokenGenerator:
    def __init__(self, topk=10,Node=64,DEPTH=4):
        self.topk = topk
        self.interval = [0]
        self.NODE = Node  # number of nodes in the tree
        self.DEPTH = DEPTH  # depth of the tree
        for i in range(self.DEPTH):
            self.interval.append(self.interval[-1] + topk ** (i + 1))

    def sortedIdx_to_tree_buffer(self, sortedIdx):
        tree_indices = torch.zeros(self.NODE, dtype=torch.int64, device=sortedIdx.device)
        position_ids = torch.zeros(self.NODE, dtype=torch.int64, device=sortedIdx.device)
        retrieve_indices = torch.zeros((self.NODE, self.DEPTH + 1), dtype=torch.int64, device=sortedIdx.device)
        retrieve_indices[:, 1:] = -1
        tree_attn_mask = torch.zeros((self.NODE, self.NODE), device=sortedIdx.device)
        tree_attn_mask[:, 0] = 1.0

        nodes = {}
        is_leaf = torch.ones(self.NODE, dtype=torch.bool, device=sortedIdx.device)
        is_leaf[0] = False

        for i, idx in enumerate(sortedIdx):
            idx = idx.item()
            nodes[idx] = i + 1

            if idx < self.interval[1]:
                tree_indices[i + 1] = idx + 1
                position_ids[i + 1] = 1
                retrieve_indices[i + 1, 1] = i + 1
                tree_attn_mask[i + 1, idx + 1] = 1.0

            elif idx < self.interval[2]:
                idx -= self.interval[1]
                tree_indices[i + 1] = idx % self.topk + self.topk + 1
                position_ids[i + 1] = 2
                parent = nodes[idx // self.topk]
                is_leaf[parent] = False
                retrieve_indices[i + 1] = retrieve_indices[parent]
                retrieve_indices[i + 1, 2] = i + 1
                tree_attn_mask[i + 1, retrieve_indices[i + 1, 1:3]] = 1.0

            elif idx < self.interval[3]:
                idx -= self.interval[2]
                tree_indices[i + 1] = idx % self.topk + 2 * self.topk + 1
                position_ids[i + 1] = 3
                parent = nodes[idx // self.topk + self.topk]
                is_leaf[parent] = False
                retrieve_indices[i + 1] = retrieve_indices[parent]
                retrieve_indices[i + 1, 3] = i + 1
                tree_attn_mask[i + 1, retrieve_indices[i + 1, 1:4]] = 1.0

            elif idx < self.interval[4]:
                idx -= self.interval[3]
                tree_indices[i + 1] = idx % self.topk + 3 * self.topk + 1
                position_ids[i + 1] = 4
                parent = nodes[idx // self.topk + self.interval[2]]
                is_leaf[parent] = False
                retrieve_indices[i + 1] = retrieve_indices[parent]
                retrieve_indices[i + 1, 4] = i + 1
                tree_attn_mask[i + 1, retrieve_indices[i + 1, 1:5]] = 1.0

            else:
                idx -= self.interval[4]
                tree_indices[i + 1] = idx % self.topk + 4 * self.topk + 1
                position_ids[i + 1] = 5
                parent = nodes[idx // self.topk + self.interval[3]]
                is_leaf[parent] = False
                retrieve_indices[i + 1] = retrieve_indices[parent]
                retrieve_indices[i + 1, 5] = i + 1
                tree_attn_mask[i + 1, retrieve_indices[i + 1, 1:6]] = 1.0

        retrieve_indices = retrieve_indices[is_leaf]

        return tree_indices, position_ids, retrieve_indices, tree_attn_mask

    def generate_candidates(
        self,
        medusa_logits,
        logits,
        temperature=0,
        posterior_threshold=0.3,
        posterior_alpha=0.09,
        top_p=0.8,
        sampling='typical',
        fast=False,
    ):
        top_probs, top_indices = torch.topk(medusa_logits, self.topk, dim=-1)
        top_probs = top_probs.float()
        top_probs = F.softmax(top_probs, dim=-1).squeeze()

        # Greedy decoding: Select the most probable candidate from the original logits.
        if temperature == 0 or fast:
            candidates_logit = torch.argmax(logits).unsqueeze(0)
        else:
            if sampling == 'typical':
                candidates_logit = get_typical_one_token(
                    logits[:, -1], temperature, posterior_threshold, posterior_alpha
                ).squeeze(0)
            elif sampling == 'nucleus':
                candidates_logit = get_nucleus_one_token(logits[:, -1], temperature, top_p).squeeze(0)
            else:
                raise NotImplementedError

        # level 1
        p1_joint = top_probs[0]
        # level 2
        p2_joint = p1_joint.view(self.topk, 1) * top_probs[1].view(1, self.topk)
        # level 3
        p3_joint = p2_joint.view(self.topk, self.topk, 1) * top_probs[2].view(1, 1, self.topk)
        # level 4
        p4_joint = p3_joint.view(self.topk, self.topk, self.topk, 1) * top_probs[3].view(1, 1, 1, self.topk)

        if self.DEPTH == 5:
            p5_joint = p4_joint.view(self.topk, self.topk, self.topk, self.topk, 1) * top_probs[4].view(
                1, 1, 1, 1, self.topk
            )
            p_joint = torch.cat([p1_joint, p2_joint.view(-1), p3_joint.view(-1), p4_joint.view(-1), p5_joint.view(-1)])
        else:
            p_joint = torch.cat([p1_joint, p2_joint.view(-1), p3_joint.view(-1), p4_joint.view(-1)])

        _, tree_top_indices = torch.topk(p_joint, self.NODE - 1)
        tree_top_indices = tree_top_indices.sort()[0]

        # dynamic tree buffer
        tree_indices, position_ids, retrieve_indices, tree_attn_mask = self.sortedIdx_to_tree_buffer(tree_top_indices)

        medusa_buffers = {
            "medusa_attn_mask": tree_attn_mask,
            "tree_indices": tree_indices,
            "medusa_position_ids": position_ids,
            "retrieve_indices": retrieve_indices,
        }

        # Combine the selected candidate from the original logits with the topk medusa logits.
        candidates = torch.cat([candidates_logit, top_indices[:self.DEPTH].view(-1)], dim=-1)

        # Map the combined candidates to the tree indices to get tree candidates.
        tree_candidates = candidates[tree_indices]

        # Extend the tree candidates by appending a zero.
        tree_candidates_ext = torch.cat(
            [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0
        )
        # Retrieve the cartesian candidates using the retrieve indices.
        cart_candidates = tree_candidates_ext[retrieve_indices]

        return cart_candidates, tree_candidates , tree_attn_mask ,retrieve_indices

allocate_node=[12,12,12,12,12]
logits=torch.randn(1,1, 32000).to("cuda")
medusa_logits=torch.randn(1,5,32000).to("cuda")

for _ in range(5):
    start_time =time.time()
    for i in range(5):
        MT=MultiTokenGenerator(Node=12)
        cart_candidates, tree_candidates , tree_attn_mask ,retrieve_indices=MT.generate_candidates(medusa_logits=medusa_logits, logits=logits)
    end_time=time.time()
    print(end_time-start_time)


# allocate_node=[20,20,20,20,20]
# logits=torch.randn(1,1, 32000)
# medusa_logits=torch.randn(1,5,32000)
# end_time=time.time()


# allocate_node=[20,20,20,20,20]
# logits=torch.randn(1,1, 32000)
# medusa_logits=torch.randn(1,5,32000)
# start_time =time.time()
# for i in range(5):
#     MT=MultiTokenGenerator(Node=allocate_node[i])
#     MT.generate_candidates(medusa_logits=medusa_logits, logits=logits)
# end_time=time.time()
# print(end_time-start_time)

# allocate_node=[50,20,10,10,10]
# logits=torch.randn(1,1, 32000)
# medusa_logits=torch.randn(1,5,32000)
# start_time =time.time()
# for i in range(5):
#     MT=MultiTokenGenerator(Node=allocate_node[i])
#     MT.generate_candidates(medusa_logits=medusa_logits, logits=logits)
# end_time=time.time()
# print(end_time-start_time)