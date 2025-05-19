from Parallel_decoding import  Parallel_decoding
from transformers import AutoTokenizer, AutoConfig
import torch
import os
import json
import argparse
import tqdm

#the model chat format
chat_format={
    "vicuna7b":"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {}. ASSISTANT:"
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model_path", default="/home/zhouyh/model/medusa_vicuna_7b", type=str)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--Node_num", default=70, type=int)
    parser.add_argument("--threshold", default=8,type=float)
    parser.add_argument("--bench_name", default="MIX",choices=["GSM8k", "MBPP","Mt_bench","MIX"], type=str)
    parser.add_argument("--data_path",default="/home/zhouyh/homework/代码备份/Batch_medusa（支持QKVOMLP优化)修复多batch问题 实验搭建/data", type=str)
    parser.add_argument("--max_gen_len", default=300, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    device_index = torch.cuda.device_count() - 1
    #short context now
    context_length = 300
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    meta_prompts = []
    data_path = args.data_path +"/"+args.bench_name +"/"+ "torchquestion.jsonl"
    data_path = "/home/zhouyh/homework/代码备份2.0/Batch_medusa（细粒度剪枝） 复原优化/Ablation Experiment/Find_best_threshold/data/torchquestion.jsonl"
    raw_data = [json.loads(line) for line in open(data_path).readlines()]
    print("load data......")
    for data in raw_data[:args.batch_size]:
        chat_str=chat_format["vicuna7b"]
        prompt = chat_str.format(data["turns"][0])
        # print("prompt:",prompt)
        a = tokenizer(prompt, return_tensors="pt", padding='max_length', max_length=100, truncation=True, padding_side="right")
        a_cuda = a['input_ids'].cuda()
        len_cuda = a['attention_mask'].sum(dim=-1).cuda()
        meta_prompts.append({
            "prompt": prompt,
            "length": a['attention_mask'].sum(dim=-1).cuda(device=device_index),
            "input_ids": a_cuda,
        })
    print("data have finshed!")
    print("load model......")
    Medusa = Parallel_decoding(args.target_model_path,args.Node_num,threshold=args.threshold,Device_index=device_index)
    print("model have loaded!")
    counts = 0
    nums = 0
    glide_time = .0
    with torch.inference_mode():
        device_index = torch.cuda.device_count() - 1
        metaprompt_inputids = torch.stack([
            torch.tensor(ids["input_ids"], dtype=torch.long) for ids in meta_prompts
            ]).squeeze(1).cuda()
        pormpt_len = torch.stack([
            torch.tensor(ids["length"], dtype=torch.long) for ids in meta_prompts
            ]).squeeze(1).cuda(device=device_index)
        #warm up
        for i in range(1):
            output_ids, count, elapsed_time= Medusa.tree_spec_generate(
                # meta_prompt["input_ids"], 
                metaprompt_inputids,
                prompt_length=pormpt_len, 
                max_gen_len=3000
                )
        print("warm up finshed!")

        output_ids, count, elapsed_time= Medusa.tree_spec_generate(
            # meta_prompt["input_ids"], 
            metaprompt_inputids,
            prompt_length=pormpt_len, 
            max_gen_len=3000, 
            temperature=args.temperature,
        )

        print("-"*100)
        print("Node_num :",args.Node_num)
        print("threshold: ", args.threshold)
        print("inference time: ", elapsed_time)
        print("count: ", int(sum(count)))
        print(int(sum(count))/elapsed_time)
        print("-"*100)

        # for i in range(len(output_ids)):
        #     print(tokenizer.decode(output_ids[i]))
        #     print("\n")
        #     print("-"*100)
