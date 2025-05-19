import asyncio
import time
from tqdm.asyncio import tqdm
import argparse
import json
import os
from transformers import PreTrainedTokenizerBase
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from Parallel_decoding import Parallel_decoding
from Node_choice import *

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
def calculate_average_by_type(types, values):
    # 创建一个字典来存储每个类型的数据
    data_by_type = {}

    # 遍历类型和值
    for type_, value in zip(types, values):
        # 如果类型不在字典中，则添加
        if type_ not in data_by_type:
            data_by_type[type_] = []
        # 将值添加到对应类型的列表中
        data_by_type[type_].append(value)
    with open("/data/home/weijh/zyh/emnlp/BE/AblationExperiment/E_value/result/e13b.jsonl", 'a', encoding='utf-8') as f:
        json_line = json.dumps(data_by_type, ensure_ascii=False)
        f.write(json_line + '\n')
    # 计算每个类型的平均值
    average_by_type = {type_: sum(values) / len(values) for type_, values in data_by_type.items()}

    return average_by_type
class ModelRequest:
    def __init__(self, prompt):
        self.prompt = prompt
        self.input_len = None
        self.input_ids = None  # 后面需要处理为pad
        self.output_len = 0
        self.output_ids = None
        self.steps = 0
        self.arriver_time = 0
        self.start_time = 0
        self.prefill_end_time = 0 
        self.end_time = 0

class RequestProcessor:

    def __init__(self, args, tokenizer, model):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, " \
                      "detailed, and polite answers to the user's questions. USER: {}. ASSISTANT:"
        self.free_num = args.max_batch_size  # 当前模型Kvcahe空闲的数量
        self.process_flag = False #当前是否有请求需要插入进去
        self.completed_stack = []  # 已经完成的请求往里面打
        self.model_forward = False  # 模型是否在运行
        self.finsh_request_num = 0
        self.input_request_num =0
        self.start_time = 0
        self.end_time =0
        self.insert_flag = False#表示当前有请求需要插入但是还没有处理
        self.batch = [] #存储当前需要插入的请求

    def sample_requests(self, num_reqs):
        dataset = [json.loads(line) for line in open(self.args.data_path).readlines()]
        prompts = [self.prompt.format(d["turns"][0]) for d in dataset]
        ModelRequest_list = []
        for i in range(len(prompts)):
            requst = ModelRequest(
                prompt=prompts[i]
            )
            ModelRequest_list.append(requst)
        return ModelRequest_list[:num_reqs]

    async def get_request(self, queue, data):#按照一定速率发送请求
        print("投入请求总数：", len(data))
        for item in data:
            await asyncio.sleep(float(1/self.args.Request_speed))  # 按速率间隔生成请求
            item.arriver_time = time.time()
            await queue.put(item)  # 确保放入队列完成
            
        await queue.put(None)  # 添加结束标记

    def eval(self):#对延迟和吞吐进行测评
        print("-"*20+"eval"+ "-"*20)
        all_latency= 0
        all_count =0
        count =0
        for request in self.completed_stack:
            all_latency+= request.end_time-request.arriver_time
            all_count  += request.output_len
            count +=1
            
            # print("-"*20 + " request " + str(count) + "-"*20 + str(request.output_len) + "\n")
            # print(f"finsh_request_list[0].output: {self.tokenizer.decode(request.output_ids[0], skip_special_tokens=True) }")
            # print("-"*20+" end"+ "-"*20 + "\n")
            #sentence = self.tokenizer.decode(request.output_ids[0])
        print(calculate_average_by_type(self.model.acc_num_list,self.model.weight_list))
        output ={"batch_size:":self.args.max_batch_size,"request_speed:":self.args.Request_speed,"average_latency:":all_latency/len(self.completed_stack),"throughput:":all_count/(self.end_time-self.start_time)}
        print(output)
        # print(f"execute time: {(self.end_time - self.start_time) * 1000} ms")
        # print(f"output token: {all_count}")
        # print(f"request_num: {count}")
        # with open(self.args.save_path, 'a', encoding='utf-8') as f:
        #     json_line = json.dumps(output, ensure_ascii=False)
        #     f.write(json_line + '\n')
            
    def process_requests(self, requests_list):
        self.model.model_forward = True  # 模型开始运行不接受请求
        if sum(self.model.free_flag) != 0 or len(requests_list) != 0:
            self.input_request_num += len(requests_list)
            self.free_num -=len(requests_list)
            finsh_requestlist = self.model.tree_spec_generate(requests_list)
            if finsh_requestlist is not None:
                end_time = time.time()
                for request in finsh_requestlist:
                    request.end_time = end_time
                    self.completed_stack.append(request)
                    
                self.free_num += len(finsh_requestlist)
                self.finsh_request_num += len(finsh_requestlist)
        self.batch =[]
        self.model.insert_flag = False
        self.insert_flag = False
        self.model_forward = False

    async def main(self):
        print("warm up")
        for _ in tqdm(range(1)):
            count =1
            warm_requests =self.sample_requests(count)#用五个请求进行设备预热
            while True:
                finsh_request_list=self.model.tree_spec_generate(warm_requests)
                count -=len(finsh_request_list)
                print(f"finsh_request_list: {len(finsh_request_list)}")
                # print('-'*20+" inpur"+ "-"*20 + "\n")
                # print(f"finsh_request_list[0].input_ids: {finsh_request_list[0].input_ids}")
                # print(f"finsh_request_list[0].input: {self.tokenizer.decode(finsh_request_list[0].input_ids, skip_special_tokens=True) }")
                print('-'*20+" output"+ "-"*20 + "\n")
                print(f"finsh_request_list[0].output_ids: {len(finsh_request_list[0].output_ids)}")
                print(f"finsh_request_list[0].output: {self.tokenizer.decode(finsh_request_list[0].output_ids[0], skip_special_tokens=True) }")
                warm_requests = []
                if count ==0:
                    break
        print("warm up finished")
        torch.cuda.synchronize()
        self.start_time =time.time()

        ModelRequest_list = self.sample_requests(self.args.All_request_num)
        queue = asyncio.Queue()
        producer = asyncio.create_task(self.get_request(queue, ModelRequest_list))
        producer_done = False

        with tqdm(total=len(ModelRequest_list), desc="Total progress") as pbar:
            loop = asyncio.get_event_loop()
            while True:
                if self.finsh_request_num >= len(ModelRequest_list):
                    break

                if producer.done() and not producer_done:#队列内请求已经全部打出
                    try:
                        await producer
                    except:
                        pass
                    producer_done = True

                #收集请求
                while len(self.batch) < self.free_num :
                    queue_flag =True
                    try:
                        request = queue.get_nowait()
                        if request is None:
                            producer_done = True
                        else:
                            self.batch.append(request)
                    except asyncio.QueueEmpty:#队列内无请求
                        break


                # 处理请求
                if self.model_forward ==False :
                    before = self.finsh_request_num
                    await loop.run_in_executor(
                        None, 
                        lambda: self.process_requests(self.batch)
                    )
                    delta = self.finsh_request_num - before
                    pbar.update(delta)


                elif producer_done and self.model_forward == False:
                    before = self.finsh_request_num
                    await loop.run_in_executor(
                        None, 
                        lambda: self.process_requests([])
                    )
                    delta = self.finsh_request_num - before
                    pbar.update(delta)
                    
                if len(self.batch)>0:
                    # self.insert_flag = True
                    self.model.insert_flag = True

                else:
                    await asyncio.sleep(0.000005)

        await producer
        torch.cuda.synchronize()
        self.end_time =time.time()
        

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Medusa Model Evaluator")
    # base_model_path = "/home/weijh/model_weight/base_model/vicuna-7b-v1.3"
    # EAGLE_model_path = "/home/weijh/model_weight/speculative_decoding/EAGLE-Vicuna-7B-v1.3"
    # args.add_argument("--base_model_path", type=str,
    #                     default="/data/home/weijh/models_weight/EMNLP/Eagle/vicuna-13b-v1.3",
    #                     help="Path to the pre-trained Medusa model.")
    # args.add_argument("--EAGLE_model_path", type=str,
    #                     default="/data/home/weijh/models_weight/EMNLP/Eagle/Eagle-vicuna-13b-v1.3",
    #                     help="Path to the pre-trained EAGLE model.")
    args.add_argument("--base_model_path", type=str,
                        default="/data/home/weijh/models_weight/EMNLP/Eagle/vicuna-13b-v1.3",
                        help="Path to the pre-trained Medusa model.")
    args.add_argument("--EAGLE_model_path", type=str,
                        default="/data/home/weijh/models_weight/EMNLP/Eagle/Eagle-vicuna-13b-v1.3",
                        help="Path to the pre-trained EAGLE model.")
    args.add_argument("--medusa_num_heads", type=int, default=5,
                        help="Number of medusa heads.")
    args.add_argument("--data_path", type=str,
                        default="/data/home/weijh/zyh/毕设/BM2.5/data/MIX/shuffled_file.jsonl",
                        help="Path to the evaluation data in JSON format.")
    # args.add_argument("--data_path", type=str,
    #                     default="/data/home/weijh/project/EM/data/Mt_bench/question.jsonl",
    #                     help="Path to the evaluation data in JSON format.")
    args.add_argument("--save_path", type=str, default="/data/home/weijh/project/EAGLE_TEST/EAGLE_batch/MT-bench/coutput.jsonl",
                        help="Directory to save the results.")
    args.add_argument("--max_batch_size", default=20, type=int,
                        help="max Batch size for request")
    args.add_argument("--max_gen_len", default=900, type=int,
                        help="Maximum length of generated text.")
    args.add_argument("--temperature", default=0.0, type=float)
    args.add_argument("--Node_choice",default="L20_7b",type=str)
    args.add_argument("--All_request_num", default=50, type=int)
    args.add_argument("--Request_speed", default=3, type=float,
                        help="num of request per second")
    args = args.parse_args()

    if args.Node_choice =="L20_7b":
        Node_choice = L20_7B_batch_node

    device_index = torch.cuda.device_count() - 1
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    Eagle = Parallel_decoding(args.base_model_path, args.EAGLE_model_path, Node_choice, Device_index=device_index)
    Eagle.param_init(args.max_batch_size, args.max_gen_len)  # 初始化Kvcahe
    RP = RequestProcessor(args, tokenizer, Eagle)
    
    asyncio.run(RP.main())
    RP.eval()