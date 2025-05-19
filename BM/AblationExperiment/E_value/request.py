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
    with open("/data/home/weijh/zyh/emnlp/BM/AblationExperiment/E_value/result/e13b.jsonl", 'a', encoding='utf-8') as f:
        json_line = json.dumps(data_by_type, ensure_ascii=False)
        f.write(json_line + '\n')
    # 计算每个类型的平均值
    average_by_type = {type_: sum(values) / len(values) for type_, values in data_by_type.items()}

    return average_by_type
class ModelRequest:
    def __init__(self, prompt,request_id):
        self.prompt = prompt
        self.request_id = request_id
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
                prompt=prompts[i],
                request_id=i
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
            # sentence = self.tokenizer.decode(request.output_ids[0])
        output ={"batch_size":self.args.max_batch_size,"request_speed":self.args.Request_speed,"average_latency":all_latency/len(self.completed_stack),"throughput":all_count/(self.end_time-self.start_time),"Node_num":self.args.Node_num}
        print(calculate_average_by_type(self.model.acc_num_list,self.model.weight_list))
        print(output)
        # with open(self.args.save_path, 'a', encoding='utf-8') as f:
        #     json_line = json.dumps(output, ensure_ascii=False)
        #     f.write(json_line + '\n')
            
    def process_requests(self, requests_list):
        self.model_forward = True  # 模型开始运行不接受请求
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

        self.model_forward = False

    async def main(self):
        print("warm up")
        for _ in tqdm(range(1)):
            count =1
            warm_requests =self.sample_requests(count)#用五个请求进行设备预热
            while True:
                finsh_request_list=self.model.tree_spec_generate(warm_requests)
                count -=len(finsh_request_list)
                warm_requests = []
                if count ==0:
                    break
        print("warm up finished")
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

                if self.model.insert_flag ==False:
                    #收集请求
                    while len(self.batch) < self.free_num :
                        try:
                            request = queue.get_nowait()
                            if request is None:
                                producer_done = True
                            else:
                                self.batch.append(request)
                        except asyncio.QueueEmpty:#队列内无请求
                            break
                    #有请求需要插入释放插入信号
                    if len(self.batch)>0 :
                        self.model.insert_flag = True

                if self.model_forward ==False :# 模型不在运行的时候处理请求,有两种情况，一个是模型完成请求，另一个是模型被强制退出有新的请求
                    before = self.finsh_request_num
                    requset_list=self.batch
                    self.batch =[]
                    await loop.run_in_executor(
                        None, 
                        lambda: self.process_requests(requset_list)
                    )
                    self.model.insert_flag = False
                    # print("-"*50)
                    delta = self.finsh_request_num - before
                    pbar.update(delta)

                await asyncio.sleep(0.000005)

        await producer
        self.end_time =time.time()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Medusa Model Evaluator")
    
    args.add_argument("--model_path", type=str,
                        default="/data/home/weijh/models_weight/medusa/medusa_2_13b/Medusa_13B",
                        help="Path to the pre-trained Medusa model.")
    args.add_argument("--medusa_num_heads", type=int, default=5,
                        help="Number of medusa heads.")
    args.add_argument("--data_path", type=str,
                        default="/data/home/weijh/zyh/毕设/BM2.5/data/MIX/shuffled_file.jsonl",
                        help="Path to the evaluation data in JSON format.")
    args.add_argument("--save_path", type=str, default="/data/home/weijh/zyh/emnlp/BM/static/ours/result/output.jsonl",
                        help="Directory to save the results.")
    args.add_argument("--max_batch_size", default=30, type=int,
                        help="max Batch size for request")
    args.add_argument("--max_gen_len", default=700, type=int,
                        help="Maximum length of generated text.")
    args.add_argument("--temperature", default=0.0, type=float)
    args.add_argument("--Node_num",default=256,type=int)
    args.add_argument("--All_request_num", default=50, type=int)
    args.add_argument("--Request_speed", default=3, type=float,
                        help="num of request per second")
    args = args.parse_args()

    device_index = torch.cuda.device_count() - 1
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    Medusa = Parallel_decoding(args.model_path,args.Node_num,Device_index=device_index)
    Medusa.param_init(args.max_batch_size,args.max_gen_len)#初始化Kvcahe
    RP = RequestProcessor(args,tokenizer,Medusa)
    
    asyncio.run(RP.main())
    RP.eval()