import json
import random

with open('/home/zhouyh/homework/代码备份5.0实验配置/Batch_medusa/data/MIX/torchquestion.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

data = [json.loads(line) for line in lines]

random.shuffle(data)

with open('/home/zhouyh/homework/代码备份5.0实验配置/Batch_medusa/data/MIX/shuffled_file.jsonl', 'w', encoding='utf-8') as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
