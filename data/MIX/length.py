import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# 读取数据
with open('/data/home/weijh/zyh/毕设/BM1.0/data/MIX/shuffled_file.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

data = [json.loads(line) for line in lines]

# 加载分词器
model_path = "/data/home/weijh/models_weight/medusa/medusa-v1.0-vicuna-7b-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

# 统计句子长度
lengths = []
for i in data:
    input_sent = i["turns"][0]
    tokens = tokenizer.encode(input_sent, add_special_tokens=True)
    lengths.append(len(tokens))

# 配置科研级绘图参数
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.edgecolor': '0.2',
    'axes.spines.right': False,
    'axes.spines.top': False,
    'grid.color': '#e0e0e0',
    'grid.alpha': 0.3
})

# 创建画布
fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

# 绘制直方图
n, bins, patches = ax.hist(lengths, bins=30, 
                          color='#89CFF0', 
                          edgecolor='black',
                          linewidth=0.8,
                          alpha=0.5)

# 优化显示范围
max_freq = max(n)
ax.set_ylim(0, max_freq*1.05)

# 设置标题和标签
ax.set_title("Token Length Distribution of Request Sentences", 
            fontweight='bold', pad=20)
ax.set_xlabel("Token Length", labelpad=10)
ax.set_ylabel("Frequency", labelpad=10)

# 网格配置
ax.grid(True, which='major', linestyle='--', linewidth=0.5)

# 刻度优化
ax.tick_params(axis='both', which='both', length=5, pad=5)

# 调整布局
fig.tight_layout(pad=3)

# 保存图像
plt.savefig("/data/home/weijh/zyh/毕设/BM1.0/data/MIX/length_distribution.png", 
           dpi=300, 
           bbox_inches='tight',
           facecolor=fig.get_facecolor())

plt.close()