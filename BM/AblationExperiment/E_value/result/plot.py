import json
import matplotlib.pyplot as plt
import seaborn as sns

# 设置科研绘图风格
plt.style.use('seaborn-whitegrid')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.unicode_minus'] = False

# 读取JSONL文件
x_values = []
y_values = []

with open('data.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        for key, ys in data.items():
            x = float(key)  # 假设x存储为字符串类型的数值
            x_values.extend([x] * len(ys))
            y_values.extend(ys)

# 创建图表
plt.figure(figsize=(8, 6))
plt.scatter(x_values, y_values, 
           s=50,            # 点大小
           alpha=0.6,       # 透明度
           edgecolor='k',   # 边缘颜色
           linewidths=0.5,  # 边缘线宽
           c='#1f77b4')     # 点颜色

# 添加标签和标题
plt.xlabel('X Axis Label', fontsize=14, labelpad=10)
plt.ylabel('Y Axis Label', fontsize=14, labelpad=10)
plt.title('Scatter Plot Title', fontsize=16, pad=20)

# 优化布局并保存/显示
plt.tight_layout()
plt.savefig('scatter_plot.pdf', bbox_inches='tight')  # 保存为矢量图
plt.show()
