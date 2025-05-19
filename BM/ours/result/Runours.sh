#!/bin/bash

# 脚本名称：run_request.sh
#nohup bash /data/home/weijh/zyh/emnlp/BM/static/ours/result/Runours.sh > /data/home/weijh/zyh/emnlp/BM/static/ours/result/run_request.log 2>&1 &

# Python脚本路径
PYTHON_SCRIPT="/data/home/weijh/zyh/emnlp/BM/static/ours/request.py"


# 枚举所有参数组合
for speed in 1 2 3 4; do
    for batch_size in {10..40..5}; do
        for node_num in 138 207 256 276; do
            echo "Running with Request_speed=$speed, max_batch_size=$batch_size, Node_num=$node_num"
            
            # 运行Python脚本并将输出追加到文件
            CUDA_VISIBLE_DEVICES=2 python $PYTHON_SCRIPT \
                --Request_speed $speed \
                --max_batch_size $batch_size \
                --Node_num $node_num 
            
            # 在每个参数组合运行后添加分隔线
            echo "--------------------------------------------------" 
        done
    done
done

echo "All runs completed. "