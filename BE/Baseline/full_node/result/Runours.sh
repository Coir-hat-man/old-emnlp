#!/bin/bash

# 请把这里改成你固定的max_batch_size值，例如15
MAX_BATCH_SIZE_FIXED=20

# 计算choose_ad_num数组，取整数（向下取整）
choose_ad_num_list=(
    $(( MAX_BATCH_SIZE_FIXED * 30 / 100 ))
    $(( MAX_BATCH_SIZE_FIXED * 60 / 100 ))
    $(( MAX_BATCH_SIZE_FIXED * 90 / 100 ))
)

# 计算总循环次数
count=0
total=0

for choose_ad_num in "${choose_ad_num_list[@]}"
do
    for Node_num in $(seq 100 20 300)
    do
        for Request_speed in $(seq 1 2 6)
        do
            ((total++))
        done
    done
done

# 进度条函数
print_progress() {
    local progress=$1
    local total=$2
    local width=50
    local percent=$(( progress*100/total ))
    local filled=$(( progress*width/total ))
    local empty=$(( width - filled ))

    printf "\rProgress : ["
    for ((i=0; i<filled; i++)); do printf "#"; done
    for ((i=0; i<empty; i++)); do printf "-"; done
    printf "] %d%% (%d/%d)" "$percent" "$progress" "$total"
}

count=0
for choose_ad_num in "${choose_ad_num_list[@]}"
do
    for Node_num in $(seq 100 20 300)
    do
        for Request_speed in $(seq 1 2 6)
        do
            ((count++))
            print_progress $count $total

            # 执行命令示例，替换成你实际的Python脚本命令
            echo " --max_batch_size $MAX_BATCH_SIZE_FIXED --choose_ad_num $choose_ad_num --Node_num $Node_num --Request_speed $Request_speed"

            python /home/zhouyh/homework/代码备份5.0实验配置/Batch_medusa/ours/request.py --max_batch_size $MAX_BATCH_SIZE_FIXED --choose_ad_num $choose_ad_num --Node_num $Node_num --Request_speed $Request_speed

            sleep 0.01  # 可选，模拟运行延迟
        done
    done
done

echo
