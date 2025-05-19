#!/bin/bash

start=0.5
end=5
step=0.5

echo "开始循环，从 $start 到 $end，步长 $step"

mkdir output

for current in $(awk "BEGIN{for(i=$start; i<=$end; i+=$step) print i}")
do
    echo "当前值: $current"
    
    CUDA_VISIBLE_DEVICES=2 python request.py --Request_speed ${current} > ./output/output_time_request_rate_${current}_our.csv
    
    sleep 0.3  # 控制请求间隔
done

echo "循环结束"