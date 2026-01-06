#!/bin/bash

# 测试thinking对action预测影响的脚本

MODEL_PATH="/apdcephfs_cq11_1567347/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_cog-sft_alf_lr2e6_bs16_epoch5_full_0810"

echo "开始测试thinking对action预测的影响..."

python test_thinking_impact.py \
    --model_path $MODEL_PATH \
    --output_file thinking_impact_results.json

echo "测试完成！"
echo "查看结果: cat thinking_impact_results.json"
