#!/bin/bash

# 设置路径变量
HF_MODEL_PATH="/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/llama3.1-8b_cog-sft_balance_sci_lr2e6_bs32_epoch3_full_1021/checkpoint-1500" # 原始HuggingFace模型路径（用于获取config）
LOCAL_DIR="/apdcephfs_cq11/share_1567347/share_info/rhyang/RLVMR/checkpoints/rlvcr_sciworld/gigpo_llama3.1-8b_sci_balance_cold_start_verl_v2_lr2e7_bs32_kl0.05_1108/global_step_130/actor"  # 您的checkpoint目录
TARGET_DIR="/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/llama3.1-8b_sci_cold_start_gigpo_kl0.05_1108_ckp130"  # 目标保存路径

# 创建目标目录
mkdir -p $TARGET_DIR

# 运行转换命令
python model_merger.py \
    --backend fsdp \
    --hf_model_path $HF_MODEL_PATH \
    --local_dir $LOCAL_DIR \
    --target_dir $TARGET_DIR

echo "转换完成！模型已保存到: $TARGET_DIR"