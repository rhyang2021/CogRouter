#!/bin/bash

# 设置路径变量
HF_MODEL_PATH="/apdcephfs_cq11_1567347/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_cog-sft_sci_lr2e6_bs16_epoch5_full_0810"  # 原始HuggingFace模型路径（用于获取config）
LOCAL_DIR="/apdcephfs_cq11_1567347/share_1567347/share_info/rhyang/RLVMR/checkpoints/mcrl_sciworld/grpo_qwen2.5_7b_sci_cog_cold_start_0827/global_step_40/actor"  # 您的checkpoint目录
TARGET_DIR="/apdcephfs_cq11_1567347/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_sci_cog_code_start_0827_ckp40"  # 目标保存路径

# 创建目标目录
mkdir -p $TARGET_DIR

# 运行转换命令
python model_merger.py \
    --backend fsdp \
    --hf_model_path $HF_MODEL_PATH \
    --local_dir $LOCAL_DIR \
    --target_dir $TARGET_DIR

echo "转换完成！模型已保存到: $TARGET_DIR"