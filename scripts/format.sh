#!/bin/bash
pip3 install --upgrade yapf
python3 -m yapf -ir -vv --style ./.style.yapf verl tests examples recipe


python model_merger.py \
    --backend fsdp \
    --hf_model_path /apdcephfs_cq11_1567347/share_1567347/share_info/llm_models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
    --local_dir /apdcephfs_cq11_1567347/share_1567347/share_info/rhyang/arpo_qwen2.5_7b_sci_cog_0815/global_step_120/actor \
    --target_dir /apdcephfs_cq11_1567347/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_sci_cog_grpo_0815_ckp120