from transformers import AutoTokenizer

# 加载原始tokenizer
tokenizer = AutoTokenizer.from_pretrained("/apdcephfs_cq11_1567347/share_1567347/share_info/llm_models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")

# 保存到转换后的模型目录
tokenizer.save_pretrained("/apdcephfs_cq11_1567347/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_grpo_sci_lr1e5_bs128_step170_0808")