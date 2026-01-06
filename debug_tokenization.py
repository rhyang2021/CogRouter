#!/usr/bin/env python3
"""
简单检查tokenization是否一致
"""
from transformers import AutoTokenizer

model_path = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_cog-sft_alf_lr2e6_bs16_epoch5_full_0810"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 测试action
action = "go to cabinet 1"
print(f"Action: '{action}'")

# 方法1: 直接tokenize action
action_tokens_direct = tokenizer.encode(action, add_special_tokens=False)
print(f"Direct tokenization: {action_tokens_direct}")
print(f"Token texts: {[tokenizer.decode([t]) for t in action_tokens_direct]}")

# 方法2: 在context中tokenize
prefix_part = "Response: Given that counters might have fruit stored on them for easy access, going to a countertop is a logical choice next to find an apple.</think><action>"
full_text = prefix_part + action

prefix_tokens = tokenizer.encode(prefix_part, add_special_tokens=False)
full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

action_start = len(prefix_tokens)
action_tokens_context = full_tokens[action_start:]

print(f"\nContext tokenization:")
print(f"Prefix tokens: {len(prefix_tokens)}")
print(f"Full tokens: {len(full_tokens)}")
print(f"Action in context: {action_tokens_context}")
print(f"Token texts: {[tokenizer.decode([t]) for t in action_tokens_context]}")

print(f"\nMatch: {action_tokens_direct == action_tokens_context}")

# 检查RLVCR期望的token IDs
expected_ids = [3346, 311, 21921, 220, 16]
print(f"\nExpected from RLVCR: {expected_ids}")
print(f"Expected texts: {[tokenizer.decode([t]) for t in expected_ids]}")
print(f"Matches direct: {action_tokens_direct == expected_ids}")
print(f"Matches context: {action_tokens_context == expected_ids}")
