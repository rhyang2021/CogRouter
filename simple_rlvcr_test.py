#!/usr/bin/env python3
"""
超简单版本：测试RLVCR thinking advantage
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# 加载模型
model_path = "/apdcephfs_cq11_1567347/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_cog-sft_alf_lr2e6_bs16_epoch5_full_0810"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()

# 你的数据 - 构建完整的history
history = """Your task is to measure the temperature of unknown substance B, which is located around the bathroom. First, focus on the thermometer. Next, focus on the unknown substance B. If the unknown substance B temperature is above 200.0 degrees celsius, place it in the green box. If the unknown substance B temperature is below 200.0 degrees celsius, place it in the blue box. The boxes are located around the living room.
This room is called the bathroom. In it, you see:   the agent  a substance called air  a bathtub, which is turned off. In the bathtub is: nothing.  a glass cup (containing nothing)  a painting  a sink, which is turned off. In the sink is: nothing.  a toilet. In the toilet is: A drain, which is open, a substance called water.  unknown substance B You also see:  A door to the kitchen (that is closed)

You open the door to the kitchen

Now it's your turn to generate next step response."""
action = "go to kitchen"

# 定义4种thinking
thinkings = {
    1: "",  # Level 1没有thinking
    2: "Goal: Measure substance B temperature and place it in correct box. I need to go to kitchen to continue exploring.",
    3: "Goal: Measure temperature of substance B and place in correct box based on temperature. Current state: In bathroom, opened kitchen door. Available actions: Go to kitchen to look for thermometer. Reflection: Need thermometer first to measure temperature. Response: Go to kitchen to find thermometer.",
    4: "Goal: The goal is to measure the temperature of unknown substance B and place it in the appropriate box based on temperature (green if >200°C, blue if <200°C). Current state: I am in the bathroom where substance B is located, and I have opened the door to the kitchen. Available actions: I can go to the kitchen to search for a thermometer, or continue exploring other rooms. Reflection: The task requires measuring temperature first, so I need to find a thermometer before I can measure substance B. The kitchen is a logical place to look for measuring instruments. Response: I should go to the kitchen to search for a thermometer that I can use to measure the temperature of substance B."
}

print(f"History: {history[:80]}...")
print(f"目标行动: {action}")
print("="*50)

results = {}

# 测试每个level
for level in [1, 2, 3, 4]:
    print(f"\n--- Level {level} ---")
    
    # 构建完整的input (history + response)
    if level == 1:
        response = f"<level>1</level><action>{action}</action>"
        full_input = history + response
    else:
        thinking = thinkings[level]
        response = f"<level>{level}</level><think>{thinking}</think><action>{action}</action>"
        full_input = history + response
    
    print(f"Response: {response[:80]}...")
    
    # 计算thinking cost (token数量)
    thinking_text = thinkings[level]
    thinking_cost = len(tokenizer.encode(thinking_text)) if thinking_text else 0
    print(f"Thinking cost: {thinking_cost} tokens")
    
    # 计算action confidence (log probability)
    prefix = full_input.split(f"<action>")[0] + "<action>"
    full_text = prefix + action
    
    # Tokenize
    inputs = tokenizer(full_text, return_tensors="pt")
    action_tokens = tokenizer(action, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
    
    # 计算logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
    
    # 计算action tokens的probabilities
    prefix_len = len(tokenizer(prefix, return_tensors="pt")['input_ids'][0])
    probs_list = []
    for i, token_id in enumerate(action_tokens):
        pos = prefix_len + i - 1
        if pos >= 0 and pos < logits.shape[0]:
            token_logits = logits[pos]
            probs = torch.softmax(token_logits, dim=-1)
            prob = probs[token_id].item()
            probs_list.append(torch.log(torch.tensor(prob)).item())
    
    confidence = torch.exp(torch.tensor(np.mean(probs_list))).item() if probs_list else 0.0
    print(f"Confidence: {confidence:.4f}")
    
    results[level] = {
        'thinking_cost': thinking_cost,
        'confidence': confidence
    }

print("\n" + "="*50)
print("=== RLVCR Thinking Reward计算 ===")

# RLVCR参数
thinking_cost_alpha = 1.0
cost_max = 250

# 计算每个level的thinking reward
thinking_rewards = []
for level in [1, 2, 3, 4]:
    cost = results[level]['thinking_cost']
    confidence = results[level]['confidence']
    
    # R_entropy: confidence已经是[0,1]的概率值
    R_entropy = confidence
    
    # R_length: cost penalty
    normalized_cost = min(1.0, cost / cost_max)
    R_length = 0.5 - normalized_cost
    
    # 总reward
    reward = R_entropy + thinking_cost_alpha * R_length
    thinking_rewards.append(reward)
    
    print(f"Level {level}: R_entropy={R_entropy:.3f}, R_length={R_length:.3f}, reward={reward:.3f}")

print("\n--- Thinking Advantages (零均值) ---")
mean_reward = np.mean(thinking_rewards)
for level in [1, 2, 3, 4]:
    advantage = thinking_rewards[level-1] - mean_reward
    print(f"Level {level} advantage: {advantage:+.4f}")

# 找最优level
best_idx = np.argmax(thinking_rewards)
best_level = best_idx + 1
print(f"\n最优thinking level: {best_level}")
print(f"Advantage范围: {max(thinking_rewards) - min(thinking_rewards):.4f}")
