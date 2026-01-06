#!/usr/bin/env python3
"""
快速版本：直接使用你提供的trajectory数据测试RLVCR
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re

# 加载模型
model_path = "/apdcephfs_cq11_1567347/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_cog-sft_alf_lr2e6_bs16_epoch5_full_0810"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()

# 从你的数据中选择一个高分的step进行分析
# Step 12: focus on unknown substance B in inventory (score: 85)
history = """Your task is to measure the temperature of unknown substance B, which is located around the bathroom. First, focus on the thermometer. Next, focus on the unknown substance B. If the unknown substance B temperature is above 50.0 degrees celsius, place it in the yellow box. If the unknown substance B temperature is below 50.0 degrees celsius, place it in the purple box. The boxes are located around the bathroom.

This room is called the greenhouse. In it, you see:   the agent  a substance called air  a bee hive. The bee hive door is closed.   a jug (containing nothing)  a shovel  a sink, which is turned off. In the sink is: nothing. You also see:  A door to the hallway (that is closed)  A door to the outside (that is closed)

You open door to hallway
You move to the hallway.
You open door to kitchen
You move to the kitchen.
You look around
You move the thermometer to the inventory.
You focus on the thermometer.
You look around
You open door to bathroom
You move to the bathroom.
You look around
You move the unknown substance B to the inventory.

Now it's your turn to generate next step response."""

action = "focus on unknown substance B in inventory"
original_thinking = "Goal: The task is to measure the temperature of unknown substance B and then place it in the appropriate box based on its temperature.\nCurrent state: I am currently in the bathroom, and I have moved the unknown substance B to my inventory. I also have the thermometer in my inventory.\nAvailable actions: I can focus on the unknown substance B in my inventory, use the thermometer on it, or interact with other objects in the bathroom.\nReflection: I have already picked up the thermometer and moved the unknown substance B to my inventory. The next logical step is to focus on the unknown substance B to proceed with measuring its temperature.\nResponse: By focusing on the unknown substance B in my inventory, I can prepare to measure its temperature, which is necessary to determine which box to place it in."
original_level = 3

# RLVCR thinking level prompts
THINK_MODE_2 = """Given the current state and available actions, assess the situation before acting:

{history}

Now use Level 2 thinking (Situational Awareness) to analyze the current state and choose the best action "{action}". 

<level>2</level>
<think>
Current state: [Analyze the current environment state]
Available actions: [What actions are possible right now]
Response: [Choose the best action and explain why]
</think>
<action>{action}</action>"""

THINK_MODE_3 = """Given the current state and available actions, reflect on past experiences to inform your decision:

{history}

Now use Level 3 thinking (Experience Integration) to reflect on past actions and outcomes before choosing action "{action}".

<level>3</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Analyze the current environment state]
Available actions: [What actions are possible right now]
Reflection: [How effective were recent actions, what was learned]
Response: [Choose the best action based on experience]
</think>
<action>{action}</action>"""

THINK_MODE_4 = """Given the current state and available actions, strategically plan and evaluate each option:

{history}

Now use Level 4 thinking (Strategic Planning) to assess the task goal, past lessons, and current state to analyze the future impact of action "{action}".

<level>4</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Analyze the current environment state]
Available actions: [What actions are possible right now]
Reflection: [How effective were recent actions, what was learned]
Evaluation: [Assess the potential effectiveness of each candidate action]
Response: [Choose the optimal action with strategic reasoning]
</think>
<action>{action}</action>"""

def generate_thinking(history: str, action: str, target_level: int) -> str:
    """使用RLVCR prompt生成指定level的thinking"""
    if target_level == 1:
        return ""
    
    clean_history = history.split("Now it's your turn to generate next step response.")[0].strip()
    
    if target_level == 2:
        prompt = THINK_MODE_2.format(history=clean_history, action=action)
    elif target_level == 3:
        prompt = THINK_MODE_3.format(history=clean_history, action=action)
    elif target_level == 4:
        prompt = THINK_MODE_4.format(history=clean_history, action=action)
    else:
        return ""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取thinking部分
        thinking_match = re.search(r'<think>(.*?)</think>', generated_text, re.DOTALL)
        if thinking_match:
            return thinking_match.group(1).strip()
        else:
            # fallback解析
            prompt_len = len(tokenizer.decode(tokenizer(prompt, return_tensors="pt")['input_ids'][0], skip_special_tokens=True))
            if len(generated_text) > prompt_len:
                generated_part = generated_text[prompt_len:].strip()
                if generated_part.startswith('<think>'):
                    generated_part = generated_part[7:]
                if generated_part.endswith('</think>'):
                    generated_part = generated_part[:-8]
                return generated_part.strip()
            return ""
            
    except Exception as e:
        print(f"Error generating thinking for level {target_level}: {e}")
        return f"Generated thinking for level {target_level}"

def compute_confidence(full_context: str, action: str) -> float:
    """计算action confidence"""
    prefix = full_context.split(f"<action>")[0] + "<action>"
    
    try:
        inputs = tokenizer(prefix + action, return_tensors="pt")
        action_tokens = tokenizer(action, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
        
        prefix_len = len(tokenizer(prefix, return_tensors="pt")['input_ids'][0])
        log_probs = []
        
        for i, token_id in enumerate(action_tokens):
            pos = prefix_len + i - 1
            if pos >= 0 and pos < logits.shape[0]:
                token_logits = logits[pos]
                probs = torch.softmax(token_logits, dim=-1)
                prob = probs[token_id].item()
                log_probs.append(torch.log(torch.tensor(prob + 1e-10)).item())
        
        if log_probs:
            confidence = torch.exp(torch.tensor(np.mean(log_probs))).item()
            return min(max(confidence, 0.0), 1.0)
        else:
            return 0.0
            
    except Exception as e:
        print(f"Error computing confidence: {e}")
        return 0.0

print(f"原始step: focus on unknown substance B in inventory")
print(f"原始level: {original_level}")
print(f"原始thinking: {original_thinking[:100]}...")
print("="*60)

# 测试每个level
results = {}
thinking_rewards = []

for level in [1, 2, 3, 4]:
    print(f"\n--- Level {level} ---")
    
    # 生成对应level的thinking
    if level == original_level:
        thinking = original_thinking
    else:
        print(f"Generating Level {level} thinking...")
        thinking = generate_thinking(history, action, level)
    
    # 构建完整response
    if level == 1:
        response = f"<level>1</level><action>{action}</action>"
    else:
        response = f"<level>{level}</level><think>{thinking}</think><action>{action}</action>"
    
    full_context = history + response
    
    # 计算thinking cost
    thinking_cost = len(tokenizer.encode(thinking)) if thinking else 0
    print(f"Thinking: {thinking[:80]}..." if len(thinking) > 80 else f"Thinking: {thinking}")
    print(f"Thinking cost: {thinking_cost} tokens")
    
    # 计算confidence
    confidence = compute_confidence(full_context, action)
    print(f"Confidence: {confidence:.4f}")
    
    # RLVCR reward计算
    R_entropy = confidence
    normalized_cost = min(1.0, thinking_cost / 250)  # cost_max = 250
    R_length = 0.5 - normalized_cost
    reward = R_entropy + 1.0 * R_length  # thinking_cost_alpha = 1.0
    
    thinking_rewards.append(reward)
    results[level] = {
        'thinking_cost': thinking_cost,
        'confidence': confidence,
        'reward': reward
    }
    
    print(f"R_entropy: {R_entropy:.3f}")
    print(f"R_length: {R_length:.3f} (cost penalty)")
    print(f"Total reward: {reward:.3f}")

print(f"\n{'='*60}")
print("=== RLVCR Advantage Analysis ===")

# 计算advantages (zero-mean)
mean_reward = np.mean(thinking_rewards)
print(f"Mean reward: {mean_reward:.4f}")

advantages = []
for level in [1, 2, 3, 4]:
    advantage = thinking_rewards[level-1] - mean_reward
    advantages.append(advantage)
    marker = " ⭐ (Original)" if level == original_level else ""
    print(f"Level {level} advantage: {advantage:+.4f}{marker}")

# 分析结果
best_level = np.argmax(thinking_rewards) + 1
original_advantage = advantages[original_level - 1]
best_advantage = max(advantages)

print(f"\n--- Results ---")
print(f"Original Level {original_level} advantage: {original_advantage:+.4f}")
print(f"Best Level {best_level} advantage: {best_advantage:+.4f}")
print(f"Advantage range: {max(advantages) - min(advantages):.4f}")

if best_level != original_level and best_advantage > original_advantage:
    improvement = best_advantage - original_advantage
    print(f"✓ Better thinking found! Level {best_level} improves by {improvement:+.4f}")
else:
    print(f"✗ Original thinking is optimal")

# 显示最优thinking
print(f"\n--- Best Thinking (Level {best_level}) ---")
if best_level == 1:
    print("(No thinking - Level 1)")
elif best_level == original_level:
    print(f"{original_thinking[:200]}...")
else:
    # 重新生成最优level的thinking用于显示
    best_thinking = generate_thinking(history, action, best_level)
    print(f"{best_thinking[:200]}...")
