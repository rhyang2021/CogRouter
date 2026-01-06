#!/usr/bin/env python3
"""
使用真实RLVCR日志中的例子测试token概率计算
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# 加载模型
model_path = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_cog-sft_alf_lr2e6_bs16_epoch5_full_0810"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map="auto")
model.eval()

# 使用真实RLVCR日志中的例子
prefix = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn.  
If you choose 'ACTION', you should directly output the action in this turn. Your output must strictly follow this format: 'Action: your next action'. After each turn, the environment will give you immediate feedback based on which you plan your next few steps. If the environment outputs 'Nothing happened', that means the previous action is invalid and you should try more options. 
Reminder: the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. Think when necessary, try to act directly more in the process.

There are four thinking levels:
Level 1 - Instinctive Response: Immediate reaction based on intuition, no analysis.
Level 2 - Situational Awareness: Assess current state and available actions before acting.
Level 3 - Experience Integration: Reflect on past actions and outcomes to inform current decisions.
Level 4 - Strategic Planning: Assess the task goal, past lessons, and current state to analyze the future impact of each candidate action and optimize the decision.

At each step, you must first choose an appropriate level of thinking (one of the four levels) to respond based on the given scenario. The chosen level MUST be enclosed within <level> </level> tags.
Next, reason through the problem step-by-step using the chosen thinking level. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.

[Output Format] 
Your output must adhere to the following format:
EXAMPLE 1:
<level>1</level>
<action>your_next_action</action>

EXAMPLE 2:
<level>2</level>
<think>
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Response: [Give a preliminary response]
</think>
<action>your_next_action</action>

EXAMPLE 3:
<level>3</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Response: [Give a preliminary response]
</think>
<action>your_next_action</action>

EXAMPLE 4:
<level>4</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Evaluation: [Assess the potential effectiveness of each candidate action]
Response: [Give a preliminary response]
</think>
<action>your_next_action</action>

-= Welcome to TextWorld, ALFRED! =-

You are in the middle of a room. Looking quickly around you, you see a cabinet 20, a cabinet 19, a cabinet 18, a cabinet 17, a cabinet 16, a cabinet 15, a cabinet 14, a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.

Your task is to: put a cool apple in microwave.

Here is the task history for past 10 steps:
Observation: -= Welcome to TextWorld, ALFRED! =-

You are in the middle of a room. Looking quickly around you, you see a cabinet 20, a cabinet 19, a cabinet 18, a cabinet 17, a cabinet 16, a cabinet 15, a cabinet 14, a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.

Your task is to: put a cool apple in microwave.
Output: <level>3</level>
<think>
Goal: The goal is to put a cool apple in the microwave.
Current state: I am in the middle of a room with various cabinets, appliances, and other kitchen furnishings. The environment includes a fridge and a microwave, both of which could potentially store a cool apple.
Available actions: I can move to various cabinets, the fridge, the microwave, and other appliances or use the inventory command to check if I already possess the apple.
Reflection: Since the task is to place a cool apple into the microwave, it suggests that I first need to locate a cool apple. From my knowledge, apples would likely be found in the fridge, as they need to remain cool.
Response: To find a cool apple most efficiently, the logical step is to go to the fridge, where fruits are usually kept cold and fresh.
</think>
<action>go to fridge 1</action>
Observation: You arrive at fridge 1. The fridge 1 is closed.
Output: For this situation, let's choose Level 2 thinking, as it requires assessing the current state and the available actions to make a decision.

<level>2</level>
<think>
Current state: I am currently in front of fridge 1, which is closed. 
Available actions: I can examine or open fridge 1. Other available actions include moving to other objects in the room, but those are irrelevant to finding a cool apple.
Response: To put a cool apple in the microwave, I first need to find the apple. The logical step is to open fridge 1 to check if there's a cool apple inside.
</think>
<action>open fridge 1</action>
Observation: You open the fridge 1. The fridge 1 is open. In it, you see a lettuce 1, a pot 1, and a tomato 3.
Output: <level>2</level>
<think>
Current state: The fridge is open, and I am standing in front of it. I can see three items inside: a lettuce, a pot, and a tomato. However, there is no apple present.
Available actions: I can close the fridge, examine its contents further, or consider other actions to look for another cooling place for the apple.
Response: Since there is no apple in the fridge and it seems unnecessary to look through irrelevant contents, closing the fridge is a reasonable next step to keep things organized before continuing my search elsewhere.
</think>
Action: close fridge 1
Observation: Nothing happens. AVAILABLE ACTIONS: 'close fridge 1'
 'examine fridge 1'
 'go to cabinet 1'
 'go to cabinet 10'
 'go to cabinet 11'
 'go to cabinet 12'
 'go to cabinet 13'
 'go to cabinet 14'
 'go to cabinet 15'
 'go to cabinet 16'
 'go to cabinet 17'
 'go to cabinet 18'
 'go to cabinet 19'
 'go to cabinet 2'
 'go to cabinet 20'
 'go to cabinet 3'
 'go to cabinet 4'
 'go to cabinet 5'
 'go to cabinet 6'
 'go to cabinet 7'
 'go to cabinet 8'
 'go to cabinet 9'
 'go to coffeemachine 1'
 'go to countertop 1'
 'go to countertop 2'
 'go to countertop 3'
 'go to drawer 1'
 'go to drawer 2'
 'go to drawer 3'
 'go to garbagecan 1'
 'go to microwave 1'
 'go to sinkbasin 1'
 'go to stoveburner 1'
 'go to stoveburner 2'
 'go to stoveburner 3'
 'go to stoveburner 4'
 'go to toaster 1'
 'inventory'
 'look'
 'take lettuce 1 from fridge 1'
 'take pot 1 from fridge 1'
 'take tomato 3 from fridge 1'

Now it's your turn to generate next step response.<level>3</level><think>Goal: The task is to put a cool apple in the microwave.
Current state: The fridge is open; however, it only contains lettuce, a pot, and tomatoes, none of which are apples.
Available actions: I can go to various cabinets, countertops, drawers, garbagecan, microwave, sinkbasin, stoveburners, or toaster. Examine these areas to potentially find an apple.
Reflection: I checked the fridge expecting to find a cool apple but discovered only lettuce, a pot, and tomatoes. This suggests apples might be stored in a different part of the room, likely the countertop or cabinets since these are typical storage places for apples.
Response: Given that counters might have fruit stored on them for easy access, going to a countertop is a logical choice next to find an apple.</think><action>"""

# 测试action: "go to cabinet 1"
# RLVCR日志显示的token IDs: [3346, 311, 21921, 220, 16]
# RLVCR日志显示的概率: ['0.6057', '0.7983', '0.0012', '0.5152', '0.3908']
action = "go to cabinet 1"

print("="*80)
print("🧪 测试RLVCR实际例子")
print("="*80)
print(f"Prefix ends with: ...{prefix[-100:]}")
print(f"Action: '{action}'")

# Tokenize action
action_token_ids = tokenizer.encode(action, add_special_tokens=False)
print(f"Action token IDs: {action_token_ids}")
print(f"Expected token IDs: [3346, 311, 21921, 220, 16]")
print(f"Token IDs match: {action_token_ids == [3346, 311, 21921, 220, 16]}")
print(f"Token texts: {[tokenizer.decode([t]) for t in action_token_ids]}")

# 计算概率 - 使用temperature=1.0模拟RLVCR的confidence计算
method1_probs = []
current_input = tokenizer.encode(prefix, add_special_tokens=False, return_tensors='pt').to(model.device)

print("\n🔍 逐token计算概率:")
for i, token_id in enumerate(action_token_ids):
    # 拼接当前token
    extended_input = torch.cat([current_input[0], torch.tensor([token_id]).to(model.device)]).unsqueeze(0)
    
    # 计算这个token的概率
    with torch.no_grad():
        outputs = model(extended_input)
        
        # 使用正确的位置：当前序列长度-1的位置预测下一个token
        current_len = current_input.shape[1]
        prediction_pos = current_len - 1
        logits = outputs.logits[0, prediction_pos, :]
        
        # 调试信息：检查logits
        token_text = tokenizer.decode([token_id])
        target_logit = logits[token_id].item()
        max_logit = logits.max().item()
        logit_range = (logits.max() - logits.min()).item()
        
        # 使用temperature=1.0获得真实概率分布（不是锐化的）
        temperature = 1.0
        probs = torch.softmax(logits / temperature, dim=-1)
        token_prob = probs[token_id].item()
        
        print(f"Token {i}: '{token_text}' (id={token_id}) -> prob={token_prob:.4f}")
        print(f"  Logit={target_logit:.2f}, Max_logit={max_logit:.2f}, Range={logit_range:.2f}")
        
        # 如果概率异常，显示top tokens
        if token_prob < 1e-5 or token_prob > 0.999:
            print(f"   ⚠️ 异常概率! Top 3 tokens:")
            top_probs, top_indices = torch.topk(probs, k=3)
            for j, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                top_token_text = tokenizer.decode([idx])
                marker = " ← TARGET" if idx == token_id else ""
                print(f"     {j+1}. '{top_token_text}': {prob:.4f}{marker}")
    
    method1_probs.append(token_prob)
    
    # 更新current_input用于下一个token的自回归预测
    current_input = extended_input

print(f"\n📊 结果:")
print(f"我们计算的概率: {[f'{p:.4f}' for p in method1_probs]}")
print(f"RLVCR日志的概率: ['0.6057', '0.7983', '0.0012', '0.5152', '0.3908']")

# 计算几何平均（RLVCR使用的confidence指标）
import math
log_probs = [math.log(p + 1e-10) for p in method1_probs]
geometric_mean = math.exp(sum(log_probs) / len(log_probs))
min_prob = min(method1_probs)

print(f"\n📈 Confidence计算:")
print(f"几何平均 (geometric mean): {geometric_mean:.6f}")
print(f"最小概率 (old method): {min_prob:.6f}")

print("\n" + "="*80)
