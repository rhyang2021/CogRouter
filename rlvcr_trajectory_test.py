#!/usr/bin/env python3
"""
RLVCR Trajectory测试：从实际trajectory数据中构造thinking process并计算advantage
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from typing import Dict, List, Tuple

# 加载模型
model_path = "/apdcephfs_cq11_1567347/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_cog-sft_alf_lr2e6_bs16_epoch5_full_0810"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()

def extract_thinking_and_action(step_text: str) -> Tuple[str, str, int]:
    """从step中提取thinking, action和level"""
    # 提取level
    level_match = re.search(r'<level>(\d+)</level>', step_text)
    level = int(level_match.group(1)) if level_match else 1
    
    # 提取thinking
    think_match = re.search(r'<think>(.*?)</think>', step_text, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""
    
    # 提取action
    action_match = re.search(r'<action>(.*?)</action>', step_text)
    action = action_match.group(1).strip() if action_match else ""
    
    return thinking, action, level

def build_history_context(trajectory_data: List[Dict], step_idx: int) -> str:
    """构建到指定step为止的完整history"""
    context_parts = []
    
    # 添加任务描述
    if trajectory_data:
        first_step = trajectory_data[0]
        context_parts.append(first_step['task_description'])
        context_parts.append(first_step['next_obs'])
    
    # 添加之前所有步骤的action和observation
    for i in range(step_idx):
        step = trajectory_data[i]
        # 只添加action部分，不包括thinking
        _, action, _ = extract_thinking_and_action(step['next_step'])
        if action:
            context_parts.append(f"You {action}")
        if i + 1 < len(trajectory_data):
            context_parts.append(trajectory_data[i + 1]['next_obs'])
    
    context_parts.append("Now it's your turn to generate next step response.")
    return "\n\n".join(context_parts)

# RLVCR thinking level prompts (from rlvcr/prompt.py)
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

def generate_alternative_thinking(history: str, action: str, original_thinking: str, original_level: int, target_level: int) -> str:
    """使用RLVCR prompt让模型生成指定level的thinking"""
    if target_level == 1:
        return ""
    elif target_level == original_level:
        return original_thinking
    
    # 清理history格式
    clean_history = history.split("Now it's your turn to generate next step response.")[0].strip()
    
    # 选择对应的prompt template
    if target_level == 2:
        prompt = THINK_MODE_2.format(history=clean_history, action=action)
    elif target_level == 3:
        prompt = THINK_MODE_3.format(history=clean_history, action=action)
    elif target_level == 4:
        prompt = THINK_MODE_4.format(history=clean_history, action=action)
    else:
        return ""
    
    # 使用模型生成thinking
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
        
        # 解码生成的文本
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取thinking部分
        thinking_match = re.search(r'<think>(.*?)</think>', generated_text, re.DOTALL)
        if thinking_match:
            return thinking_match.group(1).strip()
        else:
            # 如果没找到完整的<think>标签，返回生成的部分
            prompt_end = len(prompt)
            if len(generated_text) > prompt_end:
                generated_part = generated_text[prompt_end:].strip()
                # 简单清理
                if generated_part.startswith('<think>'):
                    generated_part = generated_part[7:]
                if generated_part.endswith('</think>'):
                    generated_part = generated_part[:-8]
                return generated_part.strip()
            return ""
            
    except Exception as e:
        print(f"Error generating thinking for level {target_level}: {e}")
        return f"Generated thinking for level {target_level} (simplified)"

def compute_confidence(history: str, thinking: str, action: str, level: int) -> float:
    """计算action confidence (geometric mean of token probabilities)"""
    # 构建完整的input
    if level == 1:
        response = f"<level>1</level><action>{action}</action>"
    else:
        response = f"<level>{level}</level><think>{thinking}</think><action>{action}</action>"
    
    full_input = history + response
    prefix = full_input.split(f"<action>")[0] + "<action>"
    
    # Tokenize
    try:
        inputs = tokenizer(prefix + action, return_tensors="pt")
        action_tokens = tokenizer(action, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
        
        # 计算logits
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
        
        # 计算action tokens的probabilities
        prefix_len = len(tokenizer(prefix, return_tensors="pt")['input_ids'][0])
        log_probs = []
        
        for i, token_id in enumerate(action_tokens):
            pos = prefix_len + i - 1
            if pos >= 0 and pos < logits.shape[0]:
                token_logits = logits[pos]
                probs = torch.softmax(token_logits, dim=-1)
                prob = probs[token_id].item()
                log_probs.append(torch.log(torch.tensor(prob + 1e-10)).item())
        
        # 几何平均 (geometric mean)
        if log_probs:
            confidence = torch.exp(torch.tensor(np.mean(log_probs))).item()
            return min(max(confidence, 0.0), 1.0)  # clamp到[0,1]
        else:
            return 0.0
            
    except Exception as e:
        print(f"Error computing confidence: {e}")
        return 0.0

def analyze_trajectory_step(trajectory_data: List[Dict], step_idx: int):
    """分析trajectory中的一个step"""
    if step_idx >= len(trajectory_data):
        print(f"Step {step_idx} out of range")
        return
    
    step = trajectory_data[step_idx]
    original_thinking, action, original_level = extract_thinking_and_action(step['next_step'])
    
    print(f"\n{'='*60}")
    print(f"=== Step {step_idx}: {action} ===")
    print(f"Original Level: {original_level}")
    print(f"Score: {step['score']}")
    print(f"Original thinking: {original_thinking[:100]}..." if len(original_thinking) > 100 else f"Original thinking: {original_thinking}")
    
    # 构建history context
    history = build_history_context(trajectory_data, step_idx)
    print(f"History length: {len(history)} chars")
    
    # 测试4个thinking levels
    results = {}
    thinking_rewards = []
    
    for level in [1, 2, 3, 4]:
        # 生成对应level的thinking
        if level == original_level:
            thinking = original_thinking
        else:
            thinking = generate_alternative_thinking(history, action, original_thinking, original_level, level)
        
        # 计算thinking cost
        thinking_cost = len(tokenizer.encode(thinking)) if thinking else 0
        
        # 计算confidence
        confidence = compute_confidence(history, thinking, action, level)
        
        # RLVCR reward计算
        R_entropy = confidence
        normalized_cost = min(1.0, thinking_cost / 250)  # cost_max = 250
        R_length = 0.5 - normalized_cost
        reward = R_entropy + 1.0 * R_length  # thinking_cost_alpha = 1.0
        
        thinking_rewards.append(reward)
        results[level] = {
            'thinking': thinking,
            'thinking_cost': thinking_cost,
            'confidence': confidence,
            'reward': reward
        }
        
        print(f"\nLevel {level}:")
        print(f"  Thinking: {thinking[:80]}..." if len(thinking) > 80 else f"  Thinking: {thinking}")
        print(f"  Cost: {thinking_cost} tokens")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Reward: R_entropy({R_entropy:.3f}) + R_length({R_length:.3f}) = {reward:.3f}")
    
    # 计算advantages
    print(f"\n--- Thinking Advantages (zero-mean) ---")
    mean_reward = np.mean(thinking_rewards)
    advantages = []
    for level in [1, 2, 3, 4]:
        advantage = thinking_rewards[level-1] - mean_reward
        advantages.append(advantage)
        marker = " ⭐" if level == original_level else ""
        print(f"Level {level} advantage: {advantage:+.4f}{marker}")
    
    # 分析结果
    best_level = np.argmax(thinking_rewards) + 1
    original_advantage = advantages[original_level - 1]
    best_advantage = max(advantages)
    
    print(f"\n--- Analysis ---")
    print(f"Original level {original_level} advantage: {original_advantage:+.4f}")
    print(f"Best level {best_level} advantage: {best_advantage:+.4f}")
    print(f"Advantage range: {max(thinking_rewards) - min(thinking_rewards):.4f}")
    
    improvement_found = best_level != original_level and best_advantage > original_advantage
    print(f"Better thinking found: {'Yes' if improvement_found else 'No'}")
    
    return {
        'step_idx': step_idx,
        'original_level': original_level,
        'best_level': best_level,
        'original_advantage': original_advantage,
        'best_advantage': best_advantage,
        'improvement_found': improvement_found,
        'advantage_range': max(thinking_rewards) - min(thinking_rewards)
    }

def load_trajectory(file_path: str) -> List[Dict]:
    """加载trajectory数据"""
    trajectory = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                trajectory.append(json.loads(line.strip()))
        return trajectory
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return []

# 主程序
if __name__ == "__main__":
    # 你的trajectory文件路径
    trajectory_path = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/results/test/sciworld/qwen2.5-7b_cog_cold_start_grpo_0830_ckp50_mode5/1756447897/variation-537.jsonl"
    
    print("Loading trajectory...")
    trajectory = load_trajectory(trajectory_path)
    
    if not trajectory:
        print("Failed to load trajectory data")
        exit(1)
    
    print(f"Loaded trajectory with {len(trajectory)} steps")
    print(f"Task: {trajectory[0]['task_name']}")
    
    # 分析多个关键步骤
    key_steps = [6, 12, 14]  # 分析这几个有代表性的steps
    results = []
    
    for step_idx in key_steps:
        if step_idx < len(trajectory):
            result = analyze_trajectory_step(trajectory, step_idx)
            if result:
                results.append(result)
    
    # 总结分析
    print(f"\n{'='*60}")
    print("=== Overall Analysis ===")
    
    improvement_count = sum(1 for r in results if r['improvement_found'])
    improvement_ratio = improvement_count / len(results) if results else 0
    
    print(f"Steps analyzed: {len(results)}")
    print(f"Steps with better thinking found: {improvement_count}")
    print(f"Improvement ratio: {improvement_ratio:.2%}")
    
    if results:
        avg_advantage_range = np.mean([r['advantage_range'] for r in results])
        print(f"Average advantage range: {avg_advantage_range:.4f}")
        
        # 显示每个步骤的结果
        for r in results:
            status = "✓ Improved" if r['improvement_found'] else "✗ No improvement"
            print(f"Step {r['step_idx']}: L{r['original_level']}→L{r['best_level']}, "
                  f"adv {r['original_advantage']:+.3f}→{r['best_advantage']:+.3f}, {status}")
