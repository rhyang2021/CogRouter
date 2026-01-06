#!/usr/bin/env python3
"""
RLVCR Trajectory测试：批量处理多个jsonl文件，为每个step添加4个thinking level信息
修复版本 - 解决 batch_compute_confidence 函数的 return 语句位置问题
"""
import os
import glob
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
import pdb
from typing import Dict, List, Tuple

"""
RLVCR Thinking Level Prompts
Defines different thinking level templates for generating alternative thinking patterns
"""

ALFWORLD_TEMPLATE_ADA = """
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. 
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

{task_description}

Here is the task history
{action_history}

Now it's your turn to generate next step response.
""".strip()


THINK_MODE_2 = """
You are given a specific next step action that has already been selected. Your task is to reconstruct the logical thinking process that would lead to choosing **this exact action**, based solely on the information provided.

Context:
{history}

Next Step Action:
{action}

Your output should follow the structure below:

<think>
Goal: [Clearly state the main objective based on the task or situation]
Current State: [Describe the current situation, observations, or context]
Available Actions: [List the reasonable actions or options available at this point]
Response: [Summarize the reasoning process that logically leads to the chosen action]
</think>

Important Guidelines:
- You are **not** deciding what to do next — you are **explaining the thought process** that justifies the already chosen action.
- You **must not** mention or imply that you already know what the next step is.
- The reasoning should appear as if it led naturally to the selected action without revealing that it was pre-determined.
""".strip()


THINK_MODE_3 = """
You are given a specific next step action that has already been selected. Your task is to reconstruct the logical thinking process that would lead to choosing **this exact action**, based solely on the information provided.

Context:
{history}

Next Step Action:
{action}

Your output should follow the structure below:

<think>
Goal: [Clearly state the main objective based on the task or situation]
Current State: [Describe the current situation, observations, or context]
Available Actions: [List the reasonable actions or options available at this point]
Reflection: [Reflect on the history—what actions were taken, and what was learned from them?]
Response: [Summarize the reasoning process that logically leads to the chosen action]
</think>

Important Guidelines:
- You are **not** deciding what to do next — you are **explaining the thought process** that justifies the already chosen action.
- You **must not** mention or imply that you already know what the next step is.
- The reasoning should appear as if it led naturally to the selected action without revealing that it was pre-determined.
""".strip()

THINK_MODE_4 = """
You are given a specific next step action that has already been selected. Your task is to reconstruct the logical thinking process that would lead to choosing **this exact action**, based solely on the information provided.

Context:
{history}

Next Step Action:
{action}

Your output should follow the structure below:

<think>
Goal: [Clearly state the main objective based on the task or situation]
Current State: [Describe the current situation, observations, or context]
Available Actions: [List the reasonable actions or options available at this point]
Reflection: [Reflect on the history—what actions were taken, and what was learned from them?]
Evaluation: [Critically evaluate why the chosen action is the most effective or appropriate choice]
Response: [Summarize the reasoning process that logically leads to the chosen action]
</think>

Important Guidelines:
- You are **not** deciding what to do next — you are **explaining the thought process** that justifies the already chosen action.
- You **must not** mention or imply that you already know what the next step is.
- The reasoning should appear as if it led naturally to the selected action without revealing that it was pre-determined.
""".strip()

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
    
    # 添加之前所有步骤的action和observation
    for i in range(step_idx):
        step = trajectory_data[i]
        # 只添加action部分，不包括thinking
        _, action, _ = extract_thinking_and_action(step['next_step'])
        if action:
            context_parts.append(f"Action: {action}")
        context_parts.append(f"Observation: {trajectory_data[i]['next_obs'].split('AVAILABLE ACTIONS:')[0].strip()}")
    return "\n\n".join(context_parts)


def batch_generate_thinking(histories: List[str], actions: List[str], levels: List[int], model, tokenizer) -> List[str]:
    """批量生成thinking，提高效率"""
    if not histories:
        return []
    
    # 构建批量prompts
    prompts = []
    for history, action, level in zip(histories, actions, levels):
        if level == 2:
            prompt = THINK_MODE_2.format(history=history, action=action)
        elif level == 3:
            prompt = THINK_MODE_3.format(history=history, action=action)
        elif level == 4:
            prompt = THINK_MODE_4.format(history=history, action=action)
        else:
            prompt = ""
        prompts.append(prompt)
    
    # 批量tokenize
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 批量生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=False,  # deterministic生成，不需要temperature等参数
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码结果
    results = []
    for i, output in enumerate(outputs):
        input_len = inputs['input_ids'][i].shape[0]
        generated_text = tokenizer.decode(output[input_len:], skip_special_tokens=True)
        
        # 提取thinking部分
        thinking_match = re.search(r'<think>(.*?)</think>', generated_text, re.DOTALL)
        if thinking_match:
            thinking = thinking_match.group(1).strip()
        else:
            thinking = generated_text.strip()
        results.append(thinking)
    
    return results


def compute_confidence(history: str, thinking: str, action: str, level: int, model, tokenizer) -> float:
    """计算action confidence (geometric mean of token probabilities) - 原始逻辑"""
    
    # 构建完整的input
    if level == 1:
        response = f"<level>1</level><action>{action}</action>"
    else:
        response = f"<level>{level}</level><think>{thinking}</think><action>{action}</action>"
    
    full_input = history + response
    prefix = full_input.split(f"<action>")[0] + "<action>"
    
    # Tokenize
    inputs = tokenizer(prefix + action, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    action_tokens = tokenizer(action, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
    
    # 计算logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
    
    # 计算action tokens的probabilities
    prefix_len = len(tokenizer(prefix, return_tensors="pt", truncation=True, max_length=2048)['input_ids'][0])
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


def batch_compute_confidence(histories: List[str], thinkings: List[str], actions: List[str], levels: List[int], model, tokenizer) -> List[float]:
    """批量计算confidence，保持原始几何平均逻辑 - 修复版本"""
    if not histories:
        return []
    
    batch_inputs = []
    batch_prefixes = []
    batch_action_tokens = []
    
    # 准备批量输入
    for i, (history, thinking, action, level) in enumerate(zip(histories, thinkings, actions, levels)):
        # 构建完整的input
        if level == 1:
            response = f"<level>1</level><action>{action}</action>"
        else:
            response = f"<level>{level}</level><think>{thinking}</think><action>{action}</action>"
        
        response = response.split(f"<action>")[0] + "<action>"
        prefix = history + response
        
        batch_inputs.append(prefix + action)
        batch_prefixes.append(prefix)
        
        # tokenize action
        action_tokens = tokenizer(action, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
        batch_action_tokens.append(action_tokens)
    
    # 批量tokenize
    inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 批量推理
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
    
    # 为每个prefix单独计算prefix长度
    prefix_lengths = []
    for prefix in batch_prefixes:
        prefix_tokens = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=2048)['input_ids'][0]
        prefix_lengths.append(len(prefix_tokens))
    
    # 计算每个样本的confidence
    results = []
    for i, (action_tokens, prefix_len) in enumerate(zip(batch_action_tokens, prefix_lengths)):
        sample_logits = logits[i]  # [seq_len, vocab_size]
        log_probs = []
        for j, token_id in enumerate(action_tokens):
            pos = prefix_len + j - 1
            if pos >= 0 and pos < sample_logits.shape[0]:
                token_logits = sample_logits[pos]
                probs = torch.softmax(token_logits, dim=-1)
                prob = probs[token_id].item()
                log_probs.append(torch.log(torch.tensor(prob + 1e-10)).item())
        
        # 几何平均 (geometric mean) - 保持原始逻辑
        if log_probs:
            confidence = torch.exp(torch.tensor(np.min(log_probs))).item()
            results.append(min(max(confidence, 0.0), 1.0))
        else:
            results.append(0.0)
    
    # 🔧 修复: return语句移到for循环外部
    return results


def process_single_step(step_data: Dict, model, tokenizer, trajectory, step_idx) -> Dict:
    """处理单个step，返回包含4个thinking level结果的字典 - 优化版本"""
   
    original_thinking, action, original_level = extract_thinking_and_action(step_data['next_step'])
    
    # 构建简化的history context
    history = build_history_context(trajectory, step_idx)
    
    # 准备批量生成需要的thinking
    need_generation = []
    for level in [1, 2, 3, 4]:
        if level != original_level and level != 1:  # level 1 不需要thinking
            need_generation.append((history, action, level))
    
    # 批量生成thinking
    if need_generation:
        histories, actions, levels = zip(*need_generation)
        generated_thinkings = batch_generate_thinking(list(histories), list(actions), list(levels), model, tokenizer)
    else:
        generated_thinkings = []
    
    # 组织thinking结果
    thinking_map = {}
    thinking_map[1] = ""  # level 1 没有thinking
    thinking_map[original_level] = original_thinking
    
    gen_idx = 0
    for level in [2, 3, 4]:
        if level != original_level:
            thinking_map[level] = generated_thinkings[gen_idx] if gen_idx < len(generated_thinkings) else ""
            gen_idx += 1
    
    # 构建instruction（复用）
    task_description = step_data.get('task_description', '')
    instruction = ALFWORLD_TEMPLATE_ADA.format(
        task_description=task_description,
        action_history=history
    )
    
    # 批量计算所有level的confidence
    histories = [instruction] * 4
    thinkings = [thinking_map[level] for level in [1, 2, 3, 4]]
    actions = [action] * 4
    levels = [1, 2, 3, 4]
    
    confidences = batch_compute_confidence(histories, thinkings, actions, levels, model, tokenizer)
    
    # 计算各level的指标
    results = {}
    for i, level in enumerate([1, 2, 3, 4]):
        thinking = thinking_map[level]
        thinking_cost = len(tokenizer.encode(thinking)) if thinking else 0
        confidence = confidences[i]
        
        # RLVCR reward计算
        R_entropy = confidence
        normalized_cost = min(1.0, thinking_cost / 250)  # cost_max = 250
        R_length = 0.5 - normalized_cost
        reward = R_entropy + 1.0 * R_length  # thinking_cost_alpha = 1.0
        
        results[f'level_{level}'] = {
            'thinking': thinking,
            'thinking_cost': thinking_cost,
            'confidence': confidence,
            'R_entropy': R_entropy,
            'R_length': R_length,
            'reward': reward
        }
    
    return {
        'original_step': step_data,
        'original_level': original_level,
        'original_action': action,
        'thinking_levels': results,
        'success': True
    }
    

def load_trajectory(file_path: str) -> List[Dict]:
    """加载trajectory数据"""
    trajectory = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    trajectory.append(json.loads(line.strip()))
        return trajectory
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return []

def batch_process_jsonl_files(input_dir: str, output_file: str, model_path: str):
    """批量处理目录下的所有jsonl文件"""
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    
    # 找到所有jsonl文件
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    print(f"Found {len(jsonl_files)} jsonl files")
    
    all_results = []
    
    for file_idx, jsonl_file in enumerate(jsonl_files):
        print(f"Processing file {file_idx+1}/{len(jsonl_files)}: {os.path.basename(jsonl_file)}")
        
        # 加载trajectory数据
        trajectory = load_trajectory(jsonl_file)
        if not trajectory:
            print(f"  Skipping empty file: {jsonl_file}")
            continue

        if trajectory[-1]["score"] < 100:
            continue
        
        file_results = {
            'file_name': os.path.basename(jsonl_file),
            'file_path': jsonl_file,
            'task_name': trajectory[0].get('task_name', ''),
            'steps': [],
            'trajectory': trajectory,
        }
        
        # 处理每个step
        for step_idx, step_data in enumerate(trajectory):
            print(f"  Processing step {step_idx+1}/{len(trajectory)}...")
            
            # 处理单个step
            step_result = process_single_step(step_data, model, tokenizer, trajectory, step_idx)
            step_result['step_index'] = step_idx
            
            file_results['steps'].append(step_result)
        
        all_results.append(file_results)
        print(f"  Completed file with {len(file_results['steps'])} steps")
        
        # 每处理10个文件保存一次进度（可选）
        if (file_idx + 1) % 5 == 0:
            backup_file = output_file.replace('.json', f'_backup_{file_idx+1}.json')
            print(f"  Saving backup to {backup_file}...")
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 保存所有结果到json文件（最终保存）
    print(f"Saving final results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
        
    print(f"Batch processing completed! Results saved to {output_file}")
        
    print(f"Batch processing completed! Results saved to {output_file}")
    
    # 打印统计信息
    total_files = len(all_results)
    total_steps = sum(len(file_result['steps']) for file_result in all_results)
    successful_steps = sum(len([s for s in file_result['steps'] if s.get('success', False)]) 
                          for file_result in all_results)
    
    print(f"\nStatistics:")
    print(f"Total files processed: {total_files}")
    print(f"Total steps processed: {total_steps}")
    print(f"Successful steps: {successful_steps}")
    print(f"Success rate: {successful_steps/total_steps*100:.1f}%" if total_steps > 0 else "N/A")

# 主程序
if __name__ == "__main__":
    # 模型路径
    model_path = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_cog-sft_sci_lr2e6_bs16_epoch5_full_0810"
    
    # 输入目录 - 包含所有jsonl文件的目录（你需要修改这个路径）
    input_directory = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/results/test/sciworld/qwen2.5-7b_cog-sft_mode5/1755172523"
    
    # 输出文件
    output_file = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/results/test/sciworld/qwen2.5-7b_cog-sft_mode5/1755172523/all_thinking_levels_results.json"
    
    # 开始批量处理
    batch_process_jsonl_files(input_directory, output_file, model_path)
