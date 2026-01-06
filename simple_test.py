#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型
model_path = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_cog-sft_alf_lr2e6_bs16_epoch5_full_0810"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map="auto")
model.eval()

# 你的实际例子
prefix = """Your task is to: put two peppershaker in shelf.
You are in the middle of a room. Looking quickly around you, you see a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, and a stoveburner 1.

Your task is to: put two peppershaker in shelf.
> go to countertop 1
You arrive at loc 1. On the countertop 1, you see a butterknife 1, a cup 1, a fork 1, a mug 1, a pan 1, a peppershaker 2, a saltshaker 3, a soapbottle 2, a spatula 2, a spatula 1, and a spoon 1. AVAILABLE ACTIONS: 'examine countertop 1'
 'examine toaster 1'
 'go to cabinet 1'
 'go to cabinet 10'
 'go to cabinet 11'
 'go to cabinet 12'
 'go to cabinet 13'
 'go to cabinet 2'
 'go to cabinet 3'
 'go to cabinet 4'
 'go to cabinet 5'
 'go to cabinet 6'
 'go to cabinet 7'
 'go to cabinet 8'
 'go to cabinet 9'
 'go to coffeemachine 1'
 'go to diningtable 1'
 'go to drawer 1'
 'go to fridge 1'
 'go to garbagecan 1'
 'go to microwave 1'
 'go to shelf 1'
 'go to shelf 2'
 'go to shelf 3'
 'go to sinkbasin 1'
 'go to stoveburner 1'
 'go to stoveburner 2'
 'go to stoveburner 3'
 'go to stoveburner 4'
 'inventory'
 'look'
 'take butterknife 1 from countertop 1'
 'take cup 1 from countertop 1'
 'take fork 1 from countertop 1'
 'take mug 1 from countertop 1'
 'take pan 1 from countertop 1'
 'take peppershaker 2 from countertop 1'
 'take saltshaker 3 from countertop 1'
 'take soapbottle 2 from countertop 1'
 'take spatula 1 from countertop 1'
 'take spatula 2 from countertop 1'
 'take spoon 1 from countertop 1'

Now it's your turn to generate next step response.<level>3</level><think>Goal: The goal is to put two peppershaker in a shelf.
Current state: I am currently at countertop 1, where I see peppershaker 2 along with other items.
Available actions: I can take peppershaker 2 from countertop 1, or choose from various available actions such as moving to different locations or examining objects.
Reflection: Previously, I took peppershaker 1 from the diningtable 1 and placed it on shelf 1. My task is to add a second peppershaker to the shelf, and the current environment has peppershaker 2 located on the countertop 1. Given that the goal is to place both peppershakers in a shelf, taking peppershaker 2 is a required step before proceeding to the shelf.
Response: Taking peppershaker 2 from countertop 1 aligns with the task objective of putting two peppershaker in a shelf.</think><action>"""

action_text = "take peppershaker 2 from countertop 1"

print("测试prefix:")
print(f"长度: {len(prefix)} 字符")
print(f"Action: '{action_text}'")
print()

# ==================== 方法1：逐token计算 ====================
print("方法1：逐token计算（当前RLVCR方法）")
print("-" * 50)

# Tokenize action
action_token_ids = tokenizer.encode(action_text, add_special_tokens=False)
print(f"Action tokens: {action_token_ids}")
print(f"Token texts: {[tokenizer.decode([t]) for t in action_token_ids]}")

# 逐token计算概率
method1_probs = []
current_input = tokenizer.encode(prefix, add_special_tokens=False, return_tensors='pt').to(model.device)

for i, token_id in enumerate(action_token_ids):
    # 拼接当前token
    extended_input = torch.cat([current_input[0], torch.tensor([token_id]).to(model.device)]).unsqueeze(0)
    
    # 计算这个token的概率
    with torch.no_grad():
        outputs = model(extended_input)
        logits = outputs.logits[0, -2, :]  # 倒数第二个位置预测最后一个token
        
        # 调试信息：检查logits
        token_text = tokenizer.decode([token_id])
        target_logit = logits[token_id].item()
        max_logit = logits.max().item()
        logit_range = (logits.max() - logits.min()).item()
        
        # 模拟RLVCR的confidence计算：使用temperature=1.0获得真实概率分布
        temperature = 1.0  # confidence计算应该用真实概率，不是生成时的锐化分布
        logits = logits / temperature  # 直接修改logits，就像RLVCR中的logits.div_(temperature)
        probs = torch.softmax(logits, dim=-1)
        token_prob = probs[token_id].item()
        
        print(f"Step {i+1}: '{token_text}' -> prob={token_prob:.6f}, logit={target_logit:.2f}, max_logit={max_logit:.2f}, range={logit_range:.2f}")
        
        # 如果概率异常，显示top tokens
        if token_prob < 1e-5 or token_prob > 0.999:
            print(f"   ⚠️ 异常概率! Top 3 tokens:")
            top_probs, top_indices = torch.topk(probs, k=3)
            for j, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                top_token_text = tokenizer.decode([idx])
                marker = " ← TARGET" if idx == token_id else ""
                print(f"     {j+1}. '{top_token_text}': {prob:.6f}{marker}")
    
    method1_probs.append(token_prob)
    
    # 更新input
    current_input = extended_input

method1_min = min(method1_probs)
print(f"所有概率: {[f'{p:.6f}' for p in method1_probs]}")
print(f"最小概率: {method1_min:.6f}")
print()

# ==================== 方法2：直接自回归 ====================
print("方法2：直接自回归计算")
print("-" * 50)

# 完整文本
full_text = prefix + action_text
full_tokens = tokenizer.encode(full_text, add_special_tokens=False, return_tensors='pt').to(model.device)
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False, return_tensors='pt').to(model.device)

prefix_len = len(prefix_tokens[0])
action_start_idx = prefix_len
action_end_idx = len(full_tokens[0])

print(f"完整序列长度: {len(full_tokens[0])}")
print(f"Prefix长度: {prefix_len}")
print(f"Action位置: {action_start_idx}:{action_end_idx}")

# 提取action tokens
actual_action_tokens = full_tokens[0][action_start_idx:action_end_idx].tolist()
print(f"实际action tokens: {actual_action_tokens}")
print(f"预期action tokens: {action_token_ids}")
print(f"Token序列匹配: {actual_action_tokens == action_token_ids}")

if actual_action_tokens != action_token_ids:
    print("⚠️ TOKEN不匹配! 这就是问题所在!")
    print(f"实际: {[tokenizer.decode([t]) for t in actual_action_tokens]}")
    print(f"预期: {[tokenizer.decode([t]) for t in action_token_ids]}")

# 计算直接方法的概率
method2_probs = []
with torch.no_grad():
    outputs = model(full_tokens)
    logits = outputs.logits[0]  # [seq_len, vocab_size]
    
    for i, token_id in enumerate(actual_action_tokens):
        pos = action_start_idx + i - 1  # 预测当前token的位置
        if pos >= 0 and pos < logits.shape[0]:
            token_logits = logits[pos]
            # 模拟RLVCR的confidence计算：使用temperature=1.0获得真实概率分布
            temperature = 1.0  # confidence计算应该用真实概率，不是生成时的锐化分布
            token_logits = token_logits / temperature  # 直接修改logits
            probs = torch.softmax(token_logits, dim=-1)
            token_prob = probs[token_id].item()
            method2_probs.append(token_prob)
            
            token_text = tokenizer.decode([token_id])
            print(f"Position {pos}: '{token_text}' -> {token_prob:.6f}")

method2_min = min(method2_probs) if method2_probs else 0.0
print(f"所有概率: {[f'{p:.6f}' for p in method2_probs]}")
print(f"最小概率: {method2_min:.6f}")
print()

# ==================== 对比结果 ====================
print("对比结果")
print("-" * 50)
print(f"方法1最小概率: {method1_min:.6f}")
print(f"方法2最小概率: {method2_min:.6f}")

if len(method1_probs) == len(method2_probs) and actual_action_tokens == action_token_ids:
    differences = [abs(p1 - p2) for p1, p2 in zip(method1_probs, method2_probs)]
    max_diff = max(differences)
    print(f"最大差异: {max_diff:.6f}")
    print(f"平均差异: {sum(differences)/len(differences):.6f}")
    
    if max_diff < 1e-5:
        print("✅ 两种方法数值等价")
    elif max_diff < 1e-3:
        print("⚠️ 有小差异但基本等价")
    else:
        print("❌ 差异很大")
else:
    print("❌ 无法对比：token序列不同或长度不同")

print(f"\n总结: 如果token序列不匹配，那就是tokenization问题导致的'奇怪结果'")
