#!/usr/bin/env python3
"""
测试逐token计算和直接自回归计算的等价性
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_confidence_calculation_equivalence():
    """
    测试两种confidence计算方法的等价性
    """
    # 你的具体例子
    prefix = """Your task is to find a(n) animal. First, focus on the thing. Then, move it to the purple box in the kitchen.
This room is called the greenhouse. In it, you see:   the agent  a substance called air  a bee hive. The bee hive door is closed.   a flower pot 1 (containing a pea plant in the adult stage with a tall height, soil)  a flower pot 4 (containing a banana tree in the adult stage, soil)  a flower pot 6 (containing nothing)  a flower pot 8 (containing a avocado tree in the adult stage, soil)  a flower pot 9 (containing nothing)  a jug (containing nothing)  a sink, which is turned off. In the sink is: nothing. You also see:  A door to the hallway (that is closed)  A door to the outside (that is closed)<level>3</level>
<think>
Goal: The task is to find an animal and move it to the purple box in the kitchen.
Current state: I am in the greenhouse, which contains various plants, a closed bee hive, and closed doors to the hallway and outside.
Available actions: I can open the door to the hallway, open the door to the outside, or interact with the bee hive or plants.
Reflection: The task requires finding an animal, and the current room, the greenhouse, does not contain any visible animals. The bee hive is a potential source of animals, but it is closed. The history suggests that the next logical step is to explore other areas to find an animal.
Response: Opening the door to outside may lead to finding an animal, as the greenhouse does not contain any visible animals.
</think>
<action>"""

    action_text = "open door to outside"
    
    print("=" * 80)
    print("🧪 Testing Autoregressive Equivalence")
    print("=" * 80)
    print()
    print(f"Prefix length: {len(prefix)} characters")
    print(f"Action: '{action_text}'")
    print()
    
    # 使用一个简单的模型进行测试（你可以替换为实际使用的模型）
    model_name = "gpt2"  # 可以改为你实际使用的模型
    print(f"Loading model: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        
        print("✅ Model loaded successfully")
        print()
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("Using mock calculations for demonstration...")
        return demonstrate_with_mock_data(prefix, action_text)
    
    # 方法1：逐token计算（当前RLVCR实现）
    print("🔍 Method 1: Step-by-step token calculation")
    print("-" * 50)
    
    # Tokenize action
    action_tokens = tokenizer.encode(action_text, add_special_tokens=False)
    print(f"Action tokens: {action_tokens}")
    print(f"Action tokens text: {[tokenizer.decode([t]) for t in action_tokens]}")
    print()
    
    # 逐token计算概率
    method1_probs = []
    current_input = tokenizer.encode(prefix, add_special_tokens=False, return_tensors='pt')
    
    print("Step-by-step calculation:")
    for i, token_id in enumerate(action_tokens):
        # 扩展输入
        extended_input = torch.cat([current_input[0], torch.tensor([token_id])]).unsqueeze(0)
        
        # 计算概率
        with torch.no_grad():
            outputs = model(extended_input)
            logits = outputs.logits[0, -2, :]  # 倒数第二个位置预测最后一个token
            probs = torch.softmax(logits, dim=-1)
            token_prob = probs[token_id].item()
        
        method1_probs.append(token_prob)
        
        token_text = tokenizer.decode([token_id])
        print(f"  Step {i+1}: '{token_text}' (id={token_id}) -> prob={token_prob:.6f}")
        
        # 更新当前输入
        current_input = extended_input
    
    method1_confidence = min(method1_probs)
    print(f"\nMethod 1 Results:")
    print(f"  All probabilities: {[f'{p:.6f}' for p in method1_probs]}")
    print(f"  Min probability (confidence): {method1_confidence:.6f}")
    print()
    
    # 方法2：直接自回归计算
    print("🎯 Method 2: Direct autoregressive calculation")
    print("-" * 50)
    
    # 构建完整文本
    full_text = prefix + action_text
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False, return_tensors='pt')
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False, return_tensors='pt')
    
    prefix_length = len(prefix_tokens[0])
    action_start_idx = prefix_length
    action_end_idx = len(full_tokens[0])
    
    print(f"Full sequence length: {len(full_tokens[0])}")
    print(f"Prefix length: {prefix_length}")
    print(f"Action token positions: {action_start_idx}:{action_end_idx}")
    
    # 提取action部分的tokens
    actual_action_tokens = full_tokens[0][action_start_idx:action_end_idx].tolist()
    print(f"Actual action tokens from full sequence: {actual_action_tokens}")
    print(f"Expected action tokens: {action_tokens}")
    
    # 检查token匹配
    tokens_match = actual_action_tokens == action_tokens
    print(f"Token sequences match: {tokens_match}")
    if not tokens_match:
        print("⚠️  WARNING: Token sequences don't match! This is a key difference.")
        print("This could explain discrepancies in confidence calculations.")
    print()
    
    # 计算直接方法的概率
    method2_probs = []
    with torch.no_grad():
        outputs = model(full_tokens)
        logits = outputs.logits[0]  # [seq_len, vocab_size]
        
        print("Direct calculation:")
        for i, token_id in enumerate(actual_action_tokens):
            pos = action_start_idx + i - 1  # 预测当前token的位置
            if pos >= 0 and pos < logits.shape[0]:
                token_logits = logits[pos]
                probs = torch.softmax(token_logits, dim=-1)
                token_prob = probs[token_id].item()
                method2_probs.append(token_prob)
                
                token_text = tokenizer.decode([token_id])
                print(f"  Position {pos}: '{token_text}' (id={token_id}) -> prob={token_prob:.6f}")
            else:
                print(f"  Position {pos}: Out of range")
    
    method2_confidence = min(method2_probs) if method2_probs else 0.0
    print(f"\nMethod 2 Results:")
    print(f"  All probabilities: {[f'{p:.6f}' for p in method2_probs]}")
    print(f"  Min probability (confidence): {method2_confidence:.6f}")
    print()
    
    # 对比结果
    print("📊 Comparison")
    print("-" * 50)
    print(f"Method 1 confidence: {method1_confidence:.6f}")
    print(f"Method 2 confidence: {method2_confidence:.6f}")
    if method1_probs and method2_probs and len(method1_probs) == len(method2_probs):
        differences = [abs(p1 - p2) for p1, p2 in zip(method1_probs, method2_probs)]
        max_diff = max(differences)
        avg_diff = sum(differences) / len(differences)
        print(f"Max difference: {max_diff:.6f}")
        print(f"Average difference: {avg_diff:.6f}")
        
        if max_diff < 1e-4:
            print("✅ Methods are numerically equivalent (difference < 1e-4)")
        elif max_diff < 1e-2:
            print("⚠️  Methods have small differences (difference < 1e-2)")
        else:
            print("❌ Methods have significant differences (difference >= 1e-2)")
    else:
        print("❌ Cannot compare: different number of probabilities")
    
    print()
    print("🎯 Conclusions:")
    if not tokens_match:
        print("- Token mismatch is likely the main source of differences")
        print("- Consider using direct autoregressive method for consistency")
    else:
        print("- Token sequences match, any differences are due to numerical precision")
        print("- Both methods should give similar results")

def demonstrate_with_mock_data(prefix, action_text):
    """
    用模拟数据演示差异（当无法加载实际模型时）
    """
    print("📝 Mock Demonstration")
    print("-" * 50)
    print()
    
    # 模拟tokenization
    mock_action_tokens = [123, 456, 789, 101]  # 模拟"open door to outside"的tokens
    
    print("Mock tokenization:")
    print(f"Action: '{action_text}'")
    print(f"Tokens: {mock_action_tokens}")
    print()
    
    # 模拟两种方法的概率
    print("Method 1 (step-by-step):")
    method1_probs = [0.8234, 0.6789, 0.5432, 0.7891]
    for i, (token, prob) in enumerate(zip(mock_action_tokens, method1_probs)):
        print(f"  Token {i}: {token} -> prob={prob:.4f}")
    print(f"  Confidence: {min(method1_probs):.4f}")
    print()
    
    print("Method 2 (direct autoregressive):")
    # 模拟可能的差异
    method2_probs = [0.8234, 0.6791, 0.5430, 0.7891]  # 略有不同
    for i, (token, prob) in enumerate(zip(mock_action_tokens, method2_probs)):
        print(f"  Token {i}: {token} -> prob={prob:.4f}")
    print(f"  Confidence: {min(method2_probs):.4f}")
    print()
    
    print("Differences:")
    differences = [abs(p1 - p2) for p1, p2 in zip(method1_probs, method2_probs)]
    for i, diff in enumerate(differences):
        print(f"  Token {i}: {diff:.6f}")
    print(f"  Max difference: {max(differences):.6f}")

if __name__ == "__main__":
    test_confidence_calculation_equivalence()
