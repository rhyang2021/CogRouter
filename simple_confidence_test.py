#!/usr/bin/env python3
"""
简化的confidence计算测试
测试逐token vs 直接自回归的差异
"""
import torch
from transformers import AutoTokenizer

def simple_tokenization_test():
    """
    简单的tokenization对比测试
    """
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
    
    print("🧪 Tokenization Comparison Test")
    print("=" * 60)
    print(f"Action: '{action_text}'")
    print()
    
    # 尝试几个常用的tokenizer
    tokenizer_names = [
        "gpt2",
        "microsoft/DialoGPT-medium", 
        "meta-llama/Llama-2-7b-hf",  # 如果有访问权限
    ]
    
    for model_name in tokenizer_names:
        
    print(f"Testing with {model_name}:")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 方法1：分别tokenize
    action_tokens = tokenizer.encode(action_text, add_special_tokens=False)
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    
    # 方法2：整体tokenize
    full_text = prefix + action_text
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    
    # 从完整序列中提取action部分
    expected_action_start = len(prefix_tokens)
    actual_action_tokens = full_tokens[expected_action_start:]
    
    print(f"  Method 1 (separate): {action_tokens}")
    print(f"  Method 2 (full): {actual_action_tokens}")
    print(f"  Match: {action_tokens == actual_action_tokens}")
    
    if action_tokens != actual_action_tokens:
        print(f"  ⚠️  TOKENIZATION MISMATCH!")
        print(f"     Separate tokens: {[tokenizer.decode([t]) for t in action_tokens]}")
        print(f"     Full tokens: {[tokenizer.decode([t]) for t in actual_action_tokens]}")
    else:
        print(f"  ✅ Tokenization consistent")

if __name__ == "__main__":
    simple_tokenization_test()