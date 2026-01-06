#!/usr/bin/env python3
"""
简化版：快速测试不同thinking对action的影响
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def quick_test_thinking_impact(model_path: str):
    """快速测试thinking对action的影响"""
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 你的sample数据
    observation = "You open the cabinet 3. The cabinet 3 is open. In it, you see nothing."
    
    # 定义不同level的thinking
    thinking_variants = {
        "level_1": "I need to find a clean cloth and put it in cabinet. Cabinet 3 is empty, so I can look elsewhere.",
        
        "level_2": "Goal: Put a clean cloth in cabinet. Current state: Cabinet 3 is open and empty. I should continue searching for a clean cloth in other locations like handtowel holders.",
        
        "level_3": "Goal: The goal is to put a clean cloth in a cabinet.\nCurrent state: I am currently at cabinet 3, which is now open and empty.\nAvailable actions: I can go to other locations to find a clean cloth, such as handtowel holders or other cabinets.\nReflection: Cabinet 3 is empty, so I need to find a clean cloth first before I can place it anywhere.\nResponse: I should go to a handtowel holder to find a clean cloth.",
        
        "original": "Goal: The goal is to put a clean cloth in a cabinet.\nCurrent state: I am currently at cabinet 3, which is closed.\nAvailable actions: The reasonable actions are to open cabinet 3, examine cabinet 3, or move to another location.\nReflection: From previous actions, I have already checked cabinets 1 and 2. Cabinet 1 contains soap bottles, and cabinet 2 is empty. I need to continue searching for a suitable cabinet to place the clean cloth.\nResponse: Since cabinet 3 is the next cabinet to check and it is currently closed, the logical next step is to open cabinet 3 to see if it can be used to store the clean cloth."
    }
    
    print("\n=== 测试不同thinking对action预测的影响 ===\n")
    
    results = {}
    
    for thinking_type, thinking_content in thinking_variants.items():
        print(f"--- {thinking_type.upper()} ---")
        print(f"Thinking: {thinking_content[:80]}...")
        
        # 构建prompt
        prompt = f"""You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.

Your task is to: put a clean cloth in cabinet.

{observation}

<think>
{thinking_content}
</think>
<action>"""

        # 生成action
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # 提取action
        if "</action>" in generated:
            action = generated.split("</action>")[0].strip()
        else:
            action = generated.strip()
        
        # 计算简单的confidence (log prob of first token)
        action_tokens = tokenizer(action, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
        if len(action_tokens) > 0:
            with torch.no_grad():
                logits = model(**inputs).logits
                next_token_logits = logits[0, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1)
                confidence = torch.log(probs[action_tokens[0]]).item()
        else:
            confidence = -10.0
        
        results[thinking_type] = {
            "action": action,
            "confidence": confidence
        }
        
        print(f"预测Action: {action}")
        print(f"置信度: {confidence:.4f}")
        print()
    
    # 比较结果
    print("=== 结果比较 ===")
    original_action = results["original"]["action"]
    
    for thinking_type, result in results.items():
        if thinking_type != "original":
            action_same = result["action"] == original_action
            conf_diff = result["confidence"] - results["original"]["confidence"]
            print(f"{thinking_type}: Action{'相同' if action_same else '不同'}, 置信度差异: {conf_diff:+.4f}")
    
    # 保存结果
    with open("quick_thinking_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n结果已保存到 quick_thinking_test_results.json")

if __name__ == "__main__":
    model_path = "/apdcephfs_cq11_1567347/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_cog-sft_alf_lr2e6_bs16_epoch5_full_0810"
    quick_test_thinking_impact(model_path)
