#!/usr/bin/env python3
"""
测试不同thinking level对action预测的影响
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any
import argparse

class ThinkingImpactTester:
    def __init__(self, model_path: str):
        """初始化模型和tokenizer"""
        print(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def extract_thinking_and_action(self, response: str) -> tuple:
        """从response中提取thinking和action"""
        thinking = ""
        action = ""
        
        # 提取thinking部分
        if "<think>" in response and "</think>" in response:
            start = response.find("<think>") + len("<think>")
            end = response.find("</think>")
            thinking = response[start:end].strip()
        
        # 提取action部分
        if "<action>" in response and "</action>" in response:
            start = response.find("<action>") + len("<action>")
            end = response.find("</action>")
            action = response[start:end].strip()
        
        return thinking, action
    
    def generate_alternative_thinking(self, original_thinking: str, level: int = 2) -> str:
        """生成不同level的thinking"""
        
        # 构建prompt来生成alternative thinking
        prompt = f"""Given the original thinking below, generate a {level}-level alternative thinking that leads to the same conclusion but with different reasoning depth:

Original thinking:
{original_thinking}

Generate a level-{level} alternative thinking (where level 1 is simple, level 3 is detailed):"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return generated.strip()
    
    def predict_action_with_thinking(self, observation: str, thinking: str) -> Dict[str, Any]:
        """使用特定thinking预测action"""
        
        # 构建完整的prompt
        prompt = f"""You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.

Your task is to: put a clean cloth in cabinet.

{observation}

<think>
{thinking}
</think>
<action>"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,  # 低温度，更deterministic
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # 提取action
        if "</action>" in generated:
            action = generated.split("</action>")[0].strip()
        else:
            action = generated.strip()
        
        # 计算logits/confidence
        full_input = prompt + action
        full_inputs = self.tokenizer(full_input, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            logits = self.model(**full_inputs).logits
            action_tokens = self.tokenizer(action, return_tensors="pt")['input_ids'][0]
            
            # 计算action tokens的平均log probability
            action_start_idx = inputs['input_ids'].shape[1] - 1
            action_logits = logits[0, action_start_idx:action_start_idx+len(action_tokens)]
            action_probs = torch.softmax(action_logits, dim=-1)
            
            confidence = torch.mean(torch.log(torch.max(action_probs, dim=-1)[0])).item()
        
        return {
            "action": action,
            "confidence": confidence,
            "thinking": thinking,
            "full_prompt": prompt,
            "generated_text": generated
        }
    
    def test_thinking_impact(self, sample_data: Dict, num_alternatives: int = 3) -> Dict[str, Any]:
        """测试不同thinking对action预测的影响"""
        
        observation = sample_data["observation"]
        original_response = sample_data["generated_response"]
        
        # 提取原始thinking和action
        original_thinking, original_action = self.extract_thinking_and_action(original_response)
        
        print(f"原始Action: {original_action}")
        print(f"原始Thinking: {original_thinking[:100]}...")
        print("="*50)
        
        results = {
            "original": {
                "thinking": original_thinking,
                "action": original_action,
                "confidence": None
            },
            "alternatives": []
        }
        
        # 测试原始thinking的预测
        original_prediction = self.predict_action_with_thinking(observation, original_thinking)
        results["original"]["predicted_action"] = original_prediction["action"]
        results["original"]["confidence"] = original_prediction["confidence"]
        
        print(f"原始thinking预测的action: {original_prediction['action']}")
        print(f"原始thinking置信度: {original_prediction['confidence']:.4f}")
        print("="*50)
        
        # 生成和测试alternative thinking
        for i in range(num_alternatives):
            level = (i % 3) + 1  # level 1, 2, 3
            
            print(f"生成Level {level} alternative thinking...")
            alt_thinking = self.generate_alternative_thinking(original_thinking, level)
            
            print(f"Alternative thinking {i+1}: {alt_thinking[:100]}...")
            
            # 用alternative thinking预测action
            alt_prediction = self.predict_action_with_thinking(observation, alt_thinking)
            
            result = {
                "level": level,
                "thinking": alt_thinking,
                "predicted_action": alt_prediction["action"],
                "confidence": alt_prediction["confidence"],
                "action_changed": alt_prediction["action"] != original_action
            }
            
            results["alternatives"].append(result)
            
            print(f"Alternative预测的action: {alt_prediction['action']}")
            print(f"Alternative置信度: {alt_prediction['confidence']:.4f}")
            print(f"Action是否改变: {result['action_changed']}")
            print("="*30)
        
        return results

def main():
    parser = argparse.ArgumentParser(description="测试thinking对action预测的影响")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--sample_file", type=str, help="包含sample数据的JSON文件")
    parser.add_argument("--output_file", type=str, default="thinking_impact_results.json", help="输出结果文件")
    
    args = parser.parse_args()
    
    # 初始化tester
    tester = ThinkingImpactTester(args.model_path)
    
    # 读取sample数据
    if args.sample_file:
        with open(args.sample_file, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
    else:
        # 使用你提供的示例数据
        sample_data = {
            "step_index": 8,
            "active_mask": True,
            "reward": 0.0,
            "observation": "You open the cabinet 3. The cabinet 3 is open. In it, you see nothing.",
            "won": False,
            "action": "<level>3</level>\n<think>\nGoal: The goal is to put a clean cloth in a cabinet.\nCurrent state: I am currently at cabinet 3, which is closed.\nAvailable actions: The reasonable actions are to open cabinet 3, examine cabinet 3, or move to another location.\nReflection: From previous actions, I have already checked cabinets 1 and 2. Cabinet 1 contains soap bottles, and cabinet 2 is empty. I need to continue searching for a suitable cabinet to place the clean cloth.\nResponse: Since cabinet 3 is the next cabinet to check and it is currently closed, the logical next step is to open cabinet 3 to see if it can be used to store the clean cloth.\n</think>\n<action>open cabinet 3</action>",
            "action_valid": False,
            "action_available": True,
            "generated_response": "<level>3</level>\n<think>\nGoal: The goal is to put a clean cloth in a cabinet.\nCurrent state: I am currently at cabinet 3, which is closed.\nAvailable actions: The reasonable actions are to open cabinet 3, examine cabinet 3, or move to another location.\nReflection: From previous actions, I have already checked cabinets 1 and 2. Cabinet 1 contains soap bottles, and cabinet 2 is empty. I need to continue searching for a suitable cabinet to place the clean cloth.\nResponse: Since cabinet 3 is the next cabinet to check and it is currently closed, the logical next step is to open cabinet 3 to see if it can be used to store the clean cloth.\n</think>\n<action>open cabinet 3</action>"
        }
    
    # 运行测试
    print("开始测试thinking对action预测的影响...")
    results = tester.test_thinking_impact(sample_data, num_alternatives=5)
    
    # 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n测试完成！结果保存在 {args.output_file}")
    
    # 输出summary
    print("\n=== 测试总结 ===")
    print(f"原始action: {results['original']['action']}")
    print(f"原始confidence: {results['original']['confidence']:.4f}")
    
    action_changes = sum(1 for alt in results['alternatives'] if alt['action_changed'])
    print(f"Action改变次数: {action_changes}/{len(results['alternatives'])}")
    
    confidence_changes = []
    for alt in results['alternatives']:
        conf_change = alt['confidence'] - results['original']['confidence']
        confidence_changes.append(conf_change)
        print(f"Level {alt['level']} - 置信度变化: {conf_change:+.4f}")

if __name__ == "__main__":
    main()
