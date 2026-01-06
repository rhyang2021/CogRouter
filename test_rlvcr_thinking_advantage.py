#!/usr/bin/env python3
"""
测试RLVCR的thinking advantage计算过程
模拟真实的RLVCR算法中thinking level对advantage的影响
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from typing import List, Dict, Tuple
from collections import defaultdict

class RLVCRThinkingAdvantageTest:
    def __init__(self, model_path: str):
        """初始化模型和tokenizer"""
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def extract_action_from_response(self, response: str) -> str:
        """从response中提取action"""
        if "<action>" in response and "</action>" in response:
            start = response.find("<action>") + len("<action>")
            end = response.find("</action>")
            return response[start:end].strip()
        return ""
    
    def extract_thinking_from_response(self, response: str) -> str:
        """从response中提取thinking"""
        if "<think>" in response and "</think>" in response:
            start = response.find("<think>") + len("<think>")
            end = response.find("</think>")
            return response[start:end].strip()
        return ""
    
    def extract_level_from_response(self, response: str) -> int:
        """从response中提取level"""
        pattern = r'<level>(\d+)</level>'
        match = re.search(pattern, response)
        if match:
            level = int(match.group(1))
            return max(1, min(4, level))
        return 1
    
    def count_thinking_tokens(self, thinking_text: str) -> int:
        """计算thinking的token数量"""
        if not thinking_text:
            return 0
        return len(self.tokenizer.encode(thinking_text))
    
    def generate_level_specific_response(self, observation: str, level: int, action: str) -> str:
        """生成特定level的response"""
        
        # 根据level构建不同的prompt
        if level == 1:
            prompt = f"""You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.

Your task is to: put a clean cloth in cabinet.

{observation}

Respond with minimal thinking. Use format: <level>1</level><action>{action}</action>"""
            
        elif level == 2:
            prompt = f"""You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.

Your task is to: put a clean cloth in cabinet.

{observation}

Respond with medium-level thinking. Include goal and current state. Use format: <level>2</level><think>...</think><action>{action}</action>"""
            
        elif level == 3:
            prompt = f"""You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.

Your task is to: put a clean cloth in cabinet.

{observation}

Respond with detailed thinking. Include goal, current state, available actions, and reflection. Use format: <level>3</level><think>...</think><action>{action}</action>"""
            
        elif level == 4:
            prompt = f"""You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.

Your task is to: put a clean cloth in cabinet.

{observation}

Respond with very detailed thinking. Include comprehensive analysis, multiple reasoning steps, alternative considerations. Use format: <level>4</level><think>...</think><action>{action}</action>"""

        # 生成response
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return generated.strip()
    
    def compute_action_confidence(self, response: str, action: str) -> float:
        """计算action的置信度 (模拟RLVCR的confidence calculation)"""
        
        # 构建prompt到action的完整序列
        if "<action>" in response:
            prefix = response.split("<action>")[0] + "<action>"
        else:
            prefix = response
            
        full_sequence = prefix + action
        
        # Tokenize
        inputs = self.tokenizer(full_sequence, return_tensors="pt", truncation=True)
        action_tokens = self.tokenizer(action, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
        
        if len(action_tokens) == 0:
            return 0.0
        
        # 计算logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]  # [seq_len, vocab_size]
        
        # 计算action tokens的log probabilities
        prefix_len = len(self.tokenizer(prefix, return_tensors="pt")['input_ids'][0])
        action_log_probs = []
        
        for i, token_id in enumerate(action_tokens):
            pos = prefix_len + i - 1  # -1 because logits are shifted
            if pos >= 0 and pos < logits.shape[0]:
                token_logits = logits[pos]
                token_probs = torch.softmax(token_logits, dim=-1)
                token_log_prob = torch.log(token_probs[token_id] + 1e-10)
                action_log_probs.append(token_log_prob.item())
        
        # 返回最小log prob (RLVCR使用这个作为confidence)
        if action_log_probs:
            return min(action_log_probs)
        return 0.0
    
    def compute_rlvcr_thinking_advantages(
        self, 
        thinking_group: List[Dict], 
        thinking_weight: float = 0.5,
        thinking_cost_alpha: float = 1.0,
        cost_max: int = 250
    ) -> Dict[str, float]:
        """计算RLVCR thinking advantages (模拟真实算法)"""
        
        if len(thinking_group) != 4:
            print(f"Warning: incomplete thinking group (size={len(thinking_group)})")
            return {}
        
        # Step 1: 计算每个level的thinking reward
        group_thinking_rewards = []
        
        for sample in thinking_group:
            # R_entropy: Action confidence (已经是log probability，转换为[0,1])
            confidence = sample['confidence']
            R_entropy = max(0.0, min(1.0, (confidence + 10) / 10))  # 简单映射到[0,1]
            
            # R_length: Normalized cost penalty [0.5, -0.5]
            cost = sample['thinking_cost']
            normalized_cost = cost / cost_max
            normalized_cost = min(1.0, normalized_cost)  # Clip to [0, 1]
            R_length = 0.5 - normalized_cost
            
            # Composite thinking reward
            thinking_reward = R_entropy + thinking_cost_alpha * R_length
            group_thinking_rewards.append(thinking_reward)
            
            print(f"Level {sample['level']}: confidence={confidence:.4f}, cost={cost}, "
                  f"R_entropy={R_entropy:.4f}, R_length={R_length:.4f}, reward={thinking_reward:.4f}")
        
        # Step 2: 计算thinking advantages (zero-mean within group)
        mean_reward = np.mean(group_thinking_rewards)
        thinking_advantages = {}
        
        for i, sample in enumerate(thinking_group):
            level = sample['level']
            thinking_adv = group_thinking_rewards[i] - mean_reward
            thinking_advantages[f"level_{level}"] = thinking_adv
            
            print(f"Level {level} thinking advantage: {thinking_adv:.4f}")
        
        return thinking_advantages
    
    def test_rlvcr_thinking_impact(self, observation: str, target_action: str) -> Dict:
        """测试RLVCR thinking对advantage的影响"""
        
        print("=== RLVCR Thinking Advantage测试 ===\n")
        print(f"Observation: {observation}")
        print(f"Target Action: {target_action}")
        print("\n" + "="*50)
        
        # Step 1: 生成4个level的responses
        thinking_group = []
        
        for level in range(1, 5):
            print(f"\n--- 生成Level {level} Response ---")
            
            response = self.generate_level_specific_response(observation, level, target_action)
            
            # 提取信息
            actual_action = self.extract_action_from_response(response)
            thinking_text = self.extract_thinking_from_response(response)
            thinking_cost = self.count_thinking_tokens(thinking_text)
            
            # 计算confidence
            confidence = self.compute_action_confidence(response, actual_action)
            
            sample = {
                'level': level,
                'response': response,
                'action': actual_action,
                'thinking': thinking_text,
                'thinking_cost': thinking_cost,
                'confidence': confidence
            }
            
            thinking_group.append(sample)
            
            print(f"Action: {actual_action}")
            print(f"Thinking cost: {thinking_cost} tokens")
            print(f"Confidence: {confidence:.4f}")
            print(f"Thinking preview: {thinking_text[:80]}...")
        
        # Step 2: 计算RLVCR thinking advantages
        print(f"\n{'='*50}")
        print("=== RLVCR Thinking Advantage计算 ===")
        
        thinking_advantages = self.compute_rlvcr_thinking_advantages(thinking_group)
        
        # Step 3: 分析结果
        print(f"\n{'='*50}")
        print("=== 结果分析 ===")
        
        results = {
            'thinking_group': thinking_group,
            'thinking_advantages': thinking_advantages,
            'analysis': {}
        }
        
        # 找出最优和最差的thinking level
        if thinking_advantages:
            best_level = max(thinking_advantages.keys(), key=lambda k: thinking_advantages[k])
            worst_level = min(thinking_advantages.keys(), key=lambda k: thinking_advantages[k])
            
            print(f"最优thinking level: {best_level} (advantage: {thinking_advantages[best_level]:.4f})")
            print(f"最差thinking level: {worst_level} (advantage: {thinking_advantages[worst_level]:.4f})")
            
            results['analysis']['best_level'] = best_level
            results['analysis']['worst_level'] = worst_level
            results['analysis']['advantage_range'] = thinking_advantages[best_level] - thinking_advantages[worst_level]
        
        # 分析action一致性
        actions = [sample['action'] for sample in thinking_group]
        unique_actions = list(set(actions))
        print(f"Action一致性: {len(unique_actions)}/{len(actions)} unique actions")
        
        results['analysis']['action_consistency'] = len(unique_actions) == 1
        results['analysis']['unique_actions'] = unique_actions
        
        return results

def main():
    model_path = "/apdcephfs_cq11_1567347/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_cog-sft_alf_lr2e6_bs16_epoch5_full_0810"
    
    # 你提供的sample数据
    observation = "You open the cabinet 3. The cabinet 3 is open. In it, you see nothing."
    target_action = "go to handtowelholder 1"  # 一个合理的下一步行动
    
    # 运行测试
    tester = RLVCRThinkingAdvantageTest(model_path)
    results = tester.test_rlvcr_thinking_impact(observation, target_action)
    
    # 保存结果
    output_file = "rlvcr_thinking_advantage_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n测试完成！结果保存在 {output_file}")
    
    # 输出关键指标
    print("\n=== 关键指标总结 ===")
    if 'thinking_advantages' in results:
        for level_key, advantage in results['thinking_advantages'].items():
            print(f"{level_key}: {advantage:+.4f}")
    
    if 'analysis' in results:
        analysis = results['analysis']
        print(f"最优level: {analysis.get('best_level', 'N/A')}")
        print(f"Advantage范围: {analysis.get('advantage_range', 0):.4f}")
        print(f"Action一致性: {analysis.get('action_consistency', 'N/A')}")

if __name__ == "__main__":
    main()
