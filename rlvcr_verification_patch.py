#!/usr/bin/env python3
"""
RLVCR Confidence计算验证补丁
可以直接插入到 _compute_confidence_chunk 函数中
"""

verification_code = '''
# 在 _compute_confidence_chunk 函数的调试部分添加这段代码：
# 位置：在计算完prefix_probs之后，打印confidence之前

if debug_mode and prefix_idx == 0:  # 只验证第一个prefix，避免过多输出
    print(f"\\n🧪 TOKENIZATION & EQUIVALENCE VERIFICATION:")
    
    # 1. 验证tokenization一致性
    current_action_tokens = action_token_ids
    action_text = self.tokenizer.decode(action_token_ids, skip_special_tokens=True)
    
    print(f"  Action text: '{action_text}'")
    print(f"  Current method tokens: {current_action_tokens}")
    print(f"  Token texts: {[self.tokenizer.decode([t]) for t in current_action_tokens]}")
    
    # 直接方法tokenization
    try:
        full_input_text = input_prefixes[prefix_idx] + action_text
        full_tokens = self.tokenizer.encode(full_input_text, add_special_tokens=False)
        prefix_tokens = self.tokenizer.encode(input_prefixes[prefix_idx], add_special_tokens=False)
        
        prefix_len = len(prefix_tokens)
        direct_action_tokens = full_tokens[prefix_len:]
        
        print(f"  Direct method tokens: {direct_action_tokens}")
        print(f"  Direct token texts: {[self.tokenizer.decode([t]) for t in direct_action_tokens]}")
        
        tokens_match = (current_action_tokens == direct_action_tokens)
        print(f"  Tokenization match: {tokens_match}")
        
        if not tokens_match:
            print(f"  ❌ TOKENIZATION MISMATCH!")
            print(f"     This is likely the source of 'strange results'")
            print(f"     Current approach calculates probabilities for different tokens!")
            
            # 分析差异
            print(f"\\n  📊 Detailed token comparison:")
            max_len = max(len(current_action_tokens), len(direct_action_tokens))
            for i in range(max_len):
                curr_token = current_action_tokens[i] if i < len(current_action_tokens) else None
                direct_token = direct_action_tokens[i] if i < len(direct_action_tokens) else None
                curr_text = self.tokenizer.decode([curr_token]) if curr_token else None
                direct_text = self.tokenizer.decode([direct_token]) if direct_token else None
                
                if curr_token != direct_token:
                    print(f"     Position {i}: {curr_token}('{curr_text}') vs {direct_token}('{direct_text}') ❌")
                else:
                    print(f"     Position {i}: {curr_token}('{curr_text}') ✅")
        else:
            print(f"  ✅ Tokenization is consistent")
            
    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
    
    # 2. 概率计算等价性验证（如果tokenization一致）
    if 'tokens_match' in locals() and tokens_match:
        print(f"\\n  🔍 Probability calculation verification:")
        print(f"     Current method probabilities: {[f'{p:.6f}' for p in prefix_probs]}")
        print(f"     Min probability: {min(prefix_probs):.6f}")
        print(f"     Geometric mean: {(np.prod(prefix_probs)**(1/len(prefix_probs))):.6f}")
        print(f"     Arithmetic mean: {np.mean(prefix_probs):.6f}")
        
        # 建议：如果想要更稳定的confidence measure
        print(f"\\n  💡 Alternative confidence measures:")
        joint_prob = np.prod(prefix_probs)
        geom_mean = joint_prob**(1/len(prefix_probs))
        print(f"     Joint probability: {joint_prob:.8f}")
        print(f"     Geometric mean: {geom_mean:.6f}")
        print(f"     Current (min): {min(prefix_probs):.6f}")
'''

print("🔧 RLVCR验证补丁")
print("=" * 50)
print()
print("将以下代码添加到你的 _compute_confidence_chunk 函数中：")
print("位置：在计算完 prefix_probs 之后，打印 confidence 之前")
print()
print(verification_code)
print()
print("🎯 这个补丁会验证：")
print("1. Tokenization一致性 - 检查逐token方法和直接方法的token序列是否相同")
print("2. 如果不一致，详细显示差异位置")
print("3. 提供不同的confidence计算方式对比")
print()
print("💡 如果发现tokenization不匹配，这就是'奇怪结果'的根源！")
print("   解决方案：切换到直接自回归方法，或者修复tokenization逻辑")

# 额外的分析
print("\n" + "=" * 50)
print("🔍 可能的Tokenization问题原因")
print("=" * 50)

print("""
常见的tokenization不一致情况：

1. 📝 上下文影响分词：
   - 单独: "open door" -> ['open', ' door']  
   - 上下文: "...action>open door" -> ['open', ' door']  # 可能不同！

2. 🔤 子词分割差异：
   - 'outside' 可能被分割为 [' out', 'side'] 或 [' outside']
   - 取决于前面的context和tokenizer的状态

3. 🎯 特殊token处理：
   - '<action>' tag的存在可能影响后续token的分割
   - 某些tokenizer对特殊符号敏感

4. 🔢 边界效应：
   - 序列长度限制可能导致截断位置不同
   - 影响最后几个token的分割

解决方案：
✅ 使用直接自回归方法（推荐）
✅ 或者确保tokenization的一致性验证
✅ 在debug模式下添加详细的token对比
""")

if __name__ == "__main__":
    pass
