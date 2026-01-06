#!/usr/bin/env python3
"""
测试RLVCR chunk算法在不同action_token大小下的鲁棒性
"""
import math

def test_chunk_algorithm(token_count, gpu_count=8, confidence_chunk_size=32):
    """测试chunk算法"""
    print(f"\n=== 测试: {token_count} tokens, {gpu_count} GPUs, chunk_size={confidence_chunk_size} ===")
    
    # 基础chunk size
    base_prefix_chunk_size = max(1, confidence_chunk_size // token_count)
    print(f"基础prefix chunk size: {confidence_chunk_size} ÷ {token_count} = {base_prefix_chunk_size}")
    
    # 计算LCM和最小对齐chunk
    lcm_tokens_gpus = (token_count * gpu_count) // math.gcd(token_count, gpu_count)
    min_prefix_chunk_for_alignment = lcm_tokens_gpus // token_count
    print(f"LCM({token_count}, {gpu_count}) = {lcm_tokens_gpus}")
    print(f"最小对齐chunk: {lcm_tokens_gpus} ÷ {token_count} = {min_prefix_chunk_for_alignment}")
    
    # 决定最终chunk size
    if base_prefix_chunk_size >= min_prefix_chunk_for_alignment:
        prefix_chunk_size = (base_prefix_chunk_size // min_prefix_chunk_for_alignment) * min_prefix_chunk_for_alignment
        print(f"使用倍数: ({base_prefix_chunk_size} ÷ {min_prefix_chunk_for_alignment}) × {min_prefix_chunk_for_alignment} = {prefix_chunk_size}")
    else:
        prefix_chunk_size = min_prefix_chunk_for_alignment
        print(f"使用最小值: {min_prefix_chunk_for_alignment}")
    
    # 验证对齐
    total_computations = prefix_chunk_size * token_count
    is_aligned = total_computations % gpu_count == 0
    print(f"最终: {prefix_chunk_size} prefixes × {token_count} tokens = {total_computations} 计算")
    print(f"GPU对齐: {total_computations} % {gpu_count} = {total_computations % gpu_count} {'✅' if is_aligned else '❌'}")
    
    # 计算效率
    efficiency = base_prefix_chunk_size / prefix_chunk_size if prefix_chunk_size > 0 else 0
    print(f"效率: {efficiency:.2%} (理想chunk vs 实际chunk)")
    
    return is_aligned, efficiency

def main():
    """测试不同token大小"""
    test_cases = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20, 24, 32]
    
    print("RLVCR Chunk算法鲁棒性测试")
    print("=" * 50)
    
    results = []
    for token_count in test_cases:
        is_aligned, efficiency = test_chunk_algorithm(token_count)
        results.append((token_count, is_aligned, efficiency))
    
    print("\n" + "=" * 50)
    print("汇总结果:")
    print("Token大小 | 对齐 | 效率")
    print("-" * 25)
    
    for token_count, is_aligned, efficiency in results:
        align_status = "✅" if is_aligned else "❌"
        print(f"{token_count:8d} | {align_status:2s} | {efficiency:6.1%}")
    
    # 检查是否所有情况都对齐
    all_aligned = all(is_aligned for _, is_aligned, _ in results)
    print(f"\n总体鲁棒性: {'✅ 完全鲁棒' if all_aligned else '❌ 存在问题'}")
    
    # 效率分析
    avg_efficiency = sum(eff for _, _, eff in results) / len(results)
    print(f"平均效率: {avg_efficiency:.1%}")
    
    low_efficiency_cases = [(tc, eff) for tc, _, eff in results if eff < 0.5]
    if low_efficiency_cases:
        print(f"低效率情况 (<50%): {low_efficiency_cases}")

if __name__ == "__main__":
    main()
