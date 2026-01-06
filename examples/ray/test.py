#!/usr/bin/env python3
"""
Ray 基础教程 - 从零开始理解分布式计算
"""
import ray
import time
import numpy as np

# 1. 启动 Ray
print("=== 启动 Ray ===")
ray.init()

# 2. 普通函数 vs Ray 远程函数
print("\n=== 对比普通计算和 Ray 并行计算 ===")

# 普通函数
def normal_slow_function(x):
    """一个慢函数，模拟耗时计算"""
    time.sleep(2)  # 模拟2秒的计算
    return x * x

# Ray 远程函数 - 加上 @ray.remote 装饰器
@ray.remote
def ray_slow_function(x):
    """Ray 版本的慢函数"""
    time.sleep(2)  # 同样2秒计算
    return x * x

# 普通方式：串行计算，很慢
print("普通串行计算（会很慢）...")
start_time = time.time()
normal_results = []
for i in range(4):
    result = normal_slow_function(i)
    normal_results.append(result)
normal_time = time.time() - start_time
print(f"普通方式用时: {normal_time:.2f} 秒，结果: {normal_results}")

# Ray 方式：并行计算，很快
print("\nRay 并行计算（会很快）...")
start_time = time.time()
# 启动4个并行任务
ray_futures = []
for i in range(4):
    future = ray_slow_function.remote(i)  # .remote() 立即返回，不等待
    ray_futures.append(future)

# 等待所有任务完成
ray_results = ray.get(ray_futures)  # ray.get() 获取实际结果
ray_time = time.time() - start_time
print(f"Ray 方式用时: {ray_time:.2f} 秒，结果: {ray_results}")
print(f"加速比: {normal_time/ray_time:.1f}x")

# 3. Ray 远程类 - 可以保持状态
print("\n=== Ray 远程类示例 ===")

@ray.remote
class Counter:
    """一个分布式计数器"""
    def __init__(self):
        self.value = 0
    
    def increment(self):
        self.value += 1
        return self.value
    
    def get_value(self):
        return self.value
    
    def add(self, x):
        self.value += x
        return self.value

# 创建多个计数器实例，每个都在不同的进程中
counters = [Counter.remote() for _ in range(3)]

# 每个计数器独立工作
print("3个独立的计数器：")
for i, counter in enumerate(counters):
    value = ray.get(counter.add.remote(i * 10))
    print(f"计数器 {i}: {value}")

# 4. 实际应用：并行矩阵计算
print("\n=== 实际应用：大矩阵并行计算 ===")

@ray.remote
def matrix_multiply_chunk(A_chunk, B):
    """计算矩阵乘法的一个块"""
    return np.dot(A_chunk, B)

# 创建两个大矩阵
size = 1000
A = np.random.rand(size, size)
B = np.random.rand(size, size)

# 将矩阵 A 分成4块，并行计算
chunk_size = size // 4
chunks = [A[i*chunk_size:(i+1)*chunk_size] for i in range(4)]

print("开始并行矩阵乘法...")
start_time = time.time()

# 并行计算每个块
futures = [matrix_multiply_chunk.remote(chunk, B) for chunk in chunks]
results = ray.get(futures)

# 合并结果
final_result = np.vstack(results)
parallel_time = time.time() - start_time

print(f"并行计算用时: {parallel_time:.2f} 秒")
print(f"结果矩阵形状: {final_result.shape}")

# 验证结果正确性
print("验证结果正确性...")
start_time = time.time()
expected = np.dot(A, B)
serial_time = time.time() - start_time

print(f"串行计算用时: {serial_time:.2f} 秒")
print(f"结果是否正确: {np.allclose(final_result, expected)}")
print(f"并行加速比: {serial_time/parallel_time:.1f}x")

# 5. 清理
print("\n=== 关闭 Ray ===")
ray.shutdown()
print("Ray 已关闭")