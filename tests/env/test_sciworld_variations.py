#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试ScienceWorld环境中不同泛化等级（train、dev、test）或L2级别（任务集合分割）的任务/变体是否存在交集。
"""

import argparse
from scienceworld import ScienceWorldEnv
import pandas as pd
from tabulate import tabulate
import os
import yaml


def load_sciworld_config(config_path):
    """
    读取ppo_trainer.yaml中的scienceworld相关配置
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    sciworld_cfg = config.get('env', {}).get('sciworld', {})
    return sciworld_cfg


def check_l2_task_overlap(train_tasks, test_ood_tasks):
    """
    检查L2泛化下训练任务和测试任务编号是否有重叠
    """
    train_set = set(train_tasks)
    test_set = set(test_ood_tasks)
    overlap = train_set & test_set
    print("\nL2泛化任务集合检查:")
    print(f"训练任务数: {len(train_set)}，测试任务数: {len(test_set)}")
    if overlap:
        print(f"警告: 训练任务和测试任务有重叠: {sorted(list(overlap))}")
    else:
        print("训练任务和测试任务集合没有交集。")
    return overlap


def check_variation_overlap(jar_path=None, verbose=False, task_nums=None):
    """
    检查ScienceWorld环境中不同泛化等级的变体是否存在交集。
    支持只检查部分任务（task_nums）。
    """
    print("初始化ScienceWorld环境...")
    env = ScienceWorldEnv("", jar_path) if jar_path else ScienceWorldEnv()
    task_names = env.get_task_names()
    print(f"找到{len(task_names)}个任务。")

    # 只检查指定的任务编号
    if task_nums is not None:
        task_indices = [i for i in task_nums if i < len(task_names)]
    else:
        task_indices = list(range(len(task_names)))

    results = []
    for idx in task_indices:
        task = task_names[idx]
        print(f"[{idx+1}/{len(task_names)}] 检查任务 '{task}' (编号: {idx})")
        env.load(task)
        train_variations = set(env.get_variations_train())
        dev_variations = set(env.get_variations_dev())
        test_variations = set(env.get_variations_test())
        max_variations = env.get_max_variations(task)
        all_variations = train_variations | dev_variations | test_variations
        train_dev_overlap = train_variations & dev_variations
        train_test_overlap = train_variations & test_variations
        dev_test_overlap = dev_variations & test_variations
        result = {
            "任务名称": task,
            "任务编号": idx,
            "训练集变体数": len(train_variations),
            "开发集变体数": len(dev_variations),
            "测试集变体数": len(test_variations),
            "变体总数": max_variations,
            "已划分变体数": len(all_variations),
            "未划分变体数": max_variations - len(all_variations),
            "训练集-开发集重叠": len(train_dev_overlap) > 0,
            "训练集-测试集重叠": len(train_test_overlap) > 0,
            "开发集-测试集重叠": len(dev_test_overlap) > 0
        }
        if verbose:
            print(f"  训练集变体: {sorted(train_variations)}")
            print(f"  开发集变体: {sorted(dev_variations)}")
            print(f"  测试集变体: {sorted(test_variations)}")
            if train_dev_overlap:
                print(f"  警告: 训练集和开发集有重叠: {sorted(train_dev_overlap)}")
            if train_test_overlap:
                print(f"  警告: 训练集和测试集有重叠: {sorted(train_test_overlap)}")
            if dev_test_overlap:
                print(f"  警告: 开发集和测试集有重叠: {sorted(dev_test_overlap)}")
            if max_variations != len(all_variations):
                unassigned = set(range(max_variations)) - all_variations
                print(f"  提示: 有{max_variations - len(all_variations)}个变体未分配: {sorted(unassigned)}")
        results.append(result)
    env.close()
    df = pd.DataFrame(results)
    has_overlap = any(df["训练集-开发集重叠"]) or any(df["训练集-测试集重叠"]) or any(df["开发集-测试集重叠"])
    print("\n" + "=" * 80)
    if has_overlap:
        print("警告: 发现变体集合存在重叠!")
    else:
        print("所有泛化等级的变体集合都没有交集。")
    print("=" * 80 + "\n")
    return df


def print_summary(df):
    total_tasks = len(df)
    train_dev_overlap = df["训练集-开发集重叠"].sum()
    train_test_overlap = df["训练集-测试集重叠"].sum()
    dev_test_overlap = df["开发集-测试集重叠"].sum()
    print(f"共检查了{total_tasks}个任务")
    print(f"发现{train_dev_overlap}个任务的训练集和开发集有重叠")
    print(f"发现{train_test_overlap}个任务的训练集和测试集有重叠")
    print(f"发现{dev_test_overlap}个任务的开发集和测试集有重叠")
    total_variations = df["变体总数"].sum()
    assigned_variations = df["已划分变体数"].sum()
    train_variations = df["训练集变体数"].sum()
    dev_variations = df["开发集变体数"].sum()
    test_variations = df["测试集变体数"].sum()
    print(f"\n变体总数: {total_variations}")
    print(f"已划分变体数: {assigned_variations} ({assigned_variations / total_variations * 100:.1f}%)")
    print(f"训练集变体: {train_variations} ({train_variations / assigned_variations * 100:.1f}%)")
    print(f"开发集变体: {dev_variations} ({dev_variations / assigned_variations * 100:.1f}%)")
    print(f"测试集变体: {test_variations} ({test_variations / assigned_variations * 100:.1f}%)")
    problematic_tasks = df[(df["训练集-开发集重叠"]) | (df["训练集-测试集重叠"]) | (df["开发集-测试集重叠"])]
    if len(problematic_tasks) > 0:
        print("\n存在变体重叠的任务:")
        for _, row in problematic_tasks.iterrows():
            overlaps = []
            if row["训练集-开发集重叠"]:
                overlaps.append("训练集-开发集")
            if row["训练集-测试集重叠"]:
                overlaps.append("训练集-测试集")
            if row["开发集-测试集重叠"]:
                overlaps.append("开发集-测试集")
            print(f"- {row['任务名称']} (编号: {row['任务编号']}): {', '.join(overlaps)}重叠")


def main():
    parser = argparse.ArgumentParser(description='检查ScienceWorld环境中不同泛化等级或L2泛化的任务/变体是否存在交集')
    parser.add_argument('--jar', type=str, help='ScienceWorld JAR文件路径')
    parser.add_argument('--verbose', action='store_true', help='显示详细输出')
    parser.add_argument('--output', type=str, help='输出结果的CSV文件路径')
    parser.add_argument('--config', type=str, default='verl/trainer/config/ppo_trainer.yaml', help='ppo_trainer.yaml配置文件路径')
    args = parser.parse_args()

    # 读取配置
    sciworld_cfg = load_sciworld_config(args.config)
    generalization_level = sciworld_cfg['generalization_level']
    task_nums = sciworld_cfg['task_nums']
    train_tasks = sciworld_cfg['train_tasks']
    test_ood_tasks = sciworld_cfg['test_ood_tasks']

    print(f"generalization_level: {generalization_level}")
    print(f"task_nums: {task_nums}")
    print(f"train_tasks: {train_tasks}")
    print(f"test_ood_tasks: {test_ood_tasks}")

    if generalization_level == 2 and train_tasks is not None and test_ood_tasks is not None:
        # L2泛化：检查任务编号交集
        overlap = check_l2_task_overlap(train_tasks, test_ood_tasks)
        print("\nL2泛化任务集合检查完成。\n")
        print("训练任务编号:", train_tasks)
        print("测试任务编号:", test_ood_tasks)
        if overlap:
            print("存在任务编号重叠，请检查配置！")
        else:
            print("任务编号分割合理，无重叠。")
    else:
        # 其他情况：检查变体交集
        df = check_variation_overlap(jar_path=args.jar, verbose=args.verbose, task_nums=task_nums)
        print("\n结果摘要:")
        print_summary(df)
        print("\n详细结果:")
        display_cols = ["任务名称", "任务编号", "训练集变体数", "开发集变体数", "测试集变体数", 
                       "变体总数", "已划分变体数", "训练集-开发集重叠", 
                       "训练集-测试集重叠", "开发集-测试集重叠"]
        print(tabulate(df[display_cols], headers='keys', tablefmt='grid', showindex=False))
        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            df.to_csv(args.output, index=False, encoding='utf-8')
            print(f"\n结果已保存到: {args.output}")

if __name__ == "__main__":
    main() 