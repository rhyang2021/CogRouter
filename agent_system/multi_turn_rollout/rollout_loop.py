import torch
import numpy as np
import re
import math
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from verl.models.transformers.qwen2_vl import get_rope_index
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict, Tuple, Any, Optional, Union
from collections import defaultdict

from rlvmr import core_rlvmr as core_mcrl
from rlvcr import core_rlvcr


class TrajectoryCollector:
    """
    RLVCR 执行流程：
    1. multi_turn_loop() - 主入口，选择执行模式
    2. rlvcr_multi_turn_loop() - RLVCR 主流程
       a. vanilla_multi_turn_loop() - 收集基础轨迹
       b. gather_rollout_data() - 整理数据
       c. _generate_alternative_thinking_levels() - 生成思考变体
    3. _generate_alternative_thinking_levels() - 生成4个思考等级
       a. 分组样本（失败/相同奖励/需扩展）
       b. _batch_generate_thinking_alternatives() - 批量生成思考
       c. _rlvcr_collate_fn() - 整理最终数据
    """
    
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    # ========== 主入口函数 ==========
    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (RLVCR, dynamic, or vanilla).
        """
        # Check if RLVCR sampling should be used
        if (hasattr(self.config, 'algorithm') and 
            hasattr(self.config.algorithm, 'adv_estimator') and 
            self.config.algorithm.adv_estimator == 'rlvcr' and
            hasattr(self.config.algorithm, 'rlvcr') and 
            self.config.algorithm.rlvcr.enable and
            is_train):
            # RLVCR Sampling
            return self.rlvcr_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        elif self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (DAPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_infos = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_infos = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )

        # Process non-RLVCR results
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)

        # add mcrl rewards into total_batch_list
        if self.config.algorithm.mcrl.enable:
            total_batch_list = core_mcrl.process_trajectory_rlvmr_rewards(
                trajectory_list=total_batch_list, config=self.config, episode_rewards=total_episode_rewards
            )

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            total_infos=total_infos,
        )
        
        if self.config.algorithm.mcrl.enable:
            gen_batch_output.meta_info["mcrl_step_advantage_w"] = (self.config.algorithm.mcrl.step_advantage_w)
            gen_batch_output.meta_info["mcrl_mode"] = self.config.algorithm.mcrl.mode
        return gen_batch_output

    # ========== RLVCR 主流程 ==========
    def rlvcr_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        RLVCR trajectory collection with multi-level thinking generation.
        执行步骤：
        1. 收集基础轨迹
        2. 转换为 DataProto 格式
        3. 生成思考等级变体
        4. 添加 RLVCR 元信息
        """
        # Step 1: Collect base trajectories
        total_batch_list, episode_rewards, episode_lengths, success, traj_uid, total_infos = \
            self.vanilla_multi_turn_loop(gen_batch, actor_rollout_wg, envs)
        
        # Step 2: Convert to DataProto format
        base_result = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            success=success, 
            traj_uid=traj_uid,
            total_infos=total_infos,
        )
        
        # Step 3: Generate alternative thinking levels
        base_result = self._generate_alternative_thinking_levels(base_result, actor_rollout_wg)
        
        # Step 4: Store RLVCR configuration
        rlvcr_config = self.config.algorithm.rlvcr
        base_result.meta_info.update({
            "rlvcr_thinking_weight": getattr(rlvcr_config, 'thinking_weight', 0.5),
            "rlvcr_thinking_diversity_w": getattr(rlvcr_config, 'thinking_diversity_w', 1.0),
            "rlvcr_mode": getattr(rlvcr_config, 'mode', 'mean_norm')
        })
    
        return base_result

    # ========== 思考等级生成 ==========
    def _generate_alternative_thinking_levels(
            self, 
            base_batch: DataProto, 
            actor_rollout_wg
        ) -> DataProto:
        """
        生成 RLVCR 的思考等级变体。
        执行步骤：
        1. 按 UID 分组，检查奖励相似性
        2. 分类样本：失败/相同奖励/需扩展
        3. 批量生成思考变体
        4. 整理并返回扩展后的数据
        """
        batch_size = base_batch.batch['input_ids'].shape[0]
        
        # Step 1: 按 UID 分组，检查奖励相似性
        uid_to_rewards = {}
        for batch_idx in range(batch_size):
            uid = base_batch.non_tensor_batch['uid'][batch_idx]
            reward = base_batch.non_tensor_batch['episode_rewards'][batch_idx]
            
            if uid not in uid_to_rewards:
                uid_to_rewards[uid] = []
            uid_to_rewards[uid].append(reward)
        
        # 识别奖励相同的 UID（这些不需要扩展）
        identical_reward_uids = set()
        different_reward_uids = set()
        
        for uid, rewards in uid_to_rewards.items():
            if len(rewards) > 1:
                reward_range = max(rewards) - min(rewards)
                if reward_range < 1e-6:  # 浮点精度阈值
                    identical_reward_uids.add(uid)
                    print(f"RLVCR: UID {uid} has identical rewards ({rewards[0]:.6f}), will skip thinking expansion")
                else:
                    different_reward_uids.add(uid)
        
        print(f"RLVCR: {len(identical_reward_uids)} UIDs with identical rewards (will skip expansion)")
        print(f"RLVCR: {len(different_reward_uids)} UIDs with different rewards (will expand)")
        
        # Step 2: 分类样本
        failed_indices = []  # 失败轨迹（不再使用，但保留结构）
        skip_thinking_indices = []  # 成功但奖励相同
        expand_thinking_indices = []  # 成功且奖励不同（需要扩展）
        
        for batch_idx in range(batch_size):
            uid = base_batch.non_tensor_batch['uid'][batch_idx]
            episode_reward = base_batch.non_tensor_batch['episode_rewards'][batch_idx]

            if episode_reward <= 0:
                failed_indices.append(batch_idx)
            elif uid in identical_reward_uids:
                skip_thinking_indices.append(batch_idx)
            else:
                expand_thinking_indices.append(batch_idx)
        
        print(f"RLVCR: {len(failed_indices)} failed trajectories")
        print(f"RLVCR: {len(skip_thinking_indices)} trajectories with identical rewards (skip expansion)")
        print(f"RLVCR: {len(expand_thinking_indices)} trajectories to expand")
        
        # Step 3: 处理各类样本
        all_samples = []
        original_indices = []
        expanded_indices = []
        current_idx = 0
        
        # Process failed trajectories (simple samples)
        for batch_idx in failed_indices:
            sample = self._extract_single_sample(base_batch, batch_idx)
            sample.update({
                'is_original': True,
                'thinking_group_id': -1,
                'thinking_entropy': 0.0,
                'thinking_cost': 0,
                'original_level': -1,
                'current_level': -1
            })
            all_samples.append(sample)
            original_indices.append(current_idx)
            current_idx += 1

        # 处理奖励相同的轨迹（不扩展）
        for batch_idx in skip_thinking_indices:
            sample = self._extract_single_sample(base_batch, batch_idx)
            sample.update({
                'is_original': True,
                'thinking_group_id': -1,
                'thinking_entropy': 0.0,
                'thinking_cost': 0,
                'original_level': -1,
                'current_level': -1
            })
            all_samples.append(sample)
            original_indices.append(current_idx)
            current_idx += 1
        
        # 处理需要扩展的轨迹
        invalid_format_count = 0
        if expand_thinking_indices:
            print(f"RLVCR: Batch processing {len(expand_thinking_indices)} trajectories for thinking expansion")
            successful_samples, invalid_format_count = self._batch_generate_thinking_alternatives(
                expand_thinking_indices, base_batch, actor_rollout_wg
            )
            print(f"RLVCR: Generated {len(successful_samples)} trajectory groups")
            
            for samples_group in successful_samples:
                if len(samples_group) == 4:
                    # 有效的 4 个等级样本
                    for i, sample in enumerate(samples_group):
                        sample['thinking_group_id'] = samples_group[0]['batch_idx']
                        if i == sample['original_level'] - 1:
                            sample['is_original'] = True
                            original_indices.append(current_idx)
                        else:
                            sample['is_original'] = False
                            expanded_indices.append(current_idx)
                        all_samples.append(sample)
                        current_idx += 1
                else:
                    # 无效格式 - 使用单个样本
                    sample = samples_group[0]
                    sample.update({
                        'is_original': True,
                        'thinking_group_id': -1
                    })
                    all_samples.append(sample)
                    original_indices.append(current_idx)
                    current_idx += 1
        
        # 重新分配批次索引
        for new_idx, sample in enumerate(all_samples):
            sample['batch_idx'] = new_idx
        
        print(f"RLVCR: Final sample count:")
        print(f"  - Identical rewards (skipped): {len(skip_thinking_indices)}")
        print(f"  - Invalid format (skipped): {invalid_format_count}")
        print(f"  - Expanded groups: {len(successful_samples) if 'successful_samples' in locals() else 0}")
        print(f"  - Total samples: {len(all_samples)}")
        
        # Step 4: 创建扩展后的批次
        expanded_batch = DataProto.from_single_dict(
            data=self._rlvcr_collate_fn(all_samples),
            meta_info=base_batch.meta_info
        )
        
        # 存储元数据
        expanded_batch.meta_info.update({
            'original_indices': original_indices,
            'expanded_indices': expanded_indices,
            'rlvcr_invalid_format_count': invalid_format_count,
            'rlvcr_identical_reward_count': len(skip_thinking_indices),
            'rlvcr_expanded_count': len(expand_thinking_indices)
        })
        
        return expanded_batch

    # ========== 批量生成思考变体 ==========
    def _batch_generate_thinking_alternatives(
        self,
        successful_indices: List[int],
        base_batch: DataProto,
        actor_rollout_wg
    ) -> List[List[dict]]:
        """
        批量生成思考变体。
        执行步骤：
        1. 解析原始响应，提取动作和等级
        2. 生成其他等级的思考文本
        3. 计算所有等级的置信度（熵）
        4. 组装 4 个等级的样本
        """
        if not successful_indices:
            return []
        
        trajectory_metadata = []
        all_generation_prompts = []
        
        # Step 1: 解析轨迹并收集生成提示
        for batch_idx in successful_indices:
            original_sample = self._extract_single_sample(base_batch, batch_idx)
            original_response = self.tokenizer.decode(base_batch.batch['responses'][batch_idx], skip_special_tokens=True)
            
            original_thinking = self._extract_from_response(original_response, 'think')
            original_action = self._extract_from_response(original_response, 'action')
            original_level = self._extract_level_from_response(original_response)

            if not original_action or not original_thinking or original_level is None:
            # if not original_action or original_level is None:
                print(f"RLVCR: Original response for trajectory {batch_idx} has invalid format, skipping expansion")
                trajectory_metadata.append({
                    'batch_idx': batch_idx, 'original_sample': original_sample,
                    'original_response': original_response, 'valid': False
                })
                continue
            
            # 准备生成其他等级的提示
            prompt_text = self.tokenizer.decode(base_batch.batch['input_ids'][batch_idx], skip_special_tokens=True)
            action_token_ids = self.tokenizer.encode(original_action, add_special_tokens=False)
            generation_prompts_for_traj = []
            
            for level in [1, 2, 3, 4]:
                if level != original_level and level != 1:  # Level 1 不需要思考，其他等级需要生成
                    modified_prompt = core_rlvcr.create_level_specific_prompt(
                        history=prompt_text, target_level=level, action=original_action
                    )
                    generation_prompts_for_traj.append(modified_prompt)
            
            trajectory_metadata.append({
                'batch_idx': batch_idx, 'original_sample': original_sample,
                'original_response': original_response, 'original_thinking': original_thinking,
                'original_action': original_action, 'original_level': original_level,
                'prompt_text': prompt_text, 'action_token_ids': action_token_ids,
                'generation_prompts': generation_prompts_for_traj, 'valid': True
            })
            all_generation_prompts.extend(generation_prompts_for_traj)
        
        # Step 2: 批量生成思考序列
        generated_thinking_sequences = []
        if all_generation_prompts:
            print(f"RLVCR: Generating {len(all_generation_prompts)} thinking sequences")
            generated_thinking_sequences = self._batch_generate_thinking_sequences(
                all_generation_prompts, actor_rollout_wg, base_batch.meta_info.copy()
            )
        
        # Step 3: 准备置信度计算
        all_confidence_prompts = []
        valid_trajectories = [traj for traj in trajectory_metadata if traj['valid']]
        
        generation_idx = 0
        for traj_meta in valid_trajectories:
            thinking_texts = []
            for level in [1, 2, 3, 4]:
                if level == traj_meta['original_level']:
                    thinking_texts.append(traj_meta['original_thinking'])
                elif level == 1:
                    thinking_texts.append("Okay, I think I have finished thinking.")  # Level 1 固定思考文本
                    # thinking_texts.append("")
                else:
                    thinking_texts.append(generated_thinking_sequences[generation_idx])
                    generation_idx += 1
            
            traj_meta['thinking_texts'] = thinking_texts
            
            # 为每个等级创建置信度计算提示
            for level in [1, 2, 3, 4]:
                thinking_text = thinking_texts[level-1]
                # if level == 1:
                    # input_prefix = traj_meta['original_sample']['raw_prompt'][0]['content'] + f"<level>1</level><action>"
                # else:
                input_prefix = traj_meta['original_sample']['raw_prompt'][0]['content'] + f"<level>{level}</level><think>{thinking_text}</think><action>"
                all_confidence_prompts.append(input_prefix)
        
        # Step 4: 批量计算所有置信度
        all_batch_entropies = []
        if all_confidence_prompts and valid_trajectories:
            representative_action_token_ids = valid_trajectories[0]['action_token_ids']
            all_batch_entropies = self._batch_compute_action_token_logits(
                all_confidence_prompts, representative_action_token_ids, actor_rollout_wg
            )
        
        # Step 5: 处理结果，组装 4 个等级的样本
        results = []
        valid_count = 0
        invalid_count = 0
        
        for traj_meta in trajectory_metadata:
            if not traj_meta['valid']:
                # 无效轨迹，返回单个样本
                invalid_count += 1
                sample = traj_meta['original_sample'].copy()
                sample.update({
                    'full_output': traj_meta['original_response'],
                    'thinking_entropy': 0.0, 'thinking_cost': 0,
                    'original_level': 1, 'current_level': 1
                })
                results.append([sample])
                continue
            
            valid_count += 1
            
            # 创建 4 个等级的样本
            samples = []
            traj_idx = valid_trajectories.index(traj_meta)
            for level in [1, 2, 3, 4]:
                sample = traj_meta['original_sample'].copy()
                sample['batch_idx'] = traj_meta['batch_idx']
                
                if level == traj_meta['original_level']:
                    # 使用原始响应
                    sample['responses'] = base_batch.batch['responses'][traj_meta['batch_idx']]
                    sample['full_output'] = traj_meta['original_response']
                else:
                    # 创建新响应
                    new_response = self._create_response_with_level(
                        level, traj_meta['original_action'], traj_meta['thinking_texts'][level-1]
                    )
                    sample['responses'] = self.tokenizer.encode(new_response, return_tensors='pt')[0]
                    sample['full_output'] = new_response
                
                # 添加熵和成本信息
                entropy_idx = traj_idx * 4 + (level - 1)
                sample['thinking_entropy'] = all_batch_entropies[entropy_idx] if entropy_idx < len(all_batch_entropies) else 0.0
                sample['thinking_cost'] = len(self.tokenizer.encode(traj_meta['thinking_texts'][level-1], add_special_tokens=False)) if traj_meta['thinking_texts'][level-1] else 0
                sample['original_level'] = traj_meta['original_level']
                sample['current_level'] = level
                samples.append(sample)
            
            results.append(samples)
        
        print(f"RLVCR: {valid_count} valid, {invalid_count} invalid trajectories")
        return results, invalid_count

    # ========== 基础轨迹收集 ==========
    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> Tuple:
        """
        收集基础轨迹（标准的多轮交互）。
        """
        # Initial observations from the environment
        obs, infos = envs.reset()
        
        # Initialize trajectory collection
        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        if len(gen_batch.batch) != lenght_obs and self.config.env.rollout.n > 0:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"

        batch_size = len(gen_batch.batch['input_ids'])
        
        # 环境分组
        if self.config.env.rollout.n > 0:
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else:
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.int32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                batch_input = batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                batch_input = batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            batch_input.meta_info = gen_batch.meta_info

            batch_output = actor_rollout_wg.generate_sequences(batch_input)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)

            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            next_obs, rewards, dones, infos = envs.step(text_actions)
            print(rewards)

            batch.non_tensor_batch['full_output'] = text_actions

            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            batch.non_tensor_batch['action_available'] = np.array([info['action_available'] for info in infos], dtype=bool)

            # Create reward tensor, only assign rewards for active environments
            episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_lengths[active_masks] += 1

            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)

            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)

            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break

        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )

        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, total_infos

    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> Tuple:
        """
        动态采样（DAPO）- 持续采样直到满足目标批大小。
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, infos = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid = filter_group_data(
                batch_list=batch_list,
                episode_rewards=episode_rewards, 
                episode_lengths=episode_lengths, 
                success=success, 
                traj_uid=traj_uid, 
                config=self.config,
                last_try=(try_count == max_try_count),
            )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, None

    # ========== 辅助函数（按调用顺序） ==========
    
    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        处理一批观察样本，转换为模型可处理的格式。
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []

        for item in range(batch_size):
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)

        batch = collate_fn(processed_samples)
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        处理单个观察样本。
        """
        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        obs_content = raw_prompt[0]['content']
        if '<image>' in obs_content: 
            obs_content = obs_content.replace('<image>', '')

        if obs_text is not None:
            obs_content += obs_text
        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])

        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False
        )

        # Initialize return dict
        row_dict = {}

        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation='error'
        )
        
        if is_multi_modal:
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': self.tokenizer.encode(raw_prompt, add_special_tokens=False),
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()

        return row_dict

    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            total_infos: List[List[Dict]] = None,
            ) -> DataProto:
        """
        收集并组织轨迹数据。
        """
        batch_size = len(total_batch_list)

        episode_rewards_mean = np.mean(episode_rewards)
        episode_rewards_min = np.min(episode_rewards)
        episode_rewards_max = np.max(episode_rewards)

        episode_lengths_mean = np.mean(episode_lengths)
        episode_lengths_min = np.min(episode_lengths)
        episode_lengths_max = np.max(episode_lengths)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)

        effective_batch = []
        traj_length = []

        for bs in range(batch_size):
            valid_step = 0
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    data['episode_rewards_mean'] = episode_rewards_mean
                    data['episode_rewards_min'] = episode_rewards_min
                    data['episode_rewards_max'] = episode_rewards_max
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    data['episode_lengths_mean'] = episode_lengths_mean
                    data['episode_lengths_min'] = episode_lengths_min
                    data['episode_lengths_max'] = episode_lengths_max
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value
                    
                    # 添加环境信息（如果有）
                    if total_infos is not None and bs < len(total_infos) and len(total_infos[bs]) > 0:
                        first_info = total_infos[bs][0]
                        if 'task_description' in first_info:
                            data['task_description'] = first_info['task_description']
                        if 'task_num' in first_info:
                            data['task_num'] = first_info['task_num']
                        
                        step_idx_in_traj = valid_step
                        if step_idx_in_traj < len(total_infos[bs]):
                            current_info = total_infos[bs][step_idx_in_traj]
                            if 'observation_text' in current_info:
                                data['observation_text'] = current_info['observation_text']
                            if 'score' in current_info:
                                data['score'] = current_info['score']
                            if 'task_score' in current_info:
                                data['task_score'] = current_info['task_score']
                            if 'won' in current_info:
                                data['won'] = current_info['won']
                            if 'available_actions' in current_info:
                                data['available_actions'] = current_info['available_actions']
                            if 'possible_actions' in current_info:
                                data['possible_actions'] = current_info['possible_actions']
                    
                    valid_step += 1
                    effective_batch.append(data)
            traj_length.append(valid_step)  

        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )

        gen_batch_output.meta_info["traj_length"] = traj_length
        traj_length_stat = {
            "traj_length_mean": np.mean(traj_length),
            "traj_length_min": np.min(traj_length),
            "traj_length_max": np.max(traj_length),
        }
        gen_batch_output.meta_info["traj_length_stat"] = traj_length_stat

        return gen_batch_output

    def _extract_single_sample(self, batch: DataProto, idx: int) -> dict:
        """
        从批次中提取单个样本。
        """
        sample = {}
        
        # Extract tensor data
        for key in batch.batch.keys():
            tensor = batch.batch[key]
            if len(tensor.shape) > 0:
                sample[key] = tensor[idx]
            else:
                sample[key] = tensor
        
        # Extract non-tensor data
        for key in batch.non_tensor_batch.keys():
            sample[key] = batch.non_tensor_batch[key][idx]
        
        return sample

    def _extract_from_response(self, response: str, tag: str) -> str:
        """
        从响应中提取指定标签的内容。
        """
        pattern = rf'<{tag}>(.*?)</{tag}>'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    def _extract_level_from_response(self, response: str) -> Optional[int]:
        """
        从响应中提取思考等级。
        """
        level_str = self._extract_from_response(response, 'level')
        if not level_str:
            return None
        try:
            level = int(level_str)
            return level if 1 <= level <= 4 else None
        except ValueError:
            return None


    def _create_response_with_level(self, level: int, action: str, thinking_text: str) -> str:
        """
        创建带有指定思考等级的完整响应。
        """
        if level == 1:
            return f"<level>1</level>\n<think>\nOkay, I think I have finished thinking.\n</think>\n<action>{action}</action>"
        # if level == 1:
            # return f"<level>1</level>\n<action>{action}</action>"
        else:
            return f"<level>{level}</level>\n<think>\n{thinking_text}\n</think>\n<action>{action}</action>"

    def _batch_generate_thinking_sequences(
        self,
        prompts: List[str],
        actor_rollout_wg,
        meta_info: dict
    ) -> List[str]:
        """
        批量生成思考序列（带分块以确保内存安全）。
        """
        if not prompts:
            return []
        
        # Get chunking configuration
        generation_chunk_size = getattr(self.config.algorithm.rlvcr, 'generation_chunk_size', 256)
        
        # Process in chunks if needed
        if len(prompts) > generation_chunk_size:
            all_results = []
            for i in range(0, len(prompts), generation_chunk_size):
                chunk_prompts = prompts[i:i + generation_chunk_size]
                chunk_results = self._generate_thinking_chunk(chunk_prompts, actor_rollout_wg, meta_info)
                all_results.extend(chunk_results)
            return all_results
        else:
            return self._generate_thinking_chunk(prompts, actor_rollout_wg, meta_info)
    
    def _generate_thinking_chunk(
        self,
        prompts: List[str],
        actor_rollout_wg,
        meta_info: dict
    ) -> List[str]:
        """
        为一批提示生成思考序列。
        """
        # Tokenize and pad inputs
        batch_inputs = []
        for prompt in prompts:
            tokenized = self.tokenizer(
                prompt, return_tensors='pt', padding=False, truncation=True,
                max_length=meta_info.get('max_prompt_length', 4096)
            )
            batch_inputs.append({
                'input_ids': tokenized['input_ids'][0],
                'attention_mask': tokenized['attention_mask'][0]
            })
        
        # Pad to same length
        max_len = max(len(inp['input_ids']) for inp in batch_inputs)
        for inp in batch_inputs:
            pad_len = max_len - len(inp['input_ids'])
            if pad_len > 0:
                inp['input_ids'] = torch.cat([torch.full((pad_len,), self.tokenizer.pad_token_id), inp['input_ids']])
                inp['attention_mask'] = torch.cat([torch.zeros(pad_len), inp['attention_mask']])
        
        # Create generation batch with GPU-aligned size
        original_batch_size = len(batch_inputs)
        
        # Ensure batch size is divisible by GPU count for multi-GPU systems
        if hasattr(actor_rollout_wg, 'world_size') and actor_rollout_wg.world_size > 1:
            gpu_count = actor_rollout_wg.world_size
            if original_batch_size % gpu_count != 0:
                pad_size = gpu_count - (original_batch_size % gpu_count)
                for _ in range(pad_size):
                    batch_inputs.append(batch_inputs[-1])
        
        batch_input_ids = torch.stack([inp['input_ids'] for inp in batch_inputs])
        batch_attention_mask = torch.stack([inp['attention_mask'] for inp in batch_inputs])
        batch_position_ids = compute_position_id_with_mask(batch_attention_mask)
        
        generation_batch = DataProto.from_dict({
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'position_ids': batch_position_ids
        })
        generation_batch.meta_info = meta_info.copy()
        generation_batch.meta_info.update({
            'response_length': 512,
            'temperature': 0.7,
            'do_sample': True,
            'top_p': 0.9,
        })
        
        # Generate with multi-GPU handling
        output = self._handle_multi_gpu_generation(generation_batch, actor_rollout_wg)
        
        # Extract thinking sequences (only original batch size)
        results = []
        for i in range(len(prompts)):
            generated_response = self.tokenizer.decode(output.batch['responses'][i], skip_special_tokens=True)
            thinking_sequence = self._extract_from_response(generated_response, 'think')
            results.append(thinking_sequence if thinking_sequence else generated_response)
        
        return results

    def _batch_compute_action_token_logits(
        self,
        input_prefixes: List[str],
        action_token_ids: List[int],
        actor_rollout_wg
    ) -> List[float]:
        """
        批量计算动作令牌的置信度（优化版本）。
        """
        if not input_prefixes:
            return []
        
        # Dynamic chunking based on configuration and GPU memory
        total_computations = len(input_prefixes) * len(action_token_ids)
        
        # Calculate safe chunk size based on GPU micro batch size
        try:
            base_micro_batch_size = getattr(self.config.actor_rollout_ref.rollout, 'log_prob_micro_batch_size_per_gpu', 8)
        except AttributeError:
            base_micro_batch_size = 8
        
        try:
            # 优化1：大幅增加 batch size
            confidence_chunk_size = getattr(self.config.algorithm.rlvcr, 'confidence_chunk_size', 2048)
        except AttributeError:
            confidence_chunk_size = min(1024, base_micro_batch_size * 64)
        
        print(f"RLVCR [OPTIMIZED]: Computing {total_computations} confidence calculations, chunk_size={confidence_chunk_size}")
        print(f"RLVCR [OPTIMIZED]: {len(input_prefixes)} prefixes * {len(action_token_ids)} tokens = {total_computations} total calculations")
        
        if total_computations > confidence_chunk_size:
            # Process in chunks with GPU-friendly sizing
            gpu_count = getattr(actor_rollout_wg, 'world_size', 1)
            base_prefix_chunk_size = max(1, confidence_chunk_size // len(action_token_ids))
            
            # GPU alignment algorithm
            token_count = len(action_token_ids)
            gcd_val = math.gcd(token_count, gpu_count)
            lcm_tokens_gpus = (token_count * gpu_count) // gcd_val
            min_prefix_chunk_for_alignment = lcm_tokens_gpus // token_count
            
            # Apply efficiency and memory constraints
            max_reasonable_lcm = gpu_count * 16
            if lcm_tokens_gpus > max_reasonable_lcm:
                min_prefix_chunk_for_alignment = gpu_count
                print(f"RLVCR: Large LCM({token_count},{gpu_count})={lcm_tokens_gpus}, using fallback chunk_size={gpu_count}")
            
            # Choose final chunk size
            if base_prefix_chunk_size >= min_prefix_chunk_for_alignment:
                multiplier = base_prefix_chunk_size // min_prefix_chunk_for_alignment
                prefix_chunk_size = multiplier * min_prefix_chunk_for_alignment
                
                efficiency = prefix_chunk_size / base_prefix_chunk_size if base_prefix_chunk_size > 0 else 1
                if efficiency < 0.5:
                    prefix_chunk_size = base_prefix_chunk_size
                    print(f"RLVCR: Low efficiency ({efficiency:.1%}), using base chunk size with runtime padding")
            else:
                prefix_chunk_size = min_prefix_chunk_for_alignment
            
            if prefix_chunk_size != base_prefix_chunk_size:
                print(f"RLVCR: Adjusted chunk size from {base_prefix_chunk_size} to {prefix_chunk_size} prefixes for GPU alignment")
            
            all_results = []
            
            for i in range(0, len(input_prefixes), prefix_chunk_size):
                chunk_prefixes = input_prefixes[i:i + prefix_chunk_size]
                # 优化2：使用并行 token 处理
                chunk_results = self._compute_confidence_chunk_parallel(chunk_prefixes, action_token_ids, actor_rollout_wg)
                all_results.extend(chunk_results)
            
            return all_results
        else:
            # Process all at once
            return self._compute_confidence_chunk_parallel(input_prefixes, action_token_ids, actor_rollout_wg)

    def _compute_confidence_chunk_parallel(
        self,
        input_prefixes: List[str],
        action_token_ids: List[int],
        actor_rollout_wg
    ) -> List[float]:
        """
        优化版本：并行处理所有 tokens, 一次性计算整个 action 的 entropy。
        """
        if not input_prefixes:
            return []
        
        num_prefixes = len(input_prefixes)
        num_tokens = len(action_token_ids)
        
        # 准备批处理数据
        all_inputs = []
        
        for prefix in input_prefixes:
            # Tokenize prefix
            prefix_tokens = self.tokenizer(prefix, return_tensors='pt', padding=False, truncation=True)['input_ids'][0]
            
            # 创建完整序列: prefix + all action tokens
            full_input = torch.cat([prefix_tokens, torch.tensor(action_token_ids)])
            
            all_inputs.append({
                'input_ids': full_input,
                'attention_mask': torch.ones_like(full_input),
                'prefix_len': len(prefix_tokens)
            })
        
        # Pad all sequences to same length
        max_len = max(len(inp['input_ids']) for inp in all_inputs)
        for inp in all_inputs:
            pad_len = max_len - len(inp['input_ids'])
            if pad_len > 0:
                inp['input_ids'] = torch.cat([torch.full((pad_len,), self.tokenizer.pad_token_id), inp['input_ids']])
                inp['attention_mask'] = torch.cat([torch.zeros(pad_len), inp['attention_mask']])
        
        # GPU alignment
        original_batch_size = len(all_inputs)
        if hasattr(actor_rollout_wg, 'world_size') and actor_rollout_wg.world_size > 1:
            gpu_count = actor_rollout_wg.world_size
            if original_batch_size % gpu_count != 0:
                pad_size = gpu_count - (original_batch_size % gpu_count)
                for _ in range(pad_size):
                    all_inputs.append(all_inputs[-1])
        
        # Create batch
        batch_input_ids = torch.stack([inp['input_ids'] for inp in all_inputs])
        batch_attention_mask = torch.stack([inp['attention_mask'] for inp in all_inputs])
        batch_position_ids = compute_position_id_with_mask(batch_attention_mask)
        
        # 创建 responses：每个样本的 response 是整个 action sequence
        batch_responses = []
        for i, inp in enumerate(all_inputs):
            prefix_len = inp['prefix_len']
            response_positions = torch.arange(prefix_len, prefix_len + num_tokens)
            batch_responses.append(batch_input_ids[i][response_positions])
        
        batch_responses = torch.stack(batch_responses)
        
        batch = DataProto.from_dict({
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'position_ids': batch_position_ids,
            'responses': batch_responses,
        })
        
        # 一次性计算所有 tokens 的 entropy
        # output = self._handle_multi_gpu_entropy(batch, actor_rollout_wg)
        # all_entropies = output.batch['entropies']
        output = self._handle_multi_gpu_log_prob(batch, actor_rollout_wg)
        all_entropies = output.batch['old_log_probs']

        # 处理结果
        results = []
        for prefix_idx in range(num_prefixes):
            prefix_entropies = all_entropies[prefix_idx].cpu().numpy()
            mean_entropy = float(np.mean(prefix_entropies))
            # mean_entropy = float(np.max(prefix_entropies))  # use min entropy as confidence
            # use mean entropy
            # results.append(mean_entropy)
            # use mean probs as confidence
            confidence = np.exp(mean_entropy)
            results.append(confidence)
        
        return results

    def _rlvcr_collate_fn(self, data_list: list[dict]) -> dict:
        """RLVCR 专用的数据整理函数（处理变长响应）。"""
        tensors = defaultdict(list)
        non_tensors = defaultdict(list)

        for data in data_list:
            for key, val in data.items():
                if isinstance(val, torch.Tensor):
                    tensors[key].append(val)
                else:
                    non_tensors[key].append(val)

        # Handle tensors with potential padding
        for key, val in tensors.items():
            if len(val) > 1:
                shapes = [t.shape for t in val]
                if not all(shape == shapes[0] for shape in shapes) and len(shapes[0]) == 1:
                    # Pad 1D tensors (like responses) to same length
                    max_len = max(t.shape[0] for t in val)
                    padded_tensors = []
                    for t in val:
                        if t.shape[0] < max_len:
                            padding = torch.zeros(max_len - t.shape[0], dtype=t.dtype)
                            padded_t = torch.cat([t, padding], dim=0)
                        else:
                            padded_t = t
                        padded_tensors.append(padded_t)
                    tensors[key] = torch.stack(padded_tensors, dim=0)
                else:
                    tensors[key] = torch.stack(val, dim=0)
            else:
                tensors[key] = torch.stack(val, dim=0)

        for key, val in non_tensors.items():
            non_tensors[key] = np.array(val, dtype=object)

        return {**tensors, **non_tensors}

    # ========== GPU 处理辅助函数 ==========
    def _handle_multi_gpu_generation(self, batch: DataProto, actor_rollout_wg):
        """处理多 GPU 生成。"""
        return actor_rollout_wg.generate_sequences(batch)
    
    def _handle_multi_gpu_log_prob(self, batch: DataProto, actor_rollout_wg):
        """处理多 GPU 对数概率计算。"""
        return actor_rollout_wg.compute_log_prob(batch)
    
    '''
    def _handle_multi_gpu_entropy(self, batch: DataProto, actor_rollout_wg):
        """处理多 GPU 熵计算。"""
        return actor_rollout_wg.compute_entropy(batch)
    '''
    
