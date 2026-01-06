# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics related to the PPO trainer.
"""
import pdb
import torch
from typing import Any, Dict, List
import numpy as np
from verl import DataProto


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
        # episode
        'episode/reward/mean': 
            batch.non_tensor_batch['episode_rewards_mean'][0].item(),
        'episode/reward/max': 
            batch.non_tensor_batch['episode_rewards_max'][0].item(),
        'episode/reward/min': 
            batch.non_tensor_batch['episode_rewards_min'][0].item(),
        'episode/length/mean': 
            batch.non_tensor_batch['episode_lengths_mean'][0].item(),
        'episode/length/max':
            batch.non_tensor_batch['episode_lengths_max'][0].item(),
        'episode/length/min': 
            batch.non_tensor_batch['episode_lengths_min'][0].item(),
        **({f'episode/{k}': v[0].item() for k, v in batch.non_tensor_batch.items() if 'success_rate' in k}),
    }
    
    # Add RLVCR-specific metrics if available
    if 'rlvcr_thinking_entropies' in batch.meta_info:
        thinking_entropies = batch.meta_info['rlvcr_thinking_entropies']
        thinking_costs = batch.meta_info['rlvcr_thinking_costs']
        thinking_group_ids = batch.meta_info['rlvcr_thinking_group_ids']
        is_original = batch.meta_info['rlvcr_is_original']
        
        # Basic RLVCR metrics
        metrics.update({
            'rlvcr/entropy_mean': np.mean(thinking_entropies),
            'rlvcr/cost_mean': np.mean(thinking_costs),
        })
        
        # Add weight annealing metrics if available
        if 'rlvcr_weight_metrics' in batch.meta_info:
            weight_metrics = batch.meta_info['rlvcr_weight_metrics']
            metrics.update(weight_metrics)
        # Thinking level distribution (based on original samples only)
        if 'original_level' in batch.non_tensor_batch:
            original_levels = batch.non_tensor_batch['original_level']
            current_levels = batch.non_tensor_batch.get('thinking_levels', 
                                                      batch.non_tensor_batch.get('current_level', None))
            
            if current_levels is not None and original_levels is not None:
                # Only count samples where current_level == original_level (true original samples)
                original_sample_mask = (current_levels == original_levels)
                if np.any(original_sample_mask):
                    true_original_levels = original_levels[original_sample_mask]
                    
                    # 分离失败样本(-1)和成功样本(1,2,3,4)
                    failed_mask = (true_original_levels == -1)
                    success_mask = (true_original_levels > 0)
                    
                    # 计算失败样本比例
                    failed_ratio = np.mean(failed_mask) if len(true_original_levels) > 0 else 0.0
                    
                    # 只对成功样本计算level比例
                    if np.any(success_mask):
                        success_levels = true_original_levels[success_mask]
                        # 在成功样本中的各level比例
                        level_1_ratio = np.mean(success_levels == 1)
                        level_2_ratio = np.mean(success_levels == 2)
                        level_3_ratio = np.mean(success_levels == 3)
                        level_4_ratio = np.mean(success_levels == 4)
                        avg_thinking_level = np.mean(success_levels)
                    else:
                        # 如果没有成功样本，所有level比例为0
                        level_1_ratio = level_2_ratio = level_3_ratio = level_4_ratio = 0.0
                        avg_thinking_level = 0.0
                    
                    metrics.update({
                        'rlvcr/level_1_ratio': level_1_ratio,
                        'rlvcr/level_2_ratio': level_2_ratio,
                        'rlvcr/level_3_ratio': level_3_ratio,
                        'rlvcr/level_4_ratio': level_4_ratio,
                        'rlvcr/avg_thinking_level': avg_thinking_level,
                        'rlvcr/failed_samples_ratio': failed_ratio,  # 新增：失败样本比例
                        'rlvcr/success_samples_count': np.sum(success_mask),  # 新增：成功样本数量
                        'rlvcr/failed_samples_count': np.sum(failed_mask),    # 新增：失败样本数量
                    })

        
        # Success rate only
        sequence_reward = batch.batch['token_level_rewards'].sum(-1).cpu().numpy()
        successful_mask = sequence_reward > 0
        metrics.update({
            'rlvcr/success_rate': np.mean(successful_mask),
        })
        
        # Simplified advantage analysis
        final_advantages = batch.batch['advantages'].cpu().numpy()
        response_mask = batch.batch['response_mask'].cpu().numpy()
        
        # Get valid advantages (only response tokens)
        valid_final_adv = final_advantages[response_mask]
        
        # Simplified advantage decomposition per thinking group
        unique_groups = np.unique(thinking_group_ids[thinking_group_ids >= 0])
        
        action_advantages_list = []
        thinking_advantages_list = []
        
        if len(unique_groups) > 0:
            for group_id in unique_groups:
                group_mask = (thinking_group_ids == group_id)
                if np.sum(group_mask) == 4:  # Complete group
                    group_indices = np.where(group_mask)[0]
                    group_final_advs = []
                    
                    # Collect advantages for each sample in the group
                    for idx in group_indices:
                        # Get advantage at the last valid position
                        last_valid_pos = np.sum(response_mask[idx]) - 1
                        if last_valid_pos >= 0:
                            adv_value = final_advantages[idx, last_valid_pos]
                            group_final_advs.append(adv_value)
                    
                    if len(group_final_advs) == 4:
                        # Approximate action advantage (mean of group)
                        action_adv = np.mean(group_final_advs)
                        action_advantages_list.append(action_adv)
                        
                        # Thinking advantages (deviations from group mean)
                        thinking_advs = np.array(group_final_advs) - action_adv
                        thinking_advantages_list.extend(thinking_advs)
        
        # Core advantage metrics only
        advantage_metrics = {
            'rlvcr/final_adv_mean': np.mean(valid_final_adv),
        }
        
        # Action and thinking advantage decomposition
        if len(action_advantages_list) > 0:
            advantage_metrics.update({
                'rlvcr/action_adv_mean': np.mean(action_advantages_list),
                'rlvcr/action_adv_min': np.min(action_advantages_list),
                'rlvcr/action_adv_max': np.max(action_advantages_list),
                'rlvcr/action_adv_std': np.std(action_advantages_list),
            })
        
        if len(thinking_advantages_list) > 0:
            advantage_metrics.update({
                'rlvcr/thinking_adv_mean': np.mean(thinking_advantages_list),
                'rlvcr/thinking_adv_min': np.min(thinking_advantages_list),
                'rlvcr/thinking_adv_max': np.max(thinking_advantages_list),
                'rlvcr/thinking_adv_std': np.std(thinking_advantages_list),
            })
        
        metrics.update(advantage_metrics)
    
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info['global_token_num'])
    time = timing_raw['step']
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        'perf/total_num_tokens': total_num_tokens,
        'perf/time_per_step': time,
        'perf/throughput': total_num_tokens / (time * n_gpus),
    }
