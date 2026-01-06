
import numpy as np
import torch
from collections import defaultdict, Counter
from verl import DataProto
import re

def compute_rlvmr_outcome_advantage(
    token_level_rewards: torch.Tensor,
    rlvmr_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: np.array,
    epsilon: float = 1e-6,
    step_advantage_w: float = 1.0,
    mode: str = "mean_norm",
    tag_types: np.array = None,
    traj_id: np.array = None
):
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # 默认tag类型为'none'，如果未提供tag_types
    if tag_types is None:
        tag_types = np.array(['none'] * len(index))
    
    # 计算episode-level group reward (基于uid分组)
    episode_advantages = episode_norm_reward(token_level_rewards, eos_mask, index, epsilon, remove_std)
    
    # 计算step-level group reward (基于uid和tag_type分组)
    step_advantages = step_norm_reward_by_tag(
        step_rewards=rlvmr_rewards,
        eos_mask=eos_mask,
        index=index,
        tag_types=tag_types,
        epsilon=epsilon,
        remove_std=remove_std,
        traj_id=traj_id
    )
    
    # 组合优势值
    scores = episode_advantages + step_advantage_w * step_advantages
    return scores, scores, {
        "episode_advantages": episode_advantages,
        "step_advantages": step_advantages,
    }


def episode_norm_reward(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: np.array,
    epsilon: float = 1e-6,
    remove_std: bool = True,
):
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return episode_advantages


def process_trajectory_rlvmr_rewards(
    trajectory_list,
    config,
    episode_rewards
):
    planning_reward_values = []
    exploration_reward_values = []
    reflection_reward_values = []

    # Process each trajectory
    for traj_idx, trajectory in enumerate(trajectory_list):
        is_success = episode_rewards[traj_idx] > 0

        planning_count = 0
        exploration_count = 0
        reflection_count = 0
        total_step = 0

        for step_idx, step_data in enumerate(trajectory):
            # Skip if inactive
            if not step_data['active_masks']:
                continue

            # Extract action data
            full_output = step_data["full_output"]
            is_valid = step_data["is_action_valid"]

            # Initialize rlvmr tag and reward
            step_data['rlvmr_tag_type'] = "none"
            step_data['rlvmr_step_reward'] = torch.tensor(0.0)

            if not is_valid:
                continue

            # Process planning tag
            if '<planning>' in full_output:
                step_data["rlvmr_tag_type"] = "<planning>"
                planning_count += 1
                if is_success:
                    step_data['rlvmr_step_reward'] = torch.tensor(config.algorithm.mcrl.planning_reward)
                    planning_reward_values.append(config.algorithm.mcrl.planning_reward)
                else:
                    step_data['rlvmr_step_reward'] = torch.tensor(0.0)
                    planning_reward_values.append(0)

            # Process exploration tag
            elif '<explore>' in full_output:
                is_repeated = False
                for prev_step_idx in range(step_idx):
                    prev_output = trajectory[prev_step_idx]["full_output"]
                    prev_action = re.findall(r'<action>(.*?)</action>', prev_output)
                    prev_valid = trajectory[prev_step_idx]["is_action_valid"]
                    current_action = re.findall(r'<action>(.*?)</action>', full_output)
                    if prev_action == current_action and prev_valid:
                        is_repeated = True
                        break

                step_data["rlvmr_tag_type"] = "<explore>"
                exploration_count += 1
                if not is_repeated:
                    explore_reward = config.algorithm.mcrl.exploration_reward
                    step_data['rlvmr_step_reward'] = torch.tensor(explore_reward)
                    exploration_reward_values.append(explore_reward)
                else:
                    step_data['rlvmr_step_reward'] = torch.tensor(0.0)
                    exploration_reward_values.append(0)

            # Process reflection tag
            elif '<reflection>' in full_output:
                step_data["rlvmr_tag_type"] = "<reflection>"
                reflection_count += 1

                current_action = re.findall(r'<action>(.*?)</action>', full_output)
                if step_idx > 0:
                    prev_step = trajectory[step_idx-1]
                    prev_valid = prev_step["is_action_valid"] and prev_step['action_available']
                    prev_output = prev_step["full_output"]
                    prev_action = re.findall(r'<action>(.*?)</action>', prev_output)    
                else:
                    prev_action = ""
                    prev_valid = False
                
                if not prev_valid and not (prev_action == current_action):
                    reflection_reward = config.algorithm.mcrl.reflection_reward
                    step_data['rlvmr_step_reward'] = torch.tensor(reflection_reward)
                    reflection_reward_values.append(reflection_reward)
                else:
                    step_data['rlvmr_step_reward'] = torch.tensor(0.0)
                    reflection_reward_values.append(0)

            total_step += 1

    return trajectory_list

def extract_rlvmr_rewards_from_batch(batch_data, device=None):
    """
    Extract rlvmr rewards and tag types from batch data.
    
    Args:
        batch_data: DataProto
            Batch data containing rlvmr information
        device: torch.device
            Device to place tensors on
    
    Returns:
        Tuple[torch.Tensor, np.array]
            rlvmr rewards tensor and tag types array
    """
    if device is None:
        device = torch.device('cpu')

    step_rewards_tensor = batch_data.batch["rlvmr_rewards"]
    tag_types = batch_data.non_tensor_batch['rlvmr_tag_types']

    return step_rewards_tensor, tag_types

def step_norm_reward_by_tag(
    step_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: np.array,
    tag_types: np.array,
    epsilon: float = 1e-6,
    remove_std: bool = True,
    traj_id: np.array = None
):
    """
    Normalize step rewards by grouping according to UID and tag type.
    
    Args:
        step_rewards: torch.Tensor
            Step rewards, shape (bs,)
        eos_mask: torch.Tensor
            End-of-sequence mask, shape (bs, response_length)
        index: np.array
            Sample UID indices
        tag_types: np.array
            Sample tag types
        epsilon: float
            Small value to prevent division by zero
        remove_std: bool
            Whether to remove standard deviation in normalization
    
    Returns:
        torch.Tensor
            Normalized step advantages, shape (bs, response_length)
    """
    response_length = eos_mask.shape[-1]
    scores = step_rewards.clone()
    # Create group keys for each (uid, tag_type) combination
    group_keys = []
    for i in range(len(index)):
        group_key = (index[i], tag_types[i])
        group_keys.append(group_key)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            group_key = group_keys[i]
            id2score[group_key].append(scores[i])

        for group_key in id2score:
            if len(id2score[group_key]) == 1:
                id2mean[group_key] = torch.tensor(0.0)
                id2std[group_key] = torch.tensor(1.0)
            elif len(id2score[group_key]) > 1:
                id2mean[group_key] = torch.mean(torch.tensor(id2score[group_key]))
                id2std[group_key] = torch.std(torch.tensor(id2score[group_key]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[group_key]): {len(id2score[group_key])}")
                raise ValueError(f"no score in group: {group_key}")

        for i in range(bsz):
            group_key = group_keys[i]
            if remove_std:
                scores[i] = scores[i] - id2mean[group_key]
            else: 
                scores[i] = (scores[i] - id2mean[group_key]) / (id2std[group_key] + epsilon)

        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return step_advantages 