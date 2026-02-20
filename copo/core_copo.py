"""
CoPo (RL with Variational Chain Reasoning) core algorithms.

This module implements the CoPo advantage computation method that combines:
1. Trajectory-level advantages (similar to existing methods)
2. Thinking-level advantages based on thinking diversity and token efficiency
"""

import torch
import numpy as np
from typing import Dict, Tuple, Any
from collections import defaultdict
import re
import json
import os
from datetime import datetime
from .prompt import THINK_MODE_2, THINK_MODE_3, THINK_MODE_4
import pdb

def json_serializer(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def compute_format_penalty(responses: list, thinking_levels: np.array = None) -> np.array:
    """
    Compute the format penalty for each response.
    
    Args:
        responses: List of response strings
        thinking_levels: Array of thinking levels for each response
    
    Returns:
        np.array: Format penalties (0.0 for correct format, -20.0 for incorrect)
    """
    if responses is None or len(responses) == 0:
        return np.array([])
    
    format_penalties = []
    
    for i, response in enumerate(responses):
        if not response:
            format_penalties.append(-20.0)
            continue
            
        # Check for required tags
        has_level = bool(re.search(r'<level>\d+</level>', response))
        has_action = bool(re.search(r'<action>.*?</action>', response, re.DOTALL))

        # Check for think tag for all levels (including level 1)
        has_think = bool(re.search(r'<think>.*?</think>', response, re.DOTALL))
            
        # All conditions must be met
        is_format_valid = has_level and has_action and has_think
        format_penalties.append(0.0 if is_format_valid else -20.0)
    
    return np.array(format_penalties)

def compute_copo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: np.array,
    thinking_entropies: np.array,
    thinking_costs: np.array,
    thinking_group_ids: np.array,
    is_original: np.array,
    thinking_weight: float = 1.0,
    thinking_cost_alpha: float = 1.0,
    cost_max: int = 200,
    thinking_levels: np.array = None,
    responses: list = None,
    current_step: int = 0,
    total_steps: int = 1000,
    ada_grpo_enabled: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Compute CoPo outcome advantages with dual advantage calculation.
    
    Args:
        token_level_rewards: Episode-level token rewards
        eos_mask: End-of-sequence mask  
        index: Group indices for trajectory grouping
        thinking_entropies: Action confidence scores (min_prob)
        thinking_costs: Token costs for thinking
        thinking_group_ids: Group IDs for thinking samples
        is_original: Boolean array marking original samples
        original_indices: List of indices for original samples
        thinking_weight: Beta parameter for thinking advantage weight
        thinking_cost_alpha: Alpha parameter for cost penalty
        cost_max: Maximum cost for normalization
    
    Returns:
        advantages: Combined advantages
        returns: Same as advantages for policy gradient
        metrics: Dictionary of metrics for logging
    """
    batch_size = token_level_rewards.shape[0]
    
    print(f" CoPo Advantage Computation Debug:")
    print(f"    batch_size: {batch_size}")
    print(f"    thinking_group_ids: {thinking_group_ids}")
    print(f"    is_original: {is_original}")
    
    # Step 1: Apply format penalties and Ada-GRPO scaling to rewards
    # This is done BEFORE computing any advantages
    scaled_token_level_rewards = token_level_rewards.clone()
    
    # 1.1 Compute format penalties for all samples
    '''
    if responses is not None:
        format_penalties = compute_format_penalty(responses, thinking_levels)
        print(f"    Format penalties computed: {np.sum(format_penalties == 0.0)}/{len(format_penalties)} valid formats")
        
        # Apply format penalties with fine-grained strategy:
        # - If original reward > 0 and format is invalid: set to 0
        # - If original reward <= 0 and format is invalid: keep original (no extra penalty)
        format_penalty_tensor = torch.tensor(format_penalties, dtype=scaled_token_level_rewards.dtype, device=scaled_token_level_rewards.device)
        
        # Create a mask for invalid formats (where penalty is -20.0)
        invalid_format_mask = (format_penalty_tensor == -20.0).unsqueeze(1).expand_as(scaled_token_level_rewards)
        
        # Create a mask for positive rewards
        positive_reward_mask = scaled_token_level_rewards > 0
        
        # Only penalize samples with both invalid format AND positive reward
        penalize_mask = invalid_format_mask & positive_reward_mask
        
        # Set to 0 for samples that should be penalized, keep original otherwise
        scaled_token_level_rewards = torch.where(penalize_mask, 
                                                torch.zeros_like(scaled_token_level_rewards),
                                                scaled_token_level_rewards)
        
        # Count how many samples were penalized
        num_penalized = penalize_mask.any(dim=1).sum().item()
        num_invalid = invalid_format_mask.any(dim=1).sum().item()
        print(f"    Applied format penalties: {num_penalized}/{num_invalid} invalid formats with positive rewards set to 0")
    else:
        print(f"\n  Step 1.1: Format Penalties - SKIPPED (no responses provided)")
    '''
    # 1.2 Apply Ada-GRPO diversity scaling for original samples only
    thinking_level_scaling_factors = {}  # (uid, level) -> alpha_i
    if ada_grpo_enabled and thinking_levels is not None and is_original is not None:
        print(f"\n  Step 1.2: Ada-GRPO Diversity Scaling (step {current_step}/{total_steps}):")
        # current_step = min(total_steps, current_step+100)

        # Only process original samples with valid levels (exclude -1)
        original_indices = []
        for i in range(batch_size):
            if (i < len(is_original) and is_original[i] and 
                i < len(thinking_levels) and thinking_levels[i] != -1):
                original_indices.append(i)
        
        print(f"    Found {len(original_indices)} valid original samples (excluding Level -1)")
        
        if original_indices:
            # Extract data for original samples
            original_uids = [index[i] for i in original_indices]
            original_levels = [thinking_levels[i] for i in original_indices]
            
            # Count format frequencies by uid
            uid_to_indices = defaultdict(list)
            for idx, orig_idx in enumerate(original_indices):
                uid_to_indices[original_uids[idx]].append(idx)
            
            # Calculate scaling factors for each original sample
            scaling_factors = {}  # original_idx -> alpha_i
            
            for uid, idx_list in uid_to_indices.items():
                # Count level frequencies within this uid group
                level_counts = defaultdict(int)
                for idx in idx_list:
                    level_counts[original_levels[idx]] += 1
                
                G = len(idx_list)  # Total samples in this uid group
                
                # Calculate scaling for each sample in this uid group
                for idx in idx_list:
                    orig_idx = original_indices[idx]
                    level = original_levels[idx]
                    F_oi = level_counts[level]
                    
                    # Compute decay factor
                    t_ratio = current_step / max(total_steps, 1)
                    decay_i = (F_oi / G) + 0.5 * (1 - F_oi / G) * (1 + np.cos(np.pi * t_ratio))
                    
                    # Compute diversity scaling factor
                    alpha_i = (G / F_oi) * decay_i
                    scaling_factors[orig_idx] = alpha_i
                    
                    # Also store in thinking_level_scaling_factors for later use
                    thinking_level_scaling_factors[(uid, level)] = alpha_i
                    
                    print(f"        Sample {orig_idx}: uid={uid}, level={level}, G={G}, F_oi={F_oi}, alpha={alpha_i:.3f}")
            
            # Apply scaling to original samples' rewards only
            # for orig_idx, alpha_i in scaling_factors.items():
                # scaled_token_level_rewards[orig_idx] *= alpha_i
            
            # print(f"    Applied Ada-GRPO scaling to {len(scaling_factors)} original samples")

    # Step 2: Compute action advantages for original samples, then propagate to thinking groups
    action_advantages = torch.zeros_like(scaled_token_level_rewards)
    
    # Find all original sample indices
    actual_original_indices = []
    for i in range(batch_size):
        if i < len(is_original) and is_original[i]:
            actual_original_indices.append(i)
    
    print(f"\nStep 2: Computing action advantages:")
    print(f"    Found {len(actual_original_indices)} original samples")
    
    if actual_original_indices:
        # Extract original samples for advantage computation
        original_rewards = scaled_token_level_rewards[actual_original_indices]
        original_mask = eos_mask[actual_original_indices]
        original_uid = [index[i] for i in actual_original_indices]
        
        # Compute advantages for original samples using GRPO (inline implementation)
        response_length = original_rewards.shape[-1]
        scores = original_rewards.sum(dim=-1)
        
        id2score = defaultdict(list)
        id2mean = {}
        id2std = {}
        epsilon = 1e-6
        
        with torch.no_grad():
            bsz = scores.shape[0]
            for i in range(bsz):
                id2score[original_uid[i]].append(scores[i])
            for idx in id2score:
                if len(id2score[idx]) == 1:
                    print(f"        WARNING: Prompt {idx} has only 1 sample, will get 0 advantage")
                    id2mean[idx] = torch.tensor(0.0, device=scores.device)
                    id2std[idx] = torch.tensor(1.0, device=scores.device)
                elif len(id2score[idx]) > 1:
                    score_list = torch.stack(id2score[idx])
                    if torch.max(score_list) - torch.min(score_list) < 1e-6:
                        id2mean[idx] = torch.tensor(score_list[0], device=scores.device)
                        id2std[idx] = torch.tensor(1.0, device=scores.device)
                    else:
                        id2mean[idx] = torch.mean(score_list)
                        id2std[idx] = torch.std(score_list)
                else:
                    raise ValueError(f"no score in prompt index: {idx}")
            for i in range(bsz):
                scores[i] = (scores[i] - id2mean[original_uid[i]]) / (id2std[original_uid[i]] + epsilon)
            orig_adv = scores.unsqueeze(-1).expand(-1, response_length) * original_mask
        
        # Assign computed advantages to original samples
        for i, orig_idx in enumerate(actual_original_indices):
            action_advantages[orig_idx] = orig_adv[i]
        print(f"    Computed action advantages for {len(actual_original_indices)} original samples")
        
        # Build a mapping from group_id to original sample index for efficient copying
        print(f"    Building group -> original sample mapping...")
        group_to_original = {}
        for idx in range(batch_size):
            if idx < len(thinking_group_ids) and thinking_group_ids[idx] >= 0 and idx < len(is_original) and is_original[idx]:
                group_id = thinking_group_ids[idx]
                if group_id in group_to_original:
                    print(f"    WARNING: Group {group_id} has multiple original samples!")
                group_to_original[group_id] = idx
        
        print(f"    Found {len(group_to_original)} groups with original samples")
        
        # Copy advantages to all samples in the same group
        print(f"    Copying action advantages to expanded samples...")
        for idx in range(batch_size):
            if idx < len(thinking_group_ids):
                group_id = thinking_group_ids[idx]
                if group_id >= 0 and not is_original[idx]:
                    # This is an expanded sample, find its original
                    if group_id in group_to_original:
                        orig_idx = group_to_original[group_id]
                        action_advantages[idx] = action_advantages[orig_idx]
                    else:
                        print(f"    WARNING: Sample {idx} in group {group_id} has no original sample!")
    
    # Step 3: Compute thinking weights using GigPO-style approach on probs
    # Instead of computing thinking advantages, we'll compute weights to scale action advantages
    print(f"\nStep 3: Computing thinking weights using GigPO-style approach with cosine annealing")
    
    # Compute annealing progress for weight application
    progress = min(1.0, current_step / max(1, total_steps))
    print(f"    Annealing progress: {progress:.3f} (step {current_step}/{total_steps})")
    print(f"    Weights will be annealed from 1.0 → target_weight using cosine schedule")
    
    # Temperature parameter for the exponential weighting (similar to k in GigPO)
    k = 2.0  # You can make this configurable later
    
    # Initialize metrics collection
    weight_metrics = {
        'target_weights': [],
        'annealed_weights': [],
        'thinking_probs': [],
    }
    
    # Process each thinking group
    unique_groups = np.unique(thinking_group_ids[thinking_group_ids >= 0])
    print(f"    Processing {len(unique_groups)} thinking groups: {unique_groups}")
    
    for group_id in unique_groups:
        group_mask = (thinking_group_ids == group_id)
        group_indices = np.where(group_mask)[0]
        print(f"\n    Group {group_id}: indices = {group_indices}")
        
        if len(group_indices) == 4:  # Complete thinking group
            # Extract probability values for this group (thinking_entropies contains probs)
            group_probs = np.array([thinking_entropies[idx] for idx in group_indices])
            
            # Clamp probability values to reasonable range [1e-6, 1.0]
            group_probs = np.clip(group_probs, 1e-6, 1.0)
            
            # Step 1: Min-max normalization within the group
            prob_min = np.min(group_probs)
            prob_max = np.max(group_probs)
            prob_range = prob_max - prob_min
            
            if prob_range > 1e-6:
                # Normalize probs to [0, 1] range
                group_probs_norm = (group_probs - prob_min) / prob_range
            else:
                # All probs are the same
                group_probs_norm = np.ones_like(group_probs) * 0.5
            
            print(f"        Prob normalization: min={prob_min:.4f}, max={prob_max:.4f}, range={prob_range:.4f}")
            print(f"        Normalized probs: {[f'{p:.4f}' for p in group_probs_norm]}")
            
            # Step 2: Compute g(prob) weights using exponential function
            # Since higher prob is better, we use exp(k * prob_norm) instead of exp(-k * entropy_norm)
            exp_values = np.exp(k * group_probs_norm)
            
            # Normalize to ensure mean weight = 1 within the group
            mean_exp = np.mean(exp_values)
            group_weights = exp_values / mean_exp
            
            print(f"        Exponential values: {[f'{e:.4f}' for e in exp_values]}")
            print(f"        Normalized weights (mean=1): {[f'{w:.4f}' for w in group_weights]}")
            
            # Note: Ada-GRPO scaling already applied to rewards in Step 1.2, 
            # so action advantages already incorporate the diversity scaling
            
            # Step 3: Apply weights to action advantages with cosine annealing
            for i, sample_idx in enumerate(group_indices):
                target_weight = group_weights[i]  # The computed weight we want to reach
                actual_level = thinking_levels[sample_idx] if thinking_levels is not None else (i + 1)
                
                # Apply cosine annealing from 1.0 (initial) to target_weight (final)
                # Using cosine cool-down: starts at 1.0, gradually decreases to target_weight
                # Formula: weight = target + (1.0 - target) * 0.5 * (1 + cos(π * progress))
                # annealed_weight = target_weight + (1.0 - target_weight) * 0.5 * (1.0 + np.cos(np.pi * progress))
                annealed_weight = target_weight
                
                # Collect metrics for this sample
                weight_metrics['target_weights'].append(float(target_weight))
                weight_metrics['annealed_weights'].append(float(annealed_weight))
                weight_metrics['thinking_probs'].append(float(thinking_entropies[sample_idx]))
                
                # Store original action advantage for debug print
                orig_action_adv = action_advantages[sample_idx].clone()
                
                # Apply annealed weight to the action advantages
                action_advantages[sample_idx] = action_advantages[sample_idx] * annealed_weight
                
                # Debug print
                orig_adv_mean = orig_action_adv.mean().item()
                weighted_adv_mean = action_advantages[sample_idx].mean().item()

                # 条件日志记录
                if orig_adv_mean < 0:
                    # 创建日志数据
                    log_data = {
                        "timestamp": datetime.now().isoformat(),
                        "sample_idx": sample_idx,
                        "level": actual_level,
                        "target_weight": float(target_weight),
                        "annealed_weight": float(annealed_weight),
                        "progress": float(progress),
                        "current_step": int(current_step),
                        "total_steps": int(total_steps),
                        "orig_adv_mean": orig_adv_mean,
                        "weighted_adv_mean": weighted_adv_mean,
                        "id2mean": {k: v.item() if hasattr(v, 'item') else v for k, v in id2mean.items()},
                        "id2std": {k: v.item() if hasattr(v, 'item') else v for k, v in id2std.items()},
                        "id2score": {k: [s.item() if hasattr(s, 'item') else s for s in v] for k, v in id2score.items()},
                        "index": index,
                        'is_original': is_original,
                        "thinking_level": thinking_levels,
                        "sample_idx_rewards": [scaled_token_level_rewards[i].sum().item() if i < len(scaled_token_level_rewards) else 0.0 for i in range(len(scaled_token_level_rewards))]
                    }
                    
                    # 保存到日志文件
                    log_dir = "/apdcephfs_cq11/share_1567347/share_info/rhyang/RLVMR/data/valid/run_wofd_20250922"
                    os.makedirs(log_dir, exist_ok=True)
                    log_file = os.path.join(log_dir, f"negative_adv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    
                    with open(log_file, 'w') as f:
                        json.dump(log_data, f, indent=2, default=json_serializer)
                    
                    print(f"        [NEGATIVE ADV LOGGED] Sample {sample_idx}: Level {actual_level}, prob={thinking_entropies[sample_idx]:.4f}, "
                          f"target_weight={target_weight:.4f}, annealed_weight={annealed_weight:.4f} (progress={progress:.3f}), "
                          f"orig_adv={orig_adv_mean:.4f}, weighted_adv={weighted_adv_mean:.4f}")
                    print(f"        Log saved to: {log_file}")
                else:
                    print(f"        Sample {sample_idx}: Level {actual_level}, prob={thinking_entropies[sample_idx]:.4f}, "
                          f"target_weight={target_weight:.4f}, annealed_weight={annealed_weight:.4f} (progress={progress:.3f}), "
                          f"orig_adv={orig_adv_mean:.4f}, weighted_adv={weighted_adv_mean:.4f}")
        else:
            print(f"    Group {group_id}: Incomplete group with {len(group_indices)} samples, skipping")
    
    # Step 4: Return weighted action advantages directly (no separate thinking advantages)
    advantages = action_advantages  # Already weighted by thinking confidence
    returns = advantages  # For PPO

    # Compute summary metrics for logging
    metrics = {
        'copo/weight_annealing_progress': float(progress),
    }
    
    if weight_metrics['target_weights']:
        metrics.update({
            'copo/target_weight_mean': float(np.mean(weight_metrics['target_weights'])),
            'copo/target_weight_max': float(np.max(weight_metrics['target_weights'])),
            'copo/target_weight_min': float(np.min(weight_metrics['target_weights'])),
            'copo/annealed_weight_mean': float(np.mean(weight_metrics['annealed_weights'])),
            'copo/annealed_weight_max': float(np.max(weight_metrics['annealed_weights'])),
            'copo/annealed_weight_min': float(np.min(weight_metrics['annealed_weights'])),
            'copo/thinking_prob_mean': float(np.mean(weight_metrics['thinking_probs'])),
            'copo/thinking_prob_max': float(np.max(weight_metrics['thinking_probs'])),
            'copo/thinking_prob_min': float(np.min(weight_metrics['thinking_probs'])),
        })

    return advantages, returns, metrics


def create_level_specific_prompt(history: str, target_level: int, action: str) -> str:
    """
    Create a prompt that forces the model to use a specific thinking level.
    
    Args:
        history: The conversation history up to this point
        target_level: Target thinking level (1-4)
        task_type: Type of task ("sciworld" or "alfworld")
        
    Returns:
        Modified prompt with level constraint
    """
    history = history.split("<action>your_next_action</action>")[-1].split("Now it's your turn to generate next step response.")[0].strip()
    if target_level == 2:
        instruction = THINK_MODE_2
    elif target_level == 3:
        instruction = THINK_MODE_3
    elif target_level == 4:
        instruction = THINK_MODE_4    
        
    return instruction.format(history=history, action=action)
