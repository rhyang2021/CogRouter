"""
RLVCR Metrics Logger for Weights & Biases
Tracks thinking metrics, case studies, and detailed statistics
"""

import numpy as np
import wandb
import torch
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json
import os


@dataclass
class ThinkingCase:
    """Single thinking case for analysis"""
    trajectory_id: str
    original_level: int
    action: str
    prompt: str
    thinking_texts: List[str]  # 4 levels
    entropies: List[float]     # 4 levels
    costs: List[int]           # 4 levels
    episode_reward: float
    step_idx: int
    epoch: int


class RLVCRMetricsLogger:
    """Enhanced logger for RLVCR training metrics"""
    
    def __init__(self, save_cases: bool = True, max_cases_per_epoch: int = 20):
        self.save_cases = save_cases
        self.max_cases_per_epoch = max_cases_per_epoch
        self.cases_buffer = []
        self.epoch_stats = {}
        
    def log_thinking_metrics(
        self,
        epoch: int,
        step: int,
        thinking_entropies: np.ndarray,
        thinking_costs: np.ndarray,
        thinking_group_ids: np.ndarray,
        is_original: np.ndarray,
        action_advantages: np.ndarray,
        thinking_advantages: np.ndarray,
        final_advantages: np.ndarray,
        episode_rewards: np.ndarray
    ):
        """Log detailed thinking metrics to wandb"""
        
        # Basic statistics
        metrics = {
            f"rlvcr/epoch": epoch,
            f"rlvcr/step": step,
            
            # Entropy metrics (action confidence)
            f"rlvcr/entropy_mean": np.mean(thinking_entropies),
            f"rlvcr/entropy_std": np.std(thinking_entropies),
            f"rlvcr/entropy_min": np.min(thinking_entropies),
            f"rlvcr/entropy_max": np.max(thinking_entropies),
            
            # Cost metrics (thinking tokens)
            f"rlvcr/cost_mean": np.mean(thinking_costs),
            f"rlvcr/cost_std": np.std(thinking_costs),
            f"rlvcr/cost_min": np.min(thinking_costs),
            f"rlvcr/cost_max": np.max(thinking_costs),
            
            # Advantage metrics
            f"rlvcr/action_adv_mean": np.mean(action_advantages),
            f"rlvcr/thinking_adv_mean": np.mean(thinking_advantages),
            f"rlvcr/final_adv_mean": np.mean(final_advantages),
            f"rlvcr/action_adv_std": np.std(action_advantages),
            f"rlvcr/thinking_adv_std": np.std(thinking_advantages),
            f"rlvcr/final_adv_std": np.std(final_advantages),
        }
        
        # Success vs failure analysis
        successful_mask = episode_rewards > 0
        if np.any(successful_mask):
            metrics.update({
                f"rlvcr/success_entropy_mean": np.mean(thinking_entropies[successful_mask]),
                f"rlvcr/success_cost_mean": np.mean(thinking_costs[successful_mask]),
                f"rlvcr/success_rate": np.mean(successful_mask),
            })
        
        # Thinking level distribution (only for original samples)
        original_mask = is_original == 1
        if np.any(original_mask):
            original_costs = thinking_costs[original_mask]
            # Estimate thinking levels based on cost
            level_1_mask = original_costs == 0
            level_2_mask = (original_costs > 0) & (original_costs <= 50)
            level_3_mask = (original_costs > 50) & (original_costs <= 150)
            level_4_mask = original_costs > 150
            
            metrics.update({
                f"rlvcr/level_1_ratio": np.mean(level_1_mask),
                f"rlvcr/level_2_ratio": np.mean(level_2_mask),
                f"rlvcr/level_3_ratio": np.mean(level_3_mask),
                f"rlvcr/level_4_ratio": np.mean(level_4_mask),
            })
        
        # Log to wandb
        wandb.log(metrics, step=step)
        
        # Create and log distribution plots
        self._log_distribution_plots(
            thinking_entropies, thinking_costs, episode_rewards, step
        )
    
    def _log_distribution_plots(
        self,
        entropies: np.ndarray,
        costs: np.ndarray,
        rewards: np.ndarray,
        step: int
    ):
        """Create and log distribution visualizations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'RLVCR Metrics Distribution (Step {step})', fontsize=16)
        
        # Entropy distribution
        axes[0, 0].hist(entropies, bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_title('Action Confidence (Entropy) Distribution')
        axes[0, 0].set_xlabel('Min Probability')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(entropies), color='red', linestyle='--', label='Mean')
        axes[0, 0].legend()
        
        # Cost distribution
        axes[0, 1].hist(costs, bins=30, alpha=0.7, color='green')
        axes[0, 1].set_title('Thinking Cost Distribution')
        axes[0, 1].set_xlabel('Token Count')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(costs), color='red', linestyle='--', label='Mean')
        axes[0, 1].legend()
        
        # Entropy vs Cost scatter
        success_mask = rewards > 0
        axes[1, 0].scatter(costs[success_mask], entropies[success_mask], 
                          alpha=0.6, color='green', label='Successful', s=20)
        axes[1, 0].scatter(costs[~success_mask], entropies[~success_mask], 
                          alpha=0.6, color='red', label='Failed', s=20)
        axes[1, 0].set_xlabel('Thinking Cost')
        axes[1, 0].set_ylabel('Action Confidence')
        axes[1, 0].set_title('Cost vs Confidence')
        axes[1, 0].legend()
        
        # Reward vs Cost
        axes[1, 1].scatter(costs, rewards, alpha=0.6, color='purple', s=20)
        axes[1, 1].set_xlabel('Thinking Cost')
        axes[1, 1].set_ylabel('Episode Reward')
        axes[1, 1].set_title('Cost vs Episode Reward')
        
        plt.tight_layout()
        
        # Log to wandb
        wandb.log({f"rlvcr/distributions": wandb.Image(fig)}, step=step)
        plt.close(fig)
    
    def save_thinking_case(
        self,
        trajectory_id: str,
        original_level: int,
        action: str,
        prompt: str,
        thinking_texts: List[str],
        entropies: List[float],
        costs: List[int],
        episode_reward: float,
        step_idx: int,
        epoch: int
    ):
        """Save an interesting thinking case for analysis"""
        
        if not self.save_cases:
            return
        
        case = ThinkingCase(
            trajectory_id=trajectory_id,
            original_level=original_level,
            action=action,
            prompt=prompt[-500:],  # Last 500 chars to avoid too long
            thinking_texts=thinking_texts,
            entropies=entropies,
            costs=costs,
            episode_reward=episode_reward,
            step_idx=step_idx,
            epoch=epoch
        )
        
        self.cases_buffer.append(case)
        
        # Keep only recent cases to avoid memory issues
        if len(self.cases_buffer) > self.max_cases_per_epoch * 5:
            self.cases_buffer = self.cases_buffer[-self.max_cases_per_epoch * 3:]
    
    def log_epoch_cases(self, epoch: int):
        """Log collected cases to wandb at epoch end"""
        
        if not self.cases_buffer:
            return
        
        # Select interesting cases
        current_epoch_cases = [c for c in self.cases_buffer if c.epoch == epoch]
        
        if not current_epoch_cases:
            return
        
        # Sort by different criteria and take top cases
        interesting_cases = []
        
        # High reward cases
        high_reward_cases = sorted(current_epoch_cases, 
                                 key=lambda x: x.episode_reward, reverse=True)[:5]
        interesting_cases.extend(high_reward_cases)
        
        # High entropy variance cases (diverse thinking)
        high_variance_cases = sorted(current_epoch_cases,
                                   key=lambda x: np.std(x.entropies), reverse=True)[:5]
        interesting_cases.extend(high_variance_cases)
        
        # Efficient thinking cases (low cost, high reward)
        efficient_cases = sorted(current_epoch_cases,
                               key=lambda x: x.episode_reward / (np.mean(x.costs) + 1), 
                               reverse=True)[:5]
        interesting_cases.extend(efficient_cases)
        
        # Remove duplicates
        seen_ids = set()
        unique_cases = []
        for case in interesting_cases:
            if case.trajectory_id not in seen_ids:
                unique_cases.append(case)
                seen_ids.add(case.trajectory_id)
        
        # Create case analysis table
        case_data = []
        for case in unique_cases[:self.max_cases_per_epoch]:
            case_data.append([
                case.trajectory_id,
                case.original_level,
                case.action[:50] + "..." if len(case.action) > 50 else case.action,
                f"{case.episode_reward:.2f}",
                f"{np.mean(case.entropies):.3f}",
                f"{np.std(case.entropies):.3f}",
                f"{np.mean(case.costs):.1f}",
                f"{np.std(case.costs):.1f}",
                case.step_idx
            ])
        
        # Log table to wandb
        table = wandb.Table(
            columns=["Traj_ID", "Orig_Level", "Action", "Reward", 
                    "Avg_Entropy", "Entropy_Std", "Avg_Cost", "Cost_Std", "Step"],
            data=case_data
        )
        
        wandb.log({f"rlvcr/epoch_{epoch}_cases": table})
        
        # Log detailed case examples
        for i, case in enumerate(unique_cases[:5]):  # Top 5 detailed cases
            case_detail = {
                f"rlvcr/case_{i}/prompt": case.prompt,
                f"rlvcr/case_{i}/action": case.action,
                f"rlvcr/case_{i}/original_level": case.original_level,
                f"rlvcr/case_{i}/reward": case.episode_reward,
            }
            
            # Add thinking details for each level
            for level in range(4):
                case_detail.update({
                    f"rlvcr/case_{i}/level_{level+1}_thinking": case.thinking_texts[level][:200] + "..." if len(case.thinking_texts[level]) > 200 else case.thinking_texts[level],
                    f"rlvcr/case_{i}/level_{level+1}_entropy": case.entropies[level],
                    f"rlvcr/case_{i}/level_{level+1}_cost": case.costs[level],
                })
            
            wandb.log(case_detail)
    
    def log_thinking_level_comparison(
        self,
        step: int,
        thinking_group_data: Dict[int, Dict[str, List]]
    ):
        """Log detailed comparison between thinking levels within groups"""
        
        if not thinking_group_data:
            return
        
        # Aggregate statistics across all groups
        level_stats = {level: {"entropies": [], "costs": [], "rewards": []} 
                      for level in range(1, 5)}
        
        for group_id, group_data in thinking_group_data.items():
            if group_id == -1:  # Skip failed trajectories
                continue
                
            entropies = group_data.get("entropies", [])
            costs = group_data.get("costs", [])
            rewards = group_data.get("rewards", [])
            
            if len(entropies) == 4:  # Complete group
                for level in range(4):
                    level_stats[level + 1]["entropies"].append(entropies[level])
                    level_stats[level + 1]["costs"].append(costs[level])
                    level_stats[level + 1]["rewards"].append(rewards[level])
        
        # Compute and log level-wise statistics
        level_metrics = {}
        for level in range(1, 5):
            stats = level_stats[level]
            if stats["entropies"]:
                level_metrics.update({
                    f"rlvcr/level_{level}_entropy_mean": np.mean(stats["entropies"]),
                    f"rlvcr/level_{level}_cost_mean": np.mean(stats["costs"]),
                    f"rlvcr/level_{level}_entropy_std": np.std(stats["entropies"]),
                    f"rlvcr/level_{level}_cost_std": np.std(stats["costs"]),
                })
        
        wandb.log(level_metrics, step=step)
        
        # Create level comparison visualization
        self._create_level_comparison_plot(level_stats, step)
    
    def _create_level_comparison_plot(self, level_stats: Dict, step: int):
        """Create visualization comparing different thinking levels"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        levels = []
        entropy_means = []
        entropy_stds = []
        cost_means = []
        cost_stds = []
        
        for level in range(1, 5):
            stats = level_stats[level]
            if stats["entropies"]:
                levels.append(f"Level {level}")
                entropy_means.append(np.mean(stats["entropies"]))
                entropy_stds.append(np.std(stats["entropies"]))
                cost_means.append(np.mean(stats["costs"]))
                cost_stds.append(np.std(stats["costs"]))
        
        if levels:
            # Entropy comparison
            axes[0].bar(levels, entropy_means, yerr=entropy_stds, 
                       capsize=5, alpha=0.7, color='blue')
            axes[0].set_title('Action Confidence by Thinking Level')
            axes[0].set_ylabel('Average Min Probability')
            axes[0].set_ylim(0, 1)
            
            # Cost comparison
            axes[1].bar(levels, cost_means, yerr=cost_stds, 
                       capsize=5, alpha=0.7, color='green')
            axes[1].set_title('Thinking Cost by Level')
            axes[1].set_ylabel('Average Token Count')
        
        plt.tight_layout()
        wandb.log({f"rlvcr/level_comparison": wandb.Image(fig)}, step=step)
        plt.close(fig)


def create_rlvcr_logger(config) -> RLVCRMetricsLogger:
    """Factory function to create RLVCR logger based on config"""
    
    save_cases = config.get("rlvcr_save_cases", True)
    max_cases = config.get("rlvcr_max_cases_per_epoch", 20)
    
    return RLVCRMetricsLogger(save_cases=save_cases, max_cases_per_epoch=max_cases)
