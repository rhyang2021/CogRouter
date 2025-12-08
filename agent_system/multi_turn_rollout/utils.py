import torch
import numpy as np
import random
from typing import List, Tuple, Dict
import math
from PIL import Image
from verl import DataProto

def to_list_of_dict(batch: DataProto) -> list[dict]:
    tensors = batch.batch
    non_tensor = batch.non_tensor_batch
    batch_size = len(tensors['input_ids'])
    save_list = []
    for bs in range(batch_size):
        save_dict = dict()
        for key, val in tensors.items():
            save_dict[key] = val[bs]
        for key, val in non_tensor.items():
            save_dict[key] = val[bs]
        save_list.append(save_dict)
    return save_list


def torch_to_numpy(tensor, is_object=False):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        pass
    else:
        raise ValueError(f"Unsupported type: {type(tensor)})")

    if is_object:
        tensor = tensor.astype(object)
    return tensor

def numpy_to_torch(array, device):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array).to(device)
    elif isinstance(array, torch.Tensor):
        array = array.to(device)
    else:
        raise ValueError(f"Unsupported type: {type(array)})")
    return array


def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 256 * 256):
    if isinstance(image, torch.Tensor):
        image = torch_to_numpy(image)
    if image.max() < 1:
        image = image * 255.0
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image = Image.fromarray(image)

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def adjust_batch(config, data: DataProto) -> DataProto:
    size_divisor_ref = config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu * config.trainer.n_gpus_per_node
    size_divisor_rollout = config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu * config.trainer.n_gpus_per_node
    size_divisor_actor = config.actor_rollout_ref.actor.ppo_mini_batch_size
    size_divisor = np.lcm.reduce(np.array([size_divisor_ref, size_divisor_rollout, size_divisor_actor])).item()

    # check if the batch size is divisible by the dp size, if not, delete the last few samples to make it divisible
    bs = len(data)
    print(f"DEBUG adjust_batch: bs={bs}, size_divisor={size_divisor}")
    
    # If size_divisor is larger than batch_size, just return the original data
    if size_divisor > bs:
        print(f"WARNING: size_divisor ({size_divisor}) > batch_size ({bs}), returning original data")
        return data
        
    if bs % size_divisor != 0:
        remainder = bs % size_divisor
        
        # Smart removal strategy: prioritize failed trajectories and preserve thinking groups
        remove_indices = []
        
        # Check if we have RLVCR thinking group information
        if 'thinking_group_id' in data.non_tensor_batch:
            thinking_group_ids = data.non_tensor_batch['thinking_group_id']
            is_original = data.non_tensor_batch.get('is_original', None)
            episode_rewards = data.non_tensor_batch.get('episode_rewards', None)
            
            print(f"DEBUG adjust_batch: RLVCR mode - prioritizing failed trajectories")
            
            # Strategy 1: Remove failed trajectory samples first (thinking_group_id = -1)
            failed_indices = np.where(thinking_group_ids == -1)[0]
            failed_to_remove = min(len(failed_indices), remainder)
            
            if failed_to_remove > 0:
                remove_indices.extend(failed_indices[:failed_to_remove])
                remainder -= failed_to_remove
                print(f"DEBUG adjust_batch: removed {failed_to_remove} failed trajectory samples")
            
            # Strategy 2: If still need to remove more, remove complete thinking groups with low rewards
            if remainder > 0:
                # Get unique successful groups (group_id >= 0)
                successful_groups = np.unique(thinking_group_ids[thinking_group_ids >= 0])
                
                # Calculate average reward for each group if episode_rewards available
                group_rewards = {}
                if episode_rewards is not None:
                    for group_id in successful_groups:
                        group_mask = thinking_group_ids == group_id
                        group_rewards[group_id] = np.mean(episode_rewards[group_mask])
                    
                    # Sort groups by reward (lowest first)
                    sorted_groups = sorted(group_rewards.keys(), key=lambda x: group_rewards[x])
                    print(f"DEBUG adjust_batch: group rewards: {group_rewards}")
                else:
                    # If no reward info, use random order
                    sorted_groups = list(successful_groups)
                    np.random.shuffle(sorted_groups)
                
                # Remove complete groups (4 samples each) starting from lowest reward
                groups_to_remove = remainder // 4  # How many complete groups we can remove
                groups_removed = 0
                
                for group_id in sorted_groups:
                    if groups_removed >= groups_to_remove:
                        break
                    
                    group_indices = np.where(thinking_group_ids == group_id)[0]
                    if len(group_indices) == 4:  # Only remove complete groups
                        remove_indices.extend(group_indices)
                        remainder -= 4
                        groups_removed += 1
                        print(f"DEBUG adjust_batch: removed complete thinking group {group_id} (reward: {group_rewards.get(group_id, 'unknown')})")
            
            # Strategy 3: If still need to remove more, randomly select from remaining
            if remainder > 0:
                remaining_indices = [i for i in range(bs) if i not in remove_indices]
                if len(remaining_indices) >= remainder:
                    additional_remove = np.random.choice(remaining_indices, remainder, replace=False)
                    remove_indices.extend(additional_remove)
                    print(f"DEBUG adjust_batch: randomly removed {remainder} additional samples")
                else:
                    # Edge case: not enough samples left
                    remove_indices.extend(remaining_indices)
                    print(f"DEBUG adjust_batch: removed all remaining {len(remaining_indices)} samples")
            
            remove_indices = np.array(remove_indices)
            
        else:
            # Fallback: if no thinking group info, try to use episode_rewards
            episode_rewards = data.non_tensor_batch.get('episode_rewards', None)
            
            if episode_rewards is not None:
                print(f"DEBUG adjust_batch: No thinking groups, prioritizing by episode rewards")
                
                # Remove samples with lowest rewards first
                sorted_indices = np.argsort(episode_rewards)  # lowest rewards first
                remove_indices = sorted_indices[:remainder]
                
                removed_rewards = episode_rewards[remove_indices]
                print(f"DEBUG adjust_batch: removed {remainder} samples with rewards: min={removed_rewards.min():.3f}, max={removed_rewards.max():.3f}")
            else:
                # Original random removal as last resort
                print(f"DEBUG adjust_batch: No group info or rewards, using random removal")
                remove_indices = np.random.choice(bs, remainder, replace=False)
        
        # Sort remove_indices to maintain stability when deleting
        remove_indices = np.sort(remove_indices)
        
        # Create a boolean mask for elements to keep
        keep_mask = np.ones(bs, dtype=bool)
        keep_mask[remove_indices] = False

        keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool, device=data.batch['input_ids'].device)
        # Apply the mask to keep elements in their original order
        print(f"DEBUG adjust_batch: removing {remainder} samples, keep_mask shape: {keep_mask.shape}")
        tensor_data = data.batch[keep_mask_tensor]
        print(f"DEBUG adjust_batch: after masking, tensor_data batch_size: {tensor_data.batch_size}")
        non_tensor_data = {key: val[keep_mask] for key, val in data.non_tensor_batch.items()}
        
        # Update meta_info if it contains original_indices
        adjusted_meta_info = data.meta_info.copy()
        if 'original_indices' in adjusted_meta_info:
            original_indices = adjusted_meta_info['original_indices']
            # Create mapping from old indices to new indices
            old_to_new = {}
            new_idx = 0
            for old_idx in range(bs):
                if keep_mask[old_idx]:
                    old_to_new[old_idx] = new_idx
                    new_idx += 1
            
            # Update original_indices: filter out removed indices and remap remaining ones
            updated_original_indices = []
            for old_original_idx in original_indices:
                if old_original_idx in old_to_new:
                    updated_original_indices.append(old_to_new[old_original_idx])
            
            adjusted_meta_info['original_indices'] = updated_original_indices
            print(f"DEBUG adjust_batch: updated original_indices from {len(original_indices)} to {len(updated_original_indices)}")
        
        adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=adjusted_meta_info)
        del data
    else:
        adjusted_batch = data

    return adjusted_batch


def filter_group_data(batch_list : List[Dict],
                        episode_rewards: np.ndarray,
                        episode_lengths: np.ndarray,
                        success: Dict[str, np.ndarray],
                        traj_uid: np.ndarray,
                        config,
                        last_try: bool = False,
                        ):
    """
    Dynamic Sampling:
    Over-sample and filter out episode group in which all episodes have the same rewards.
    Adopted from DAPO (https://arxiv.org/abs/2503.14476)
    """
    if last_try:
        return batch_list, episode_rewards, episode_lengths, success, traj_uid
    
    batch_size = config.data.train_batch_size
    group_n = config.env.rollout.n
    if group_n <= 1:
        print("Warning: group_n <= 1, no need to adopt dynamic sampling")

    # Handle each group
    keep_indices = np.array([], dtype=np.int64)
    for i in range(batch_size):
        # Get the indices of the current group
        group_indices = np.arange(i * group_n, (i + 1) * group_n)
        group_rewards = episode_rewards[group_indices]

        # check if all group_traj_uid are the same
        for index in group_indices:
            assert batch_list[index][0]['uid'] == batch_list[group_indices[0]][0]['uid']

        # Check if all rewards in the group are the same
        if not np.all(group_rewards == group_rewards[0]):
            # If so, keep the entire group, otherwise, remove it
            keep_indices = np.concatenate((keep_indices, group_indices))
    
    # Filter the batch_list, episode_rewards, episode_lengths, and success based on the keep_indices
    success = {
        key: value[keep_indices]
        for key, value in success.items()
        if len(value) == len(batch_list)
    }
    batch_list = [batch_list[i] for i in keep_indices]
    episode_rewards = episode_rewards[keep_indices]
    episode_lengths = episode_lengths[keep_indices]
    # success = {key: value[keep_indices] for key, value in success.items()}
    traj_uid = traj_uid[keep_indices]

    return batch_list, episode_rewards, episode_lengths, success, traj_uid

