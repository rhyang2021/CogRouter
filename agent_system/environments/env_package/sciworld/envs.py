import torch.multiprocessing as mp
import gymnasium as gym
import numpy as np
import sys
import os
import time
import random
import pdb
from typing import Union
from itertools import product

def is_action_failed(obs):
    return "No known action matches that input." == obs or "can't" in obs or "not" in obs or "doesn't" in obs or "Nothing happened" in obs



def compute_reward(info, multi_modal=False, prev_score=0):
    """
    Compute reward based on score increment.
    Args:
        info (dict): Environment info dictionary
        multi_modal (bool): Whether the environment is multi-modal
        prev_score (float): Previous step score
    Returns:
        float: Computed reward (score increment scaled to 0-10 range)
    """
    current_score = info.get('score', 0)
    
    # Calculate score increment
    score_increment = current_score - prev_score
    
    # Scale to 0-10 range (since original max was 10 and max score is 100)
    # reward = score_increment * 0.1  # 100 -> 10
    reward = 10.0 * float(info['won'])
    
    return reward



def _worker(remote, seed, task_nums, simplifications_preset, env_step_limit, jar_path, split=None, variations_idx=None):
    """Core loop for a subprocess that hosts a *ScienceWorldEnv* instance.

    Commands sent from the main process are *(cmd, data)* tuples:

    - **'step'** *(str)*  → returns ``(obs, reward, done, info)`` where
      ``info['available_actions']`` has already been populated *after* the step.
    - **'reset'** *(int | None)* → returns ``(obs, info)`` with the same
      ``available_actions`` field (obtained immediately after reset).
    - **'close'** → terminates the subprocess.
    """
    # Lazy import to avoid issues
    from scienceworld import ScienceWorldEnv

    # Initialize environment
    env = ScienceWorldEnv("", jar_path, envStepLimit=env_step_limit)

    # Get task names
    taskNames = env.get_task_names()

    # Set random seed for task selection
    random.seed(seed)

    task_id, task_variation = random.choice(variations_idx)
    
    # Initialize previous score tracker
    prev_score = 0


    try:
        while True:
            cmd, data = remote.recv()
            # -----------------------------------------------------------------
            # Environment interaction commands
            # -----------------------------------------------------------------
            if cmd == 'step':
                action = data
                observation, reward, done, info = env.step(action)

                # Get valid actions for the next step
                valid_actions = env.get_possible_actions()
                valid_objs = env.get_possible_objects()
                valid_action_strs = f"Valid_actions: {valid_actions}, OBJ needs to be replaced with one of the following objects: {valid_objs}\n example: <action>focus on door</action>"

                info['available_actions'] = valid_action_strs
                info['observation_text'] = observation
                info["possible_actions"] = env.get_valid_action_object_combinations()

                # Add task score to info
                info['score'] = info.get('score', 0.0)
                info['task_score'] = info['score']
               
                isCompleted = done
                current_score = info['score']
                
                # Handle anomalous negative scores - keep previous score if current is -100
                if current_score == -100:
                    current_score = prev_score  # Use previous score instead
                    info['score'] = prev_score  # Update info to reflect the corrected score
                
                # Set won flag based on completion status
                info["won"] = isCompleted and info["score"] == 100
                # info["won"] = isCompleted and info["score"] > 0
                # Compute reward using score increment
                reward = compute_reward(info, multi_modal=False, prev_score=prev_score)
                # Update previous score for next step
                prev_score = current_score

                # Return the observation, reward, done status, and info
                remote.send((observation, reward, isCompleted, info))

            elif cmd == 'reset':
                # For reset, we can also choose a new random task if we want
                if data is None:
                    # 再次随机选择一个任务
                    task_id, task_variation = random.choice(variations_idx)
                    task_num = task_id
                    taskName = taskNames[task_num]
                else:
                    variation_idx = data
                simplification_str = simplifications_preset if simplifications_preset else ""
                env.load(taskName, task_variation, simplification_str)

                # Reset the environment
                observation, info = env.reset()

                # Get task description
                task_description = env.get_task_description()
                info['task_description'] = task_description

                # Get valid actions for the initial state
                valid_actions = env.get_possible_actions()
                valid_objs = env.get_possible_objects()
                valid_action_strs = f"Valid_actions: {valid_actions}, OBJ needs to be replaced with one of the following objects: {valid_objs}\n example: <action>focus on door</action>"

                info['available_actions'] = valid_action_strs
                info['observation_text'] = observation
                info["possible_actions"] = env.get_valid_action_object_combinations()

                # Set won flag to False initially
                info['won'] = False

                # Include task number in info for tracking
                info['task_num'] = task_num
                # Reset previous score tracker
                prev_score = 0

                # Return the initial observation and info
                remote.send((observation, info))

            # -----------------------------------------------------------------
            # Book‑keeping
            # -----------------------------------------------------------------
            elif cmd == 'close':
                remote.close()
                break

            else:  # pragma: no cover – helps catch typos early
                raise NotImplementedError(f"Unknown command sent to worker: {cmd}")

    finally:  # Ensure the underlying environment *always* shuts down cleanly
        env.close()


class SciWorldMultiProcessEnv(gym.Env):
    """A vectorised, multi‑process wrapper around *ScienceWorldEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        task_nums: list = [1],  # Can be single task or list of tasks
        split: str = "train",  # 数据集分割: "train", "dev", "test"，不再提供默认值None
        simplifications_preset: str = "",  # 简化策略: "easy"
        env_step_limit: int = 100,
        jar_path: str = None,
        variations_idx: list = None  # 用于指定变体索引的列表
    ) -> None:
        super().__init__()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.split = split
        
        # Support both single task and list of tasks
        self.task_nums = task_nums
        self.variations_idx = variations_idx
        
        self.simplifications_preset = simplifications_preset
        self.env_step_limit = env_step_limit
        self.jar_path = jar_path
        random.seed(seed)

        self._rng = np.random.RandomState(seed)

        # -------------------------- Multiprocessing setup --------------------
        self._parent_remotes: list[mp.connection.Connection] = []
        self._workers: list[mp.Process] = []
        ctx = mp.get_context('spawn')
        
        for i in range(self.num_processes):
            parent_remote, child_remote = ctx.Pipe()
            seed_i = seed + i

            # Create a subprocess for each environment instance，移除is_train参数
            # worker = ctx.Process(
            #     target=_worker,
            #     # 从参数列表中移除is_train
            #     args=(child_remote, seed_i, self.task_nums, self.simplifications_preset, 
            #           self.env_step_limit, self.jar_path, self.split),
            #     daemon=True,
            # )
            worker = ctx.Process(
                target=_worker,
                args=(child_remote, seed_i, self.task_nums, self.simplifications_preset, 
                      self.env_step_limit, self.jar_path, self.split, self.variations_idx),
                daemon=True,
            )
            worker.start()
            
            self._workers.append(worker)
            self._parent_remotes.append(parent_remote)
            child_remote.close()

        # Store the most recent available actions to handle timeouts
        self.prev_available_actions = [[] for _ in range(self.num_processes)]
        self.prev_possible_actions = [[] for _ in range(self.num_processes)]

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        for remote, action in zip(self._parent_remotes, actions):
            remote.send(('step', action))

        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, remote in enumerate(self._parent_remotes):
            obs, reward, done, info = remote.recv()
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            
            # Store available actions for later retrieval
            self.prev_available_actions[i] = info['available_actions']
            self.prev_possible_actions[i] = info["possible_actions"]

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        # 不再依赖is_train，统一使用None作为variation参数，让worker函数自行根据split选择变体
        variations = [None for _ in range(self.num_processes)]

        for remote, variation in zip(self._parent_remotes, variations):
            remote.send(('reset', variation))

        obs_list, info_list = [], []
        for i, remote in enumerate(self._parent_remotes):
            obs, info = remote.recv()
            obs_list.append(obs)
            info_list.append(info)
            
            # Store available actions for later retrieval
            self.prev_available_actions[i] = info['available_actions']
            self.prev_possible_actions[i] = info["possible_actions"]
        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    @property
    def get_available_actions(self):
        """Return the available actions for each environment."""
        return self.prev_available_actions

    @property
    def get_admissible_commands(self):
        """Return the available actions for each environment."""
        return self.prev_available_actions

    @property
    def get_possible_actions(self):
        """Return the possible actions for each environment."""
        return self.prev_possible_actions

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        for remote in self._parent_remotes:
            remote.send(('close', None))
        for worker in self._workers:
            worker.join()
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_sciworld_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    task_nums: Union[int, list] = 1,  # Can be single task or list of tasks
    split: str = "train",  # 数据集分割: "train", "dev", "test"，不再提供默认值None
    simplifications_preset: str = "",
    env_step_limit: int = 100,
    jar_path: str = None,
    variations_idx: list = None
):
    """Create a vectorized ScienceWorld environment."""
    return SciWorldMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        task_nums=task_nums,
        split=split,
        simplifications_preset=simplifications_preset,
        env_step_limit=env_step_limit,
        jar_path=jar_path,
        variations_idx=variations_idx
    ) 
