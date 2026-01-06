from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
import copy
import pdb

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name, config=None):
        self.buffers = None
        self.config = config
        super().__init__(envs, projection_f, env_name)

    def reset(self):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(text_obs))]
        self.plannings = ["No plan."] * len(text_obs)
        self.tasks = []
        self.pre_text_obs = text_obs
        self.meta_think = self.config is not None and self.config.env.alfworld.meta_think if hasattr(self.config.env, 'alfworld') and hasattr(self.config.env.alfworld, 'meta_think') else False
        self.action_only = getattr(getattr(self.config.env, 'alfworld', None), 'action_only', False)
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos

    def step(self, text_actions: List[str]):
        full_output = copy.deepcopy(text_actions)
        actions, valids, plannings, action_available = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.save_to_history_buffer(self.pre_text_obs, actions, full_output, plannings)
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['action_available'] = to_numpy(action_available[i])
        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')

            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False, history_length: int = 2) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if self.meta_think:
            _ALFWORLD_TEMPLATE_NO_HIS = ALFWORLD_TEMPLATE_NO_HIS_ADA
            _ALFWORLD_TEMPLATE = ALFWORLD_TEMPLATE_ADA
        elif self.action_only:
            _ALFWORLD_TEMPLATE_NO_HIS = ALFWORLD_TEMPLATE_NO_HIS_NOTHINK
            _ALFWORLD_TEMPLATE = ALFWORLD_TEMPLATE_NOTHINK
        else:
            _ALFWORLD_TEMPLATE_NO_HIS = ALFWORLD_TEMPLATE_NO_HIS
            _ALFWORLD_TEMPLATE = ALFWORLD_TEMPLATE

        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or history_length <= 0:
                obs = _ALFWORLD_TEMPLATE_NO_HIS.format(
                    task_description=text_obs[i],
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                # with all the history
                history_think_length = min(10, len(self.buffers[i]))
                start_index = len(self.buffers[i]) - history_think_length
                action_history = ""
                for record in self.buffers[i][-history_think_length:]:
                    env_obs = record["text_obs"]
                    action = record['full_output']
                    action_history += f"\nObservation: {env_obs}"
                    action_history += f"\nOutput: {action}"
                '''
                history_length = len(self.buffers[i])
                # history_length = min(10, len(self.buffers[i]))
                all_actions = [record["action"] for record in self.buffers[i]]
                recent_history = self.buffers[i][-history_length:]
                recent_start_index = len(self.buffers[i]) - history_length
                valid_history_length = len(recent_history)
                action_history = ""

                for j in range(recent_start_index):
                    action = all_actions[j]
                    step_number = j + 1
                    action_history += f"\n[Step {step_number}, Action {step_number}: '{action}']"

                for j, record in enumerate(recent_history):
                    step_number = recent_start_index + j + 1
                    env_obs = record["text_obs"]
                    action = record['action']
                    # action_history += f"\n[Step {step_number}, Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"
                    action_history += f"\n[Step {step_number}, Observation {step_number}: '{env_obs}', Output {step_number}: '{action}']"

                
                if self.meta_think:
                    history_think_length = min(3, len(self.buffers[i]))
                    start_index = len(self.buffers[i]) - history_think_length
                    action_history += "\n- recent reasoning process: \n" 
                    for j, record in enumerate(self.buffers[i][-history_think_length:]):
                        step_number = start_index + j + 1
                        action_history += f"[Step {step_number}, output {step_number}: '{record['full_output']}']\n"
                
                '''
                obs = _ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    # step_count=len(self.buffers[i]),
                    # history_length=valid_history_length,
                    action_history=action_history.strip(),
                    # current_step=len(self.buffers[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def save_to_history_buffer(self, text_obs, actions, text_actions, plannings=[]):
        for i in range(len(actions)):
            self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i], 'full_output': text_actions[i]})
        for i in range(len(plannings)):
            if plannings[i] is not None:
                self.plannings[i] = plannings[i]

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)

                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]

        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break

    def _set_meta_think(self, type: bool):
        self.meta_think = type


class SciWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name, config=None):
        self.buffers = None
        self.config = config
        self.plannings = []
        self.meta_think = self.config is not None and self.config.env.sciworld.meta_think if hasattr(self.config.env, 'sciworld') and hasattr(self.config.env.sciworld, 'meta_think') else False
        super().__init__(envs, projection_f, env_name)

    def reset(self):
        text_obs, infos = self.envs.reset()

        # initialize the history buffer
        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(text_obs))]
        self.plannings = ["No plan."] * len(text_obs)
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task_descriptions(infos)

        full_text_obs = self.build_text_obs(text_obs, [info['available_actions'] for info in infos], init=True)

        return {'text': full_text_obs, 'anchor': text_obs}, infos

    def step(self, text_actions: List[str]):
        full_output = copy.deepcopy(text_actions)
        meta_think = self.config is not None and self.config.env.sciworld.meta_think if hasattr(self.config.env, 'sciworld') and hasattr(self.config.env.sciworld, 'meta_think') else False
        actions, valids, action_available = self.projection_f(text_actions, meta_think=meta_think, available_actions=self.envs.get_possible_actions)

        # 提取MCRL的planning标签（如果有）
        plannings = []
        if meta_think:
            for action in text_actions:
                planning = None
                if "<planning>" in action and "</planning>" in action:
                    start_tag = "<planning>"
                    end_tag = "</planning>"
                    start_idx = action.find(start_tag)
                    end_idx = action.find(end_tag)
                    if start_idx != -1 and end_idx != -1:
                        planning = action[start_idx + len(start_tag):end_idx].strip()
                plannings.append(planning)
        else:
            plannings = [None] * len(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)
        self.save_to_history_buffer(self.pre_text_obs, actions, full_output, plannings)
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, [info['available_actions'] for info in infos])
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['full_output'] = full_output[i]
            info['action_available'] = to_numpy(action_available[i])
            info['score'] = info.get('score', -1)

        next_observations = {'text': full_text_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task_descriptions(self, infos: List[dict]):
        for info in infos:
            if 'task_description' in info:
                self.tasks.append(info['task_description'])
            else:
                self.tasks.append("Unknown task")

    def build_text_obs(self, text_obs: List[str], available_actions: List[List[str]], init: bool = False, history_length: int = 2) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if self.meta_think:
            _SCIWORLD_TEMPLATE_NO_HIS = SCIWORLD_TEMPLATE_NO_HIS_ADA
            _SCIWORLD_TEMPLATE = SCIWORLD_TEMPLATE_ADA
        else:
            _SCIWORLD_TEMPLATE_NO_HIS = SCIWORLD_TEMPLATE_NO_HIS
            _SCIWORLD_TEMPLATE = SCIWORLD_TEMPLATE

        for i in range(len(text_obs)):
            if init or history_length <= 0:
                obs = _SCIWORLD_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=available_actions[i]
                )
            else:
                history_length = len(self.buffers[i])
                all_actions = [record["action"] for record in self.buffers[i]]
                recent_history = self.buffers[i][-history_length:]
                recent_start_index = len(self.buffers[i]) - history_length
                valid_history_length = len(recent_history)
                action_history = ""

                for j in range(recent_start_index):
                    action = all_actions[j]
                    step_number = j + 1
                    action_history += f"\n[Step {step_number}, Action {step_number}: '{action}']"

                for j, record in enumerate(recent_history):
                    step_number = recent_start_index + j + 1
                    env_obs = record["text_obs"]
                    action = record["action"]
                    action_history += f"\n[Step {step_number}, Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"

                if self.config is not None and hasattr(self.config.env, 'sciworld') and hasattr(self.config.env.sciworld, 'meta_think') and self.config.env.sciworld.meta_think:
                    history_think_length = min(3, len(self.buffers[i]))
                    start_index = len(self.buffers[i]) - history_think_length
                    action_history += "\n- recent reasoning process: \n" 
                    for j, record in enumerate(self.buffers[i][-history_think_length:]):
                        step_number = start_index + j + 1
                        action_history += f"[Step {step_number}, output {step_number}: '{record['full_output']}']\n"

                # 统一使用简化格式，不再区分 meta_think
                obs = _SCIWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.buffers[i]),
                    history_length=valid_history_length,
                    action_history=action_history.strip(),
                    current_step=len(self.buffers[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=available_actions[i]
                )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def save_to_history_buffer(self, text_obs, actions, text_actions=None, plannings=None):
        for i in range(len(actions)):
            if text_actions:
                self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i], 'full_output': text_actions[i]})
            else:
                self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i]})

        if plannings:
            for i in range(len(plannings)):
                if plannings[i] is not None:
                    self.plannings[i] = plannings[i]

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                return

    def _set_meta_think(self, type: bool):
        self.meta_think = type

class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name):
        self.buffers = None
        super().__init__(envs, projection_f, env_name)
    
    def reset(self) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        # initialize the history buffer
        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(infos))]
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.save_to_history_buffer(self.pre_text_obs, actions)
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions

    def save_to_history_buffer(self, text_obs, actions):
        for i in range(len(actions)):
            self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i], "full_output": ""})
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False, history_length: int = 2) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                # Get last `history_length` steps
                recent_history = self.buffers[i][-history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.buffers[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\n[Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"
                
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.buffers[i]),
                    history_length=valid_history_length,
                    action_history=action_history.strip(),
                    current_step=len(self.buffers[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return


def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    if "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection, alfworld_projection_nothink, alfworld_projection_mcrl
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        if config.env.alfworld.generalization_level == 2:
            alf_train_config_path = alf_config_path.replace('config_tw.yaml', 'config_tw_train_ood.yaml')
            alf_test_config_path = alf_config_path.replace('config_tw.yaml', 'config_tw_test_ood.yaml')
            _envs = build_alfworld_envs(alf_train_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True)
            _val_envs = build_alfworld_envs(alf_test_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, unseen=True)
        elif config.env.alfworld.generalization_level == 1:
            _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True)
            _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, unseen=True)
        elif config.env.alfworld.generalization_level == 0:
            _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True)
            _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False)
        elif config.env.alfworld.generalization_level == 3:
            alf_train_config_path = alf_config_path.replace('config_tw.yaml', 'config_tw_train_agentgym.yaml')
            alf_test_config_path = alf_config_path.replace('config_tw.yaml', 'config_tw_train_agentgym.yaml')
            _envs = build_alfworld_envs(alf_train_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True)
            _val_envs = build_alfworld_envs(alf_test_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False)
        
        if config.env.alfworld.meta_think:
            projection_f = partial(alfworld_projection_mcrl)
        elif config.env.alfworld.action_only:
            projection_f = partial(alfworld_projection_nothink)
        else:
            projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config.env.env_name, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config.env.env_name, config)
        return envs, val_envs
    
    elif "sciworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.sciworld import build_sciworld_envs, sciworld_projection
        import json
        generalization_level = config.env.sciworld['generalization_level']

        if generalization_level == 2:
            variation_path = 'agent_system/environments/env_package/sciworld/variations_idx/L2_idx.json'
        elif generalization_level == 1:
            variation_path = 'agent_system/environments/env_package/sciworld/variations_idx/L1_idx.json'
        elif generalization_level == 0:
            variation_path = 'agent_system/environments/env_package/sciworld/variations_idx/L0_idx.json'
        elif generalization_level == 3:
            variation_path = 'agent_system/environments/env_package/sciworld/variations_idx/agentgym_idx.json'

        with open(variation_path, 'r') as f:
            variations_idx = json.load(f)

        simplifications_preset = config.env.sciworld.get('simplifications_preset', "easy")
        env_step_limit = config.env.sciworld.get('env_step_limit', 100)
        jar_path = config.env.sciworld.get('jar_path', None)

        _envs = build_sciworld_envs(
            seed=config.env.seed, 
            env_num=config.data.train_batch_size, 
            group_n=group_n, 
            simplifications_preset=simplifications_preset,
            env_step_limit=env_step_limit,
            jar_path=jar_path,
            variations_idx=variations_idx['train']
        )

        _val_envs = build_sciworld_envs(
            seed=config.env.seed + 1000, 
            env_num=config.data.val_batch_size, 
            group_n=1, 
            simplifications_preset=simplifications_preset,
            env_step_limit=env_step_limit,
            jar_path=jar_path,
            variations_idx=variations_idx['test']
        )

        # Create projection function
        projection_f = partial(sciworld_projection)

        # Create environment managers
        envs = SciWorldEnvironmentManager(_envs, projection_f, config.env.env_name, config)
        val_envs = SciWorldEnvironmentManager(_val_envs, projection_f, config.env.env_name, config)

        # Give some time for environments to initialize
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1)

        return envs, val_envs

    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config.env.env_name)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config.env.env_name)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)