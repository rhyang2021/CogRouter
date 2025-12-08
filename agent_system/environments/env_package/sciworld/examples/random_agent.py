#!/usr/bin/env python3
"""
A simple example of using the ScienceWorld environment with a random agent.
"""

import os
import sys
import random
import time
import numpy as np
from functools import partial

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from agent_system.environments.env_package.sciworld import build_sciworld_envs, sciworld_projection
from agent_system.environments.env_manager import SciWorldEnvironmentManager

def main():
    # Configuration
    seed = 42
    env_num = 1
    group_n = 1
    task_num = 13  # Choose a task number (0-23)
    simplifications_preset = "easy"  # Can be "easy", "medium", "hard", or ""
    env_step_limit = 100
    
    # Path to the ScienceWorld JAR file
    # You need to download this from https://github.com/allenai/ScienceWorld
    jar_path = os.path.expanduser("~/ScienceWorld.jar")
    
    # Check if JAR file exists
    if not os.path.exists(jar_path):
        print(f"ScienceWorld JAR file not found at {jar_path}")
        print("Please download it from https://github.com/allenai/ScienceWorld")
        print("You can use: wget https://github.com/allenai/ScienceWorld/releases/download/1.0.0/scienceworld.jar -O ~/ScienceWorld.jar")
        return
    
    print("Building SciWorld environment...")
    
    # Build environment
    envs = build_sciworld_envs(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=True,
        task_num=task_num,
        simplifications_preset=simplifications_preset,
        env_step_limit=env_step_limit,
        jar_path=jar_path
    )

    # Create projection function
    projection_f = partial(sciworld_projection)
    
    # Create environment manager
    env_manager = SciWorldEnvironmentManager(envs, projection_f, 'sciworld')
    
    print("Resetting environment...")
    
    # Reset environment
    obs, infos = env_manager.reset()
    
    # Print initial observation
    print("\n" + "="*80)
    print("INITIAL OBSERVATION")
    print("="*80)
    print(obs['text'][0])
    
    # Run for a number of steps
    max_steps = 10
    for i in range(max_steps):
        print("\n" + "="*80)
        print(f"STEP {i+1}/{max_steps}")
        print("="*80)
        
        # Get available actions
        available_actions = env_manager.envs.get_available_actions
        
        # Choose a random action from available actions
        random_action = "<think>Let me choose a random action.</think><action>" + random.choice(available_actions[0]) + "</action>"
        print(f"Taking action: {random_action}")

        # Take a step
        obs, rewards, dones, infos = env_manager.step([random_action])
        
        # Print observation, reward, and done status
        print("\nObservation:")
        print(obs['anchor'][0])
        print(f"\nReward: {rewards[0]}")
        print(f"Done: {dones[0]}")

        # Check if episode is done
        if dones[0]:
            print("\nEpisode completed!")
            break

        # Small delay for readability
        time.sleep(1)

    # Close environment
    env_manager.close()
    print("\nEnvironment closed.")

if __name__ == "__main__":
    main()
