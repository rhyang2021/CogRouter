from typing import List
import re

def sciworld_projection(actions: List[str], available_actions=None, meta_think=False):
    """
    A function to process the actions for ScienceWorld.
    
    Args:
        actions: The list of actions to be processed, it is a list of strings.
        available_actions: Optional list of available actions for each environment.
        meta_think: Flag to indicate if MCRL format is used.
        
    Expected format without meta_think:
        <think>some reasoning...</think><action>look at thermometer</action>
        
    Expected format with meta_think:
        <planning>some reasoning...</planning><action>look at thermometer</action>
        or
        <explore>some reasoning...</explore><action>look at thermometer</action>
        or
        <reflection>some reasoning...</reflection><action>look at thermometer</action>
        or
        <proceed>some reasoning...</proceed><action>look at thermometer</action>
        
    Returns:
        processed_actions: The processed actions
        valids: A list of 0/1 indicating if the action is valid
    """
    valids = [0] * len(actions)
    action_available = [False] * len(actions)
    processed_actions = []

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        
        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = original_str.find(start_tag)
        end_idx = original_str.find(end_tag)
        
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                processed_actions.append(original_str[-20:])  # Use last 20 chars as fallback
                continue

            # Extract just the content between the tags
            extracted_action = original_str[start_idx + len(start_tag):end_idx].strip()
            
            processed_actions.append(extracted_action)
            valids[i] = 1
            
            # Check if the action is available (if available_actions is provided)

            env_available_actions = available_actions[i]
            if extracted_action in env_available_actions:
                action_available[i] = True
        except:
            # Use last 20 chars as fallback if there's an error
            processed_actions.append(original_str[-20:])
            
        # Check for think patterns based on meta_think flag
        if meta_think:
            # Check for MCRL tags: <planning>, <explore>, <reflection>, or <proceed>
            if ("<planning>" not in original_str or "</planning>" not in original_str) and \
               ("<explore>" not in original_str or "</explore>" not in original_str) and \
               ("<reflection>" not in original_str or "</reflection>" not in original_str) and \
               ("<monitor>" not in original_str or "</monitor>" not in original_str):
                valids[i] = 0
        else:
            # Check <think>...</think>
            think_start_idx = original_str.find("<think>")
            think_end_idx = original_str.find("</think>")
            if think_start_idx == -1 or think_end_idx == -1:
                valids[i] = 0
            
        # Check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return processed_actions, valids, action_available