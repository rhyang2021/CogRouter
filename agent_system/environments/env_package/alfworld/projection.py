import re
from typing import List

def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = actions[i][-30:]  # 0 is invalid action for Sokoban
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
            
            actions[i] = extracted_action
            valids[i] = 1

        except:
            actions[i] = actions[i][-30:]

        # check <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return actions, valids, [], valids


def alfworld_projection_nothink(actions: List[str], action_pools: List[List[str]]):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = actions[i][-30:]  # 0 is invalid action for Sokoban
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
            
            actions[i] = extracted_action
            valids[i] = 1

        except:
            actions[i] = actions[i][-30:]

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return actions, valids, [], valids

import re
from typing import List

def alfworld_projection_mcrl(actions: List[str], action_pools: List[List[str]]):
    """
    Checks whether the ALFRED agent outputs are compliant with the prompt requirements.
    Args:
        actions: List of agent's output strings (one for each time step or case)
        action_pools: List of admissible action string lists (for each corresponding step)
    Returns:
        actions_out: A list of extracted action (or blank/placeholder if not found).
        valids: 1 if each response format is valid, 0 otherwise.
        plannings: Extracted <planning>...</planning> content or None if not found.
    """
    skill_tags = [
        r"<planning>.*?</planning>",
        r"<reflection>.*?</reflection>",
        r"<explore>.*?</explore>",
        r"<monitor>.*?</monitor>"
    ]

    actions_out = []
    valids = []
    plannings = []
    action_available = [False] * len(actions)
    for i, output in enumerate(actions):
        valid = 1
        act_str = ""
        # --- 0. Extract <planning> ---
        planning_content = None
        planning_match = re.search(r"<planning>([\s\S]*?)</planning>", output, re.IGNORECASE)
        if planning_match:
            planning_inner = planning_match.group(1).strip()
            planning_content = planning_inner if planning_inner else None
        plannings.append(planning_content)

        # 1. Check for Chinese
        if re.search(r'[\u4e00-\u9fff]', output):
            valid = 0

        # 2. Check ONLY ONE <action>...</action>
        matches = re.findall(r"<action>([\s\S]*?)</action>", output, re.IGNORECASE)
        if len(matches) != 1:
            valid = 0
        else:
            act_candidate = matches[0].strip()
            act_str = act_candidate
            if act_candidate in action_pools[i]:
                action_available[i] = True

        # 3. Action must be in admissible pool
        # if act_str not in action_pools[i]:
        #     valid = 0

        # 4. Check ONLY ONE skill tag, appears before <action> and is non-empty
        found_skill = False
        min_action_pos = output.lower().find("<action>")
        skill_positions = []
        skill_count = 0
        for tag in skill_tags:
            tag_matchs = list(re.finditer(tag, output, re.IGNORECASE | re.DOTALL))
            skill_count += len(tag_matchs)
            for tag_match in tag_matchs:
                if tag_match:
                    # Remove the xml tags and check if not empty
                    inner = re.sub(r"<.*?>", "", tag_match.group(0)).strip()
                    if inner:
                        found_skill = True
                        skill_positions.append(output.find(tag_match.group(0)))
                        # skill tag must appear before <action>
                        if output.find(tag_match.group(0)) > min_action_pos:
                            valid = 0

        if skill_count != 1:  # 只允许一个skill tag，且只出现一次
            valid = 0
        if not found_skill:
            valid = 0

        actions_out.append(act_str)
        valids.append(valid)

    return actions_out, valids, plannings, action_available