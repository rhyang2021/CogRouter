SCIWORLD_TEMPLATE_NO_HIS_MC = """
You are an expert agent operating in the ScienceWorld environment, which is a text-based virtual environment centered around accomplishing tasks from the elementary science curriculum.
Your current task is: {task_description}

Your current observation is: {current_observation}
Here are the actions you may take:
[
{{"action": "open/close OBJ", "description": "open/close a container"}},
{{"action": "de/activate OBJ", "description": "activate/deactivate a device"}},
{{"action": "connect OBJ to OBJ", "description": "connect electrical components"}},
{{"action": "disconnect OBJ", "description": "disconnect electrical components"}},
{{"action": "use OBJ [on OBJ]", "description": "use a device/item"}},
{{"action": "look around", "description": "describe the current room"}},
{{"action": "look at OBJ", "description": "describe an object in detail"}},
{{"action": "look in OBJ", "description": "describe a container's contents"}},
{{"action": "read OBJ", "description": "read a note or book"}},
{{"action": "move OBJ to OBJ", "description": "move an object to a container"}},
{{"action": "pick up OBJ", "description": "move an object to the inventory"}},
{{"action": "put down OBJ", "description": "drop an inventory item"}},
{{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}},
{{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}},
{{"action": "mix OBJ", "description": "chemically mix a container"}},
{{"action": "go to LOC", "description": "move to a new location"}},
{{"action": "eat OBJ", "description": "eat a food"}},
{{"action": "flush OBJ", "description": "flush a toilet"}},
{{"action": "focus on OBJ", "description": "signal intent on a task object"}},
{{"action": "wait", "description": "take no action for 10 iterations"}},
{{"action": "wait1", "description": "take no action for 1 iteration"}},
{{"action": "task", "description": "describe current task"}},
{{"action": "inventory", "description": "list your inventory"}}
]

Now it's your turn to take an action, following these steps:

1. First, reason using ONLY ONE tag pair and express your reasoning in one concise, brief sentence:

<planning>
Plan or replan the entire task by breaking it down into high-level steps. Focus on outlining the full sequence required to complete the overall task, not just the immediate next action. 
Use this at the beginning of complex tasks or whenever the previous plan is incorrect or insufficient.
It is necessary to list all the points separately. eg, step 1: xxx, step 2: xxx, step 3: xxx, etc.
</planning>

<explore>
When results are unexpected or information is lacking, use current observations to think outside the box and list as many possible locations, items, or actions as possible.
Use this approach when facing obstacles that require creative and innovative thinking.
</explore>

<reflection>
Analyze the reasons for errors in task execution and correct them by exploring alternative approaches. 'No known action matches that input.' indicates the action is invalid.
This is typically used when several consecutive actions yield no substantial progress. 
</reflection>

<monitor>  
Continuously track the current progress and history of reasoning and execution throughout the task. Recall the current subgoal and consider the next concrete action, ensuring agent alignment with the overall plan.  
Typically used when task outcomes are as expected and no other mode of reasoning is required.
</monitor>

2. After your reasoning, you MUST select and present an appropriate action for the current step within <action> </action> tags.
"""

SCIWORLD_TEMPLATE_MC = """
You are an expert agent operating in the ScienceWorld environment, which is a text-based virtual environment centered around accomplishing tasks from the elementary science curriculum.
Your current task is: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Here are the actions you may take:
[
{{"action": "open/close OBJ", "description": "open/close a container"}},
{{"action": "de/activate OBJ", "description": "activate/deactivate a device"}},
{{"action": "connect OBJ to OBJ", "description": "connect electrical components"}},
{{"action": "disconnect OBJ", "description": "disconnect electrical components"}},
{{"action": "use OBJ [on OBJ]", "description": "use a device/item"}},
{{"action": "look around", "description": "describe the current room"}},
{{"action": "look at OBJ", "description": "describe an object in detail"}},
{{"action": "look in OBJ", "description": "describe a container's contents"}},
{{"action": "read OBJ", "description": "read a note or book"}},
{{"action": "move OBJ to OBJ", "description": "move an object to a container"}},
{{"action": "pick up OBJ", "description": "move an object to the inventory"}},
{{"action": "put down OBJ", "description": "drop an inventory item"}},
{{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}},
{{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}},
{{"action": "mix OBJ", "description": "chemically mix a container"}},
{{"action": "go to LOC", "description": "move to a new location"}},
{{"action": "eat OBJ", "description": "eat a food"}},
{{"action": "flush OBJ", "description": "flush a toilet"}},
{{"action": "focus on OBJ", "description": "signal intent on a task object"}},
{{"action": "wait", "description": "take no action for 10 iterations"}},
{{"action": "wait1", "description": "take no action for 1 iteration"}},
{{"action": "task", "description": "describe current task"}},
{{"action": "inventory", "description": "list your inventory"}}
]

Your previous overall plan is: {planning}.  Please strictly adhere to your plan.

Now it's your turn to take an action, following these steps:

1. First, reason using ONLY ONE tag pair and express your reasoning in one concise, brief sentence:

<planning>
Plan or replan the entire task by breaking it down into high-level steps. Focus on outlining the full sequence required to complete the overall task, not just the immediate next action. 
Use this at the beginning of complex tasks or whenever the previous plan is incorrect or insufficient.
It is necessary to list all the points separately. eg, step 1: xxx, step 2: xxx, step 3: xxx, etc.
</planning>

<explore>
When results are unexpected or information is lacking, use current observations to think outside the box and list as many possible locations, items, or actions as possible.
Use this approach when facing obstacles that require creative and innovative thinking.
</explore>

<reflection>
Analyze the reasons for errors in task execution and correct them by exploring alternative approaches. 'No known action matches that input.' indicates the action is invalid.
This is typically used when several consecutive actions yield no substantial progress. 
</reflection>

<monitor>  
Continuously track the current progress and history of reasoning and execution throughout the task. Recall the current subgoal and consider the next concrete action, ensuring agent alignment with the overall plan.  
Typically used when task outcomes are as expected and no other mode of reasoning is required.
</monitor>

2. After your reasoning, you MUST select and present an appropriate action for the current step within <action> </action> tags.
"""


from datasets import load_dataset
import logging
import numpy as np
import time
import json
import re

dataset = load_dataset("AgentGym/AgentTraj-L", split="train")
sciworld_dataset = dataset.filter(lambda x: "sciworld" in x["item_id"])

# with open("alfworld_dataset.json", "w", encoding="utf-8") as f:
#     for item in alfworld_dataset:
#         json.dump(item, f, ensure_ascii=False)
#         f.write("\n")

# with open("alfworld_dataset.json", "r", encoding="utf-8") as f:
#     alfworld_dataset = [json.loads(line) for line in f]

trajs = []
for traj in sciworld_dataset:
    conversations = traj["conversations"][2:]
    traj_log = []
    task = (
        conversations[0]["value"]
        .split("\n")[0]
        .strip()
    )
    for i in range(0, len(conversations), 2):
        obs = conversations[i]["value"].strip()
        llm_action = conversations[i + 1]["value"]
        # Extracting the action from the llm_action: Action: xxx
        action = llm_action.split("Action:")[-1].strip()
        traj_log.append({"observation": obs, "action": action})
    # print(f"Task: {task}")
    # print(f"Trajectory Log: {traj_log}")
    if len(traj_log) < 30:
        trajs.append({"task": task, "traj": traj_log})

# 平均轨迹长度
avg_traj_length = sum(len(traj["traj"]) for traj in trajs) / len(trajs)
print(f"Average trajectory length: {avg_traj_length}")

# 平均轨迹长度小于30的轨迹数量
short_traj_count = sum(len(traj["traj"]) < 30 for traj in trajs)
print(f"Number of trajectories with length less than 30: {short_traj_count}")

print(f"Total trajectories: {len(trajs)}")

import random

random.shuffle(trajs)

import openai

openai.api_key = "PZV47hdixAvMq1U4CxPMb1mk6xGpOHOk"
openai.api_base = "https://gptproxy.llmpaas.woa.com/v1"


def llm(prompt, model="gpt-4o-nlp", temperature=0.0, max_tokens=1024, retries=3):
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            time.sleep(2**attempt)
    return None


def llm_json(prompt, model="gpt-4o-nlp", temperature=0.0, max_tokens=1024, retries=5):
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = (
                response["choices"][0]["message"]["content"]
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )
            return json.loads(content)
        except Exception as e:
            print(f"Error: {e}. Retrying... (Attempt {attempt + 1}/{retries})")
            time.sleep(2**attempt)
    
    return []


TEMPLATE = """
You are an expert agent operating in the ScienceWorld environment.  
I will provide you with a successful trajectory. You need to supplement the reasoning process.

**Reason using ONLY ONE tag pair and express your reasoning in one concise, brief sentence:**

<planning>
Ignore any steps provided in the original task instructions. Break down the complex overall task into more detailed subgoals (at least 2), listing each milestone as a separate bullet point.
All steps must be listed explicitly and comprehensively, for example:
Step 1: xxx
Step 2: xxx
Step 3: xxx
This approach is typically used at the initial stage of a task.
</planning>

<explore>
When immediate next steps have a clear exploratory nature—such as when searching for an unknown object or information—use current observations to think outside the box and generate as many possible hypotheses, locations, items, or actions as possible.
Employ this approach when results are unexpected, information is lacking, or obstacles demand creative and innovative problem-solving.
</explore>

<reflection>
Reflect on the current state, task progress, objectives, and reasons for failures when the task has stalled for an extended period, incorrect actions have been taken, or the current situation has been misjudged. Analyze potential causes for errors or lack of progress, and consider alternative strategies or perspectives to overcome obstacles.
This is especially useful when several consecutive actions do not yield breakthroughs, or when persistent mistakes indicate the need for a deeper reassessment.
</reflection>

<monitor>  
Continuously track the current progress and history of reasoning and execution throughout the task.
Firstly recall the current subgoal based on the previously established overall plan, then consider the next action required to achieve this subgoal.
Typically used when task outcomes are as expected and no other mode of reasoning is required.
</monitor>

You need to output a list in JSON format, with the same length as the trajectory. Each element should contain two key-value pairs, for example:  
```json
[{{"reason": "<explore>The book may be in the cabinet, shelf, so in the next steps I need to search these locations.</explore>", "action": "go to shelf 1"}},
{{"reason": "<monitor>Currently, my sub-goal is to obtain item A. I have already spotted A, and in order to accomplish this objective, I need to pick it up.</monitor>", "action": "pick up A"}}]
```
The "action" field must match the action in the trajectory, and the "reason" field should be a reasonable reasoning process inferred from the context of previous actions and the next few actions.

now the trajectory is as follows: {traj}
"""


def merge(res, traj):
    if len(res) != len(traj):
        print(f"Length mismatch: {len(res)} vs {len(traj)}")
        return False, None

    if "action" not in res[0] or "reason" not in res[0]:
        print("Missing 'action' or 'reason' in the response structure.")
        return False, None

    output = []
    for a, b in zip(res, traj):
        if a["action"] != b["action"]:
            print(f"Action mismatch: {a['action']} vs {b['action']}")
            return False, None
        if not a["reason"].startswith("<"):
            print(f"Invalid reason format: {a['reason']}")
            return False, None
        output.append(
            {
                "obs": b["observation"],
                "reason": a["reason"],
                "action": a["action"],
            }
        )
    return True, output


meta_traj = []
sft_data = []

for traj in trajs[:300]:
    prompt = TEMPLATE.format(traj=json.dumps(traj["traj"], ensure_ascii=False))
    # print(prompt)
    res = llm_json(prompt)
    valid, res = merge(res, traj["traj"])

    if not valid:
        print(f"Invalid response for task: {traj['task']}")
        print(f"Response: {res}")
        continue
    for i, item in enumerate(res):
        print(f"Step {i + 1} --------------------")
        print(f"Observation: {item['obs']}")
        print(f"Reason: {item['reason']}")
        print(f"Action: {item['action']}")

    if valid:
        meta_traj.append({"task": traj["task"], "traj": res})

        step_level_data = []
        current_planning = "No plan."
        for i, item in enumerate(res):
            if i == 0:
                prompt = SCIWORLD_TEMPLATE_NO_HIS_MC.format(
                    task_description=traj["task"],
                    current_observation=item["obs"],
                )
            else:
                action_history = "\n".join(
                    [
                        f"[Observation {j + 1}: '{res[j]['obs']}', Action {j + 1}: '{res[j]['action']}']"
                        for j in range(i)
                    ]
                )
                history_think_length = min(3, i)
                action_history += "\n- recent reasoning process: \n"
                for j in range(i - history_think_length, i):
                    action_history += f"[Observation {j + 1}: {res[j]['obs']}, output: '{res[j]['reason']} <action>{res[j]['action']}</action>']\n"
                prompt = SCIWORLD_TEMPLATE_MC.format(
                    task_description=traj["task"],
                    step_count=i,
                    history_length=i,
                    action_history=action_history,
                    current_step=i + 1,
                    current_observation=item["obs"],
                    planning=current_planning,
                )

                # update current planning
                if "<planning>" in item["reason"]:
                    current_planning = re.search(
                        r"<planning>(.*?)</planning>", item["reason"], re.DOTALL
                    )
                    if current_planning:
                        current_planning = current_planning.group(1).strip()
                    else:
                        current_planning = "No plan."
            response = f"{item['reason']}\n<action>{item['action']}</action>\n"

            # print(f"Prompt: {prompt}")
            # print(f"Response: {response}")
            step_level_data.append(
                {
                    "step": i + 1,
                    "obs": item["obs"],
                    "prompt": prompt,
                    "response": response,
                }
            )

        sft_data.append({"task": traj["task"], "done": "True", "data": step_level_data})

        with open(f"data/sciworld_sft_traj_0705.json", "w", encoding="utf-8") as f:
            json.dump(meta_traj, f, ensure_ascii=False, indent=4)

        with open(f"data/sciworld_sft_data_0705.json", "w", encoding="utf-8") as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=4)


# recent_history = self.buffers[i][-history_length:]
# valid_history_length = len(recent_history)
# start_index = len(self.buffers[i]) - valid_history_length
# action_history = ""
# for j, record in enumerate(recent_history):
#     step_number = start_index + j + 1
#     action = record["action"]
#     env_obs = record["text_obs"]
#     action_history += f"\n[Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"

# if self.config is not None and self.config.env.alfworld.meta_think:
#     history_think_length = min(3, len(self.buffers[i]))
#     start_index = len(self.buffers[i]) - history_think_length
#     action_history += "\n- recent reasoning process: \n"
#     for j, record in enumerate(self.buffers[i][-history_think_length:]):
#         step_number = start_index + j + 1
#         action_history += f"[Observation {step_number}: {record["text_obs"]}, output: '{record['full_output']}']\n"
