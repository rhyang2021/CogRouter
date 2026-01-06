
ALFWORLD_TEMPLATE_NO_HIS_MC = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}

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

2. After your reasoning, you MUST select and present an admissible action for the current step within <action> </action> tags.

Specify the next action the agent should take to progress toward the task goal, following these guidelines:
1. Object and Receptacle References: Use specific identifiers:
- [obj id] for objects (e.g., apple 1).
- [recep id] for receptacles (e.g., countertop 1).
2. Action Validity: Follow the exact format below. Any deviation renders the action invalid:
Valid actions: go to [recep id], take [obj id] from [recep id], put [obj id] in/on [recep id], open/close [recep id], use [obj id], heat/cool/clean [obj id] with [recep id]

<action>
Choose the most appropriate action from the valid actions.
</action>
"""

ALFWORLD_TEMPLATE_MC = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observaitons and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}

Your previous overall plan is: {planning}.

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

2. After your reasoning, you MUST select and present an admissible action for the current step within <action> </action> tags.

Specify the next action the agent should take to progress toward the task goal, following these guidelines:
1. Object and Receptacle References: Use specific identifiers:
- [obj id] for objects (e.g., apple 1).
- [recep id] for receptacles (e.g., countertop 1).
2. Action Validity: Follow the exact format below. Any deviation renders the action invalid:
Valid actions: go to [recep id], take [obj id] from [recep id], put [obj id] in/on [recep id], open/close [recep id], use [obj id], heat/cool/clean [obj id] with [recep id]

<action>
Choose the most appropriate action from the valid actions.
</action>
"""

from datasets import load_dataset
import logging
import numpy as np
import time
import json
import re

dataset = load_dataset("AgentGym/AgentTraj-L", split="train")
alfworld_dataset = dataset.filter(lambda x: "pick" in x["item_id"])

# with open("alfworld_dataset.json", "w", encoding="utf-8") as f:
#     for item in alfworld_dataset:
#         json.dump(item, f, ensure_ascii=False)
#         f.write("\n")

# with open("alfworld_dataset.json", "r", encoding="utf-8") as f:
#     alfworld_dataset = [json.loads(line) for line in f]

trajs = []
for traj in alfworld_dataset:
    conversations = traj["conversations"][2:]
    traj_log = []
    task = conversations[0]['value'].split("Your task is to:")[-1].split("AVAILABLE ACTIONS:")[0].strip()
    for i in range(0, len(conversations), 2):
        obs = conversations[i]['value'].split("AVAILABLE ACTIONS:")[0]
        llm_action = conversations[i + 1]['value']
        # Extracting the action from the llm_action: Action: xxx
        action = llm_action.split("Action:")[-1].strip()
        traj_log.append({
            "observation": obs,
            "action": action
        })
    # print(f"Task: {task}")
    # print(f"Trajectory Log: {traj_log}")
    if len(traj_log) < 30:
        trajs.append({
            "task": task,
            "traj": traj_log
        })

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
                max_tokens=max_tokens
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            time.sleep(2 ** attempt)
    return None

def llm_json(prompt, model="gpt-4o-nlp", temperature=0.0, max_tokens=1024, retries=5):
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response['choices'][0]['message']['content'].replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            print(f"Error: {e}. Retrying... (Attempt {attempt + 1}/{retries})")
            time.sleep(2 ** attempt)
    
    return []


TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment.  
I will provide you with a successful trajectory. You need to supplement the reasoning process.

**Reason using ONLY ONE tag pair and express your reasoning in one concise, brief sentence:**

<planning>
Decompose a complex overall task into clear subgoals, listing each milestone as a separate point. Focus on outlining the full sequence required to complete the overall task, not just the immediate next action.
This approach is typically used at the initial stage of a task, or when significant problems or uncertainties arise that may require re-planning.
All points must be listed explicitly and separately, such as: Step 1: xxx; Step 2: xxx; Step 3: xxx; and so on.
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
    if (len(res) != len(traj)):
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
        output.append({
            "obs": b["observation"],
            "reason": a["reason"],
            "action": a["action"],
        })
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
        meta_traj.append({
            "task": traj["task"],
            "traj": res
        })
        
        step_level_data = []
        current_planning = "No plan."
        for i, item in enumerate(res):
            if i == 0:
                prompt = ALFWORLD_TEMPLATE_NO_HIS_MC.format(
                    current_observation=item["obs"],
                )
            else:
                action_history = "\n".join([f"[Observation {j + 1}: '{res[j]['obs']}', Action {j + 1}: '{res[j]['action']}']" for j in range(i)])
                history_think_length = min(3, i)
                action_history += "\n- recent reasoning process: \n"
                for j in range(i - history_think_length, i):
                    action_history += f"[Observation {j + 1}: {res[j]['obs']}, output: '{res[j]['reason']} <action>{res[j]['action']}</action>']\n"
                prompt = ALFWORLD_TEMPLATE_MC.format(
                    task_description=traj["task"],
                    step_count=i,
                    history_length=i,
                    action_history=action_history,
                    current_step=i + 1,
                    current_observation=item["obs"],
                    planning=current_planning
                )

                # update current planning
                if '<planning>' in item["reason"]:
                    current_planning = re.search(r'<planning>(.*?)</planning>', item["reason"], re.DOTALL)
                    if current_planning:
                        current_planning = current_planning.group(1).strip()
                    else:
                        current_planning = "No plan."
            response = f"{item['reason']}\n<action>{item['action']}</action>\n"

            # print(f"Prompt: {prompt}")
            # print(f"Response: {response}")
            step_level_data.append({
                "step": i + 1,
                "obs": item["obs"],
                "prompt": prompt,
                "response": response,
            })

        sft_data.append({
            "task": traj["task"],
            "done": "True",
            "data": step_level_data
        })

        with open(f"data/alfworld_sft_traj_0705.json", "w", encoding="utf-8") as f:
            json.dump(meta_traj, f, ensure_ascii=False, indent=4)

        with open(f"data/alfworld_sft_data_0705.json", "w", encoding="utf-8") as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=4)