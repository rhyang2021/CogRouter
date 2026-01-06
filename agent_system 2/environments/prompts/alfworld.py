# projection_f# --------------------- ALFWorld --------------------- #
'''
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observaitons and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""
'''

ALFWORLD_TEMPLATE_NO_HIS = """
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. After each turn, the environment will give you immediate feedback based on which you plan your next few steps. If the environment outputs 'Nothing happened', that means the previous action is invalid and you should try more options. 
Reminder: the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. 

{task_description}
Here is the task history for past 10 steps:
Observation: {current_observation} AVAILABLE ACTIONS: {admissible_actions}

Now it's your turn to generate next step.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
""".strip()


ALFWORLD_TEMPLATE = """
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. After each turn, the environment will give you immediate feedback based on which you plan your next few steps. If the environment outputs 'Nothing happened', that means the previous action is invalid and you should try more options. 
Reminder: the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. 

{task_description}
Here is the task history for past 10 steps:
{action_history}
Observation: {current_observation} AVAILABLE ACTIONS: {admissible_actions}

Now it's your turn to generate next step.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
""".strip()


ALFWORLD_TEMPLATE_NO_HIS_NOTHINK = """
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. After each turn, the environment will give you immediate feedback based on which you plan your next few steps. If the environment outputs 'Nothing happened', that means the previous action is invalid and you should try more options. 
Reminder: the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. 

At each step, you MUST respond in the following format and output NOTHING else:
1. If the next step is straightforward and does not require explicit reasoning:
<think></think>
<action>your_next_action</action>
Example:
<think></think>
<action>go to kitchen</action>
2. If you need to reason about what to do next based on the observation and task:
<think>Your concise reasoning about what to do next</think>
<action>your_chosen_action</action>
Example:
<think>The task is to find something to eat. I am in the kitchen and see an orange on the counter, so I should pick it up.</think>
<action>pick up orange</action>

{task_description}
Your current observation is: {current_observation} AVAILABLE ACTIONS: {admissible_actions}

Now it's your turn to generate next step.
""".strip()

ALFWORLD_TEMPLATE_NOTHINK = """
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. After each turn, the environment will give you immediate feedback based on which you plan your next few steps. If the environment outputs 'Nothing happened', that means the previous action is invalid and you should try more options. 
Reminder: the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. 

At each step, you MUST respond in the following format and output NOTHING else:
1. If the next step is straightforward and does not require explicit reasoning:
<think></think>
<action>your_next_action</action>
Example:
<think></think>
<action>go to kitchen</action>
2. If you need to reason about what to do next based on the observation and task:
<think>Your concise reasoning about what to do next</think>
<action>your_chosen_action</action>
Example:
<think>The task is to find something to eat. I am in the kitchen and see an orange on the counter, so I should pick it up.</think>
<action>pick up orange</action>

{task_description}
Here is the task history for past 10 steps:
{action_history}
Observation: {current_observation} AVAILABLE ACTIONS: {admissible_actions}

Now it's your turn to generate next step.
""".strip()


# ALFWORLD_TEMPLATE_NO_HIS_MC = """
# You are an expert agent operating in the ALFRED Embodied Environment.
# Your current observation is: {current_observation}
# Your admissible actions of the current situation are: [{admissible_actions}].

# Now it's your turn to take an action, following these steps:

# 1. First, reason using ONLY ONE tag pair and express your reasoning in one concise, brief sentence:

# <planning>
# Plan or replan the entire task by breaking it down into high-level steps. Focus on outlining the full sequence required to complete the overall task, not just the immediate next action. 
# Use this at the beginning of complex tasks or whenever the previous plan is incorrect or insufficient.
# </planning>

# <explore>
# When results are unexpected or information is lacking, use current observations to think outside the box and list as many possible locations, items, or actions as possible.
# Use this approach when facing obstacles that require creative and innovative thinking.
# </explore>

# <reflection>
# Analyze the reasons for errors in task execution and correct them by exploring alternative approaches. 'Nothing happens' indicates the action is invalid.
# This is typically used when several consecutive actions yield no substantial progress. 
# </reflection>

# <proceed>
# Proceed to the next step based on the prior overall plan or the most recent unfinished exploration.
# This is most often used when the model clearly knows what to do next.
# </proceed>

# 2. After your reasoning, you MUST select and present an admissible action for the current step within <action> </action> tags.

# Specify the next action the agent should take to progress toward the task goal, following these guidelines:
# 1. Object and Receptacle References: Use specific identifiers:
# - [obj id] for objects (e.g., apple 1).
# - [recep id] for receptacles (e.g., countertop 1).
# 2. Action Validity: Follow the exact format below. Any deviation renders the action invalid:
# Valid actions: go to [recep id], take [obj id] from [recep id], put [obj id] in/on [recep id], open/close [recep id], use [obj id], heat/cool/clean [obj id] with [recep id]

# <action>
# Choose the most appropriate action from the valid actions.
# </action>
# """

ALFWORLD_TEMPLATE_NO_HIS_MC = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

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
Your admissible actions of the current situation are: [{admissible_actions}].

Your previous overall plan is: {planning}. Please strictly adhere to your plan.

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



ALFWORLD_TEMPLATE_NO_HIS_ADA = """
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. After each turn, the environment will give you immediate feedback based on which you plan your next few steps. If the environment outputs 'Nothing happened', that means the previous action is invalid and you should try more options. 
Reminder: the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. 

There are four thinking levels:
Level 1 - Instinctive Response: Immediate reaction based on intuition, no analysis.
Level 2 - Situational Awareness: Assess current state and available actions before acting.
Level 3 - Experience Integration: Reflect on past actions and outcomes to inform current decisions.
Level 4 - Strategic Planning: Assess the task goal, past lessons, and current state to analyze the future impact of each candidate action and optimize the decision.

At each step, you must first choose an appropriate level of thinking (one of the four levels) to respond based on the given scenario. The chosen level MUST be enclosed within <level> </level> tags.
Next, reason through the problem step-by-step using the chosen thinking level. This reasoning process MUST be enclosed within <think> </think> tags. For Level 1 (Instinctive Response), use the fixed text: "Okay, I think I have finished thinking." For Levels 2-4, provide detailed reasoning as shown in examples.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.

[Output Format] 
Your output must adhere to the following format:
EXAMPLE 1:
<level>1</level>
<think>Okay, I think I have finished thinking.</think>
<action>your_next_action</action>

EXAMPLE 2:
<level>2</level>
<think>
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reasoning: [Choose the best action and explain why]
</think>
<action>your_next_action</action>

EXAMPLE 3:
<level>3</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Reasoning: [Choose the best action based on experience]
</think>
<action>your_next_action</action>

EXAMPLE 4:
<level>4</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Evaluation: [Assess the potential effectiveness of each candidate action]
Reasoning: [Choose the optimal action with strategic reasoning]
</think>
<action>your_next_action</action>

{task_description}
Here is the task history for past 10 steps:
Observation: {current_observation} AVAILABLE ACTIONS: {admissible_actions}

Now it's your turn to generate next step.
""".strip()

'''
{task_description}
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to generate next step response.
'''

ALFWORLD_TEMPLATE_ADA = """
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. After each turn, the environment will give you immediate feedback based on which you plan your next few steps. If the environment outputs 'Nothing happened', that means the previous action is invalid and you should try more options. 
Reminder: the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. 

There are four thinking levels:
Level 1 - Instinctive Response: Immediate reaction based on intuition, no analysis.
Level 2 - Situational Awareness: Assess current state and available actions before acting.
Level 3 - Experience Integration: Reflect on past actions and outcomes to inform current decisions.
Level 4 - Strategic Planning: Assess the task goal, past lessons, and current state to analyze the future impact of each candidate action and optimize the decision.

At each step, you must first choose an appropriate level of thinking (one of the four levels) to respond based on the given scenario. The chosen level MUST be enclosed within <level> </level> tags.
Next, reason through the problem step-by-step using the chosen thinking level. This reasoning process MUST be enclosed within <think> </think> tags. For Level 1 (Instinctive Response), use the fixed text: "Okay, I think I have finished thinking." For Levels 2-4, provide detailed reasoning as shown in examples.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.

[Output Format] 
Your output must adhere to the following format:
EXAMPLE 1:
<level>1</level>
<think>Okay, I think I have finished thinking.</think>
<action>your_next_action</action>

EXAMPLE 2:
<level>2</level>
<think>
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reasoning: [Choose the best action and explain why]
</think>
<action>your_next_action</action>

EXAMPLE 3:
<level>3</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Reasoning: [Choose the best action based on experience]
</think>
<action>your_next_action</action>

EXAMPLE 4:
<level>4</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Evaluation: [Assess the potential effectiveness of each candidate action]
Reasoning: [Choose the optimal action with strategic reasoning]
</think>
<action>your_next_action</action>

{task_description}
Here is the task history for past 10 steps:
{action_history}
Observation: {current_observation} AVAILABLE ACTIONS: {admissible_actions}

Now it's your turn to generate next step.
""".strip()

'''
{task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: 
{action_history}
You are now at step {current_step} and your current observation is: {current_observation} 
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to generate next step response. 
'''