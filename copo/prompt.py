"""
CoPo Thinking Level Prompts
Defines different thinking level templates for generating alternative thinking patterns
"""

THINK_MODE_2 = """Given the current state and available actions, assess the situation before acting:

{history}

Now use Level 2 thinking (Situational Awareness) to analyze the current state and choose the best action "{action}". 

<level>2</level>
<think>
Current state: [Analyze the current environment state]
Available actions: [What actions are possible right now]
Reasoning: [Choose the best action and explain why]
</think>
<action>{action}</action>"""

THINK_MODE_3 = """Given the current state and available actions, reflect on past experiences to inform your decision:

{history}

Now use Level 3 thinking (Experience Integration) to reflect on past actions and outcomes before choosing action "{action}".

<level>3</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Analyze the current environment state]
Available actions: [What actions are possible right now]
Reflection: [How effective were recent actions, what was learned]
Reasoning: [Choose the best action based on experience]
</think>
<action>{action}</action>"""

THINK_MODE_4 = """Given the current state and available actions, strategically plan and evaluate each option:

{history}

Now use Level 4 thinking (Strategic Planning) to assess the task goal, past lessons, and current state to analyze the future impact of action "{action}".

<level>4</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Analyze the current environment state]
Available actions: [What actions are possible right now]
Reflection: [How effective were recent actions, what was learned]
Evaluation: [Assess the potential effectiveness of each candidate action]
Reasoning: [Choose the optimal action with strategic reasoning]
</think>
<action>{action}</action>"""
