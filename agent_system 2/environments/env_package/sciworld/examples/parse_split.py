from scienceworld import ScienceWorldEnv
import random
env = ScienceWorldEnv()
taskNames = env.get_task_names()
train_tasks = [0,22,9,20,25,26,27,5,6,7,10,2,3,17,19,12,14,15,23]
test_ood_tasks = [1,21,28,8,11,4,18,13,16,24]

# seed
random.seed(42)

# L0
L0_idx = {
    "train": [],
    "test": []
}
for tid in train_tasks + test_ood_tasks:
    task_num = tid
    taskName = taskNames[task_num]
    env.load(taskName, 0)
    print("Variations (train): " + str(env.get_variations_train()))
    print("Variations (test): " + str(env.get_variations_test()))

    train_vars = env.get_variations_train()
    random.shuffle(train_vars)
    # test_vars = env.get_variations_test()
    test_vars = env.get_variations_dev()
    random.shuffle(test_vars)

    for i, var in enumerate(train_vars):
        L0_idx["train"].append((task_num, var))
        # if i % 2 == 0:
        #     L0_idx["test"].append((task_num, var))
    for i, var in enumerate(test_vars):
        L0_idx["test"].append((task_num, var))

L1_idx = {
    "train": [],
    "test": []
}
for tid in train_tasks + test_ood_tasks:
    task_num = tid
    taskName = taskNames[task_num]
    env.load(taskName, 0)

    print("Variations (train): " + str(env.get_variations_train()))
    print("Variations (test): " + str(env.get_variations_test()))

    train_vars = env.get_variations_train()
    random.shuffle(train_vars)
    test_vars = env.get_variations_test()
    random.shuffle(test_vars)

    for i, var in enumerate(train_vars):
        L1_idx["train"].append((task_num, var))
    
    for i, var in enumerate(test_vars):
        L1_idx["test"].append((task_num, var))

print("L0_idx:", L0_idx)
print("L1_idx:", L1_idx)

L2_idx = {
    "train": [],
    "test": []
}
for tid in train_tasks:
    task_num = tid
    taskName = taskNames[task_num]
    env.load(taskName, 0)

    print("Variations (train): " + str(env.get_variations_train()))
    print("Variations (test): " + str(env.get_variations_test()))

    train_vars = env.get_variations_train()
    random.shuffle(train_vars)

    for i, var in enumerate(train_vars):
        L2_idx["train"].append((task_num, var))
    
for tid in test_ood_tasks:
    task_num = tid
    taskName = taskNames[task_num]
    env.load(taskName, 0)

    print("Variations (train): " + str(env.get_variations_train()))
    print("Variations (test): " + str(env.get_variations_test()))

    test_vars = env.get_variations_test()
    random.shuffle(test_vars)

    for i, var in enumerate(test_vars):
        L2_idx["test"].append((task_num, var))


import json

file_parent = "../variations_idx/"
with open(file_parent + "L0_idx.json", "w") as f:
    json.dump(L0_idx, f, indent=4)
with open(file_parent + "L1_idx.json", "w") as f:
    json.dump(L1_idx, f, indent=4)
with open(file_parent + "L2_idx.json", "w") as f:
    json.dump(L2_idx, f, indent=4)