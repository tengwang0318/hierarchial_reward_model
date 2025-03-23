import pickle
import os
import random

from tree import Node
from envs import *
import matplotlib.pyplot as plt
from collections import defaultdict
from construct_prompt import construct_PRM_HRM_prompt_v2
import argparse
from tqdm import tqdm
import json

import re
def ensure_step_spacing(text):
    """
    确保 '# Step X' 后面有空格或换行，如果没有则添加一个空格
    """
    pattern = r"(# Step \d)(?=[^\s])"  # 匹配 "# Step X" 且后面不是空格或换行
    corrected_text = re.sub(pattern, r"\1 " + "\n", text)  # 在匹配的后面加空格
    return corrected_text


def replace_string(a_list):
    height_2_placeholder = {1: '# Step 2', 2: "# Step 3", 3: "# Step 4", 4: "# Step 5", 5: '# END!'}
    height = len(a_list)
    # if height == 6:
    #     return a_list
    if height != 5:
        a_list[-1] = a_list[-1].replace(height_2_placeholder[height], "\n")
    if height > 1:
        a_list[-2] = a_list[-2].replace(height_2_placeholder[height - 1], "\n")
        a_list[-1] = height_2_placeholder[height - 1] + "\n\n" + a_list[- 1]
    return a_list


def load_all_pickle_file_paths(base_path="dataset/self_training"):
    answers = []
    for filename in os.listdir(base_path):
        answers.append(os.path.join(base_path, filename))
    return answers


def is_leaf_node(node: Node):
    if not node.children:
        return True
    return False


def assign_score_for_every_nodes(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    LEAF_NODE = 0

    def dfs(node: Node):
        if node.score == -1:
            node.score = 0

        if is_leaf_node(node):
            nonlocal LEAF_NODE
            LEAF_NODE += 1
            if node.is_correct:
                node.score = 1
                return 1
            else:
                node.score = 0
                return 0
        else:
            for child in node.children:
                cnt = dfs(child)
                node.score += cnt
            # if node.score!=0:
            #     print(node.score)
            return node.score

    def dfs_v2(node: Node):
        if hasattr(node, 'total'):
            pass
        else:
            node.total = 0

        if is_leaf_node(node):
            node.total = 1
            return 1
        else:
            for child in node.children:
                cnt = dfs_v2(child)
                node.total += cnt
            return node.total

    dfs(data)
    dfs_v2(data)
    queue = [data]
    CORRECT_TOTAL_NUMBERS = 0
    TOTAL_NUMBERS = 0

    score_height_2_node = defaultdict(lambda: defaultdict(list))
    while queue:
        temp_length = len(queue)
        for _ in range(temp_length):
            node = queue.pop(0)
            node.score = node.score / node.total
            if node.height != 0:
                score_height_2_node[node.height][node.score].append(node)

            if node.score == 1:
                CORRECT_TOTAL_NUMBERS += 1
            TOTAL_NUMBERS += 1
            for child in node.children:
                queue.append(child)
    # print(score_height_2_node.keys())

    data.score = 1
    # print(LEAF_NODE)
    # print(TOTAL_NUMBERS)
    #
    # print(CORRECT_TOTAL_NUMBERS)
    # from collections import Counter
    # print(Counter(scores))
    # plot_distribution(scores)
    return score_height_2_node


def write_new_data(root, idx):
    with open(f"dataset/self_training_v1_scoring/{idx}.pkl", 'wb') as f:
        pickle.dump(root, f)


def plot_distribution(data):
    plt.hist(data, bins=10, edgecolor='black')  # Adjust the bins as needed

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Numbers')

    plt.savefig("score_distribution.png")


def sort_node(score_height_2_node):
    answers = []
    for height in score_height_2_node:
        different_score_dict = score_height_2_node[height]
        if len(different_score_dict) > 1:
            min_length = float("inf")
            for score, list_ in different_score_dict.items():
                min_length = min(min_length, len(list_))
            for score, list_ in different_score_dict.items():
                nodes = random.sample(list_, max(min_length, int(len(list_) * 0.1)))
                for node in nodes:
                    if node.previous_answer:
                        # origin_answer = node.previous_answer[:]
                        # origin_length = len(node.previous_answer)

                        node.previous_answer = replace_string(node.previous_answer)

                        prompt = construct_PRM_HRM_prompt_v2(node.question, "".join(node.previous_answer[:-1]),
                                                             node.previous_answer[-1])
                        if "# Step 1" not in prompt:
                            continue

                        # prompt = construct_policy_model_prompt_for_PRM_HRM(node.question, "".join(node.previous_answer))
                        # answers.append(json.dumps(
                        #     {"input": prompt, "label": score, "reasoning_answer": origin_answer,
                        #      "length": len(node.previous_answer), "origin_length": origin_length}))
                        prompt = ensure_step_spacing(prompt)

                        answers.append(json.dumps({"input": prompt, "label": score}))

        else:
            for score, list_ in different_score_dict.items():
                k = min(5, len(list_))
                nodes = random.sample(list_, k)
                for node in nodes:
                    if node.previous_answer:
                        # origin_answer = node.previous_answer[:]
                        # origin_length = len(node.previous_answer)

                        node.previous_answer = replace_string(node.previous_answer)

                        prompt = construct_PRM_HRM_prompt_v2(node.question, "".join(node.previous_answer[:-1]),
                                                             node.previous_answer[-1])
                        if "# Step 1" not in prompt:
                            continue
                        # answers.append(json.dumps(
                        #     {"input": prompt, "label": score, "reasoning_answer": origin_answer,
                        #      "length": len(node.previous_answer), "origin_length": origin_length}))
                        prompt = ensure_step_spacing(prompt)

                        answers.append(json.dumps({"input": prompt, "label": score}))
    return set(answers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="dataset/self_training")
    parser.add_argument("--save_path", type=str, default="dataset/self_training_v1_scoring")

    args = parser.parse_args()
    base_path = args.base_path
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    paths = load_all_pickle_file_paths()
    cnt = 0
    scores = []
    train_idx = int(0.8 * len(paths))
    train_paths = paths[:train_idx]
    test_paths = paths[train_idx:]

    with open(os.path.join(save_path, "prm_train.jsonl"), 'w') as f:
        for path in tqdm(train_paths, total=len(train_paths)):
            score_height_2_node = assign_score_for_every_nodes(path)
            answers = sort_node(score_height_2_node)
            for answer in answers:
                scores.append(json.loads(answer)['label'])
                # json_string = json.dumps(answer)
                f.write(answer + "\n")
                cnt += 1

    with open(os.path.join(save_path, "prm_test.jsonl"), 'w') as f:
        for path in tqdm(test_paths, total=len(test_paths)):
            score_height_2_node = assign_score_for_every_nodes(path)
            answers = sort_node(score_height_2_node)
            for answer in answers:
                scores.append(json.loads(answer)['label'])
                # json_string = json.dumps(answer)
                f.write(answer + "\n")
                cnt += 1
    print(cnt)
    plot_distribution(scores)
    # from collections import Counter
    # print(Counter(scores))
