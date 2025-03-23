import os

from tree import Node
from llm_query import parallel_query, sequential_query
from construct_prompt import construct_policy_model_prompt_for_PRM_HRM
from tqdm import tqdm
import argparse
import json
import pickle
from envs import *


def load_train_data(data_path='dataset/prm_dataset/phase1_train.jsonl'):
    question_answer_pairs = []
    with open(data_path) as f:
        for line in f.readlines():
            dic = json.loads(line)
            if dic['label']['finish_reason'] != "solution":
                continue

            question = dic['question']['problem']
            ground_truth = dic['question']['ground_truth_answer']
            question_answer_pairs.append((question, ground_truth))

    return question_answer_pairs


def mcts_process_for_single_question(question, ground_truth, host, port, model_name, api_key, idx, version, task):
    root = Node(question=question)
    queue = [root]
    LLM_query_times = 0
    for height in range(1, MAX_HEIGHT):
        current_length = len(queue)

        for _ in range(current_length):
            current_node = queue.pop(0)

            temp_children_list = []
            previous_answer = current_node.previous_answer[:]

            if not previous_answer:
                prompt = construct_policy_model_prompt_for_PRM_HRM(question, None)
            else:
                prompt = construct_policy_model_prompt_for_PRM_HRM(question, "\n".join(previous_answer))
            try:
                LLM_query_times += 1
                intermediate_steps = parallel_query(host, port, model_name, prompt, api_key)
                # print("成功并行运算")
            except Exception as e:
                print(e)
                print("一个一个算")
                intermediate_steps = []
                for cnt_child in range(NUMBER_OF_CHILDREN):
                    try:
                        intermediate_step = sequential_query(host, port, model_name, prompt, api_key)
                    except Exception as e:
                        print(e)
                        intermediate_step = "IT SHOULD STOP!"
                    intermediate_steps.append(intermediate_step)

            for intermediate_step in intermediate_steps:
                node = Node(question=question, parent=current_node, height=height)
                previous_answer = current_node.previous_answer[:]
                node.set_previous_answer(previous_answer)

                if intermediate_step == "IT SHOULD STOP!":
                    node.should_stop = True

                node.previous_answer.append(intermediate_step)
                if whether_contain_answer(intermediate_step):
                    answer = extract_answer(intermediate_step)
                    node.have_the_answer(answer, ground_truth)
                elif node.should_stop:
                    pass
                else:
                    queue.append(node)

                temp_children_list.append(node)
                # print("------")
                # print("\n".join(node.previous_answer))
            # for cnt_child in range(NUMBER_OF_CHILDREN):
            #
            #     node = Node(question=question, parent=current_node, height=height)
            #
            #     previous_answer = current_node.previous_answer[:]
            #     node.set_previous_answer(previous_answer)
            #     # print("Current height is ", height)
            #     if not previous_answer:
            #         prompt = construct_policy_model_prompt_for_PRM_HRM(question, None)
            #     else:
            #         prompt = construct_policy_model_prompt_for_PRM_HRM(question, "\n".join(previous_answer))
            #     # print(prompt)
            #     # print("----prompt end-----")
            #     try:
            #         intermediate_step = sequential_query(host, port, model_name, prompt, api_key)
            #     except Exception as e:
            #         print(e)
            #         node.should_stop = True
            #         intermediate_step = "IT SHOULD STOP!"
            #     node.previous_answer.append(intermediate_step)
            #     # print(intermediate_step)
            #     if whether_contain_answer(intermediate_step):
            #         answer = extract_answer(intermediate_step)
            #         node.have_the_answer(answer, ground_truth)
            #     elif node.should_stop:
            #         pass
            #     else:
            #         queue.append(node)
            #
            #     # print("----------------")
            #     #
            #     # print("\n".join(node.previous_answer))
            #     # print("----------------\n\n")
            #     temp_children_list.append(node)
            current_node.add_children(temp_children_list)

        if not queue:
            break
    print(f"For question {idx}, it accesses {LLM_query_times} times.")
    os.makedirs(f"dataset/self_training_{version}_{task}", exist_ok=True)
    with open(f"dataset/self_training_{version}_{task}/{idx}.pkl", 'wb') as f:
        pickle.dump(root, f)


def mcts_process_for_special_range(start_idx, end_idx, host, port, model_name, api_key,
                                   data_path='dataset/prm_dataset/phase1_train.jsonl', version='v1', task='prm'):
    question_answer_pairs = load_train_data(data_path=data_path)
    for idx in tqdm(range(start_idx, end_idx), total=end_idx - start_idx):
        question, ground_truth = question_answer_pairs[idx]
        mcts_process_for_single_question(question, ground_truth, host, port, model_name, api_key, idx, version, task)


def extract_answer(text: str, placeholder="# Answer", end_placeholder='# END!'):
    text = text.lower()
    left_idx = text.rindex(placeholder.lower())
    length = len(placeholder)
    try:
        right_idx = text.rindex(end_placeholder.lower())
    except:
        right_idx = -1
    return text[left_idx + length:right_idx].strip()


def whether_contain_answer(text):
    return "# Answer" in text


if __name__ == '__main__':
    # 一共808个数据
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="10086")
    parser.add_argument("--model_name", type=str, default="policy_model")
    parser.add_argument("--api_key", type=str, default="xxxx")
    parser.add_argument("--train_data_path", type=str, default='dataset/prm_dataset/phase1_train.jsonl')
    parser.add_argument("--start_idx", type=str, default="0")
    parser.add_argument("--end_idx", type=str, default="1")
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--task", type=str, default="prm")
    args = parser.parse_args()

    host = args.host
    port = args.port
    model_name = args.model_name
    api_key = args.api_key
    train_data_path = args.train_data_path
    start_idx = int(args.start_idx)
    end_idx = int(args.end_idx)
    version = args.version
    task = args.task

    mcts_process_for_special_range(start_idx, end_idx, host, port, model_name, api_key, train_data_path, version, task)
#  nohup python self-training/generate_mcts.py --start_idx 0 --end_idx 400 > logs/self_train_0_400.log &
#  nohup python self-training/generate_mcts.py --port 10087 --model_name policy_model_v2 --start_idx 400 --end_idx 809 > logs/self_train_400_809.log &

# nohup python self-training/generate_mcts.py --port 10088 --model_name hrm_v1_policy_first --version v2 --task hrm --start_idx 0 --end_idx 200  > logs/self_train_hrm_0_200.log &
# nohup python self-training/generate_mcts.py --port 10088 --model_name hrm_v1_policy_first --version v2 --task hrm --start_idx 200 --end_idx 400  > logs/self_train_hrm_200_400.log &
# nohup python self-training/generate_mcts.py --port 10089 --model_name hrm_v1_policy_two --version v2 --task hrm --start_idx 400 --end_idx 600  > logs/self_train_hrm_400_600.log &
# nohup python self-training/generate_mcts.py --port 10089 --model_name hrm_v1_policy_two --version v2 --task hrm --start_idx 600 --end_idx 809  > logs/self_train_hrm_600_809.log &

