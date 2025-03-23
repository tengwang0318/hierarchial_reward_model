from datasets import load_dataset
from rm_dataset_construction_phase1 import phase1_ORM_dataset, phase1_PRM_dataset, phase1_HRM_dataset, \
    split_correct_and_wrong_step_phase1, parse_jsonl
from rm_dataset_construction_phase2 import phase2_ORM_dataset, phase2_PRM_dataset, phase2_HRM_dataset, \
    split_correct_and_wrong_step_phase2
from construct_prompt import construct_ORM_prompt, construct_PRM_HRM_prompt
import json
import random
import os
from transformers import AutoTokenizer

RATIO = 0.8


def train_eval_split_rm(a_list):
    random.shuffle(a_list)
    length = len(a_list)
    train_idx = int(length * RATIO)
    return a_list[:train_idx], a_list[train_idx:]


def write_jsonl(a_list, path):
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen2.5-1.5B-math")
    seen = set()
    with open(path, 'w') as f:
        for json_ in a_list:
            input_ = json_['input']
            length = len(tokenizer(input_, truncation=False)['input_ids'])
            if length > 4096 or input_ in seen:
                continue
            seen.add(input_)
            f.write(json.dumps(json_) + "\n")


def orm_formatter(a_list):
    datas = []
    for a_dict in a_list:
        question = a_dict['question']
        for correct_answer in a_dict['correct']:
            prompt = construct_ORM_prompt(question, correct_answer)
            datas.append({'input': prompt, 'label': 1})
        for incorrect_answer in a_dict['incorrect']:
            prompt = construct_ORM_prompt(question, incorrect_answer)
            datas.append({'input': prompt, 'label': 0})
    return datas



def prm_and_hrm_formatter(a_list):
    datas = []
    for a_dict in a_list:
        question = a_dict['question']
        for correct_answer in a_dict['correct']:
            prompt = construct_PRM_HRM_prompt(question, correct_answer)
            datas.append({'input': prompt, 'label': 1})
        for incorrect_answer in a_dict['incorrect']:
            prompt = construct_PRM_HRM_prompt(question, incorrect_answer)
            datas.append({'input': prompt, 'label': 0})
    return datas


def construct_phase1_train_eval(origin_data_location='dataset/prm_dataset/phase1_train.jsonl',
                                base_path='dataset/phase1/'):
    dicts = parse_jsonl(origin_data_location)
    questions, positive_list, negative_list, neutral_list, chosen_completion_list, status_list = split_correct_and_wrong_step_phase1(
        dicts)

    # ORM
    orm_list = phase1_ORM_dataset(questions, positive_list, negative_list, neutral_list, chosen_completion_list)
    orm_train_list, orm_eval_list = train_eval_split_rm(orm_list)
    orm_train_data = orm_formatter(orm_train_list)
    orm_eval_data = orm_formatter(orm_eval_list)

    orm_path = os.path.join(base_path, 'orm')
    os.makedirs(orm_path, exist_ok=True)

    orm_train_path = os.path.join(orm_path, 'train.jsonl')
    orm_eval_path = os.path.join(orm_path, 'eval.jsonl')

    write_jsonl(orm_train_data, orm_train_path)
    write_jsonl(orm_eval_data, orm_eval_path)

    # PRM
    print("PRM: ")
    prm_list = phase1_PRM_dataset(questions, positive_list, negative_list, neutral_list, chosen_completion_list)
    prm_train_list, prm_eval_list = train_eval_split_rm(prm_list)
    prm_train_data = prm_and_hrm_formatter(prm_train_list)
    prm_eval_data = prm_and_hrm_formatter(prm_eval_list)

    prm_path = os.path.join(base_path, 'prm')
    os.makedirs(prm_path, exist_ok=True)

    prm_train_path = os.path.join(prm_path, 'train.jsonl')
    prm_eval_path = os.path.join(prm_path, 'eval.jsonl')

    write_jsonl(prm_train_data, prm_train_path)
    write_jsonl(prm_eval_data, prm_eval_path)

    # HRM
    print("HRM: ")
    hrm_list = phase1_HRM_dataset(questions, positive_list, negative_list, neutral_list, chosen_completion_list)
    hrm_train_list, hrm_eval_list = train_eval_split_rm(hrm_list)
    hrm_train_data = prm_and_hrm_formatter(hrm_train_list)
    hrm_eval_data = prm_and_hrm_formatter(hrm_eval_list)

    hrm_path = os.path.join(base_path, 'hrm')
    os.makedirs(hrm_path, exist_ok=True)

    hrm_train_path = os.path.join(hrm_path, 'train.jsonl')
    hrm_eval_path = os.path.join(hrm_path, 'eval.jsonl')

    write_jsonl(hrm_train_data, hrm_train_path)
    write_jsonl(hrm_eval_data, hrm_eval_path)



if __name__ == '__main__':
    construct_phase1_train_eval('dataset/prm_dataset/phase1_train.jsonl',
                                base_path='dataset/phase1/')
