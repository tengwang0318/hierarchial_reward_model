"""
Goal: assign every node in MCTS a score by using RM(PRM or HRM), but the score might be affected by reward hacking.
So we don't use it and instead use the MC-score.
"""
import json

import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from construct_prm_train_data import load_all_pickle_file_paths
from construct_prompt import construct_PRM_HRM_prompt_v2, construct_policy_model_prompt_for_PRM_HRM
import torch
import argparse
import pickle


def load_rm(path):
    model = AutoModelForSequenceClassification.from_pretrained(path,
                                                               num_labels=1,
                                                               attn_implementation="flash_attention_2",
                                                               torch_dtype=torch.float16,
                                                               )
    tokenizer = AutoTokenizer.from_pretrained(path)
    model.config.pad_token_id = model.config.eos_token_id
    device = torch.device("cuda")
    model.to(device)
    return model, tokenizer


def traverse_tree(path):
    prompts = []
    with open(path, 'rb') as f:
        data = pickle.load(f)

    queue = [data]
    while queue:
        temp_length = len(queue)
        for _ in range(temp_length):
            node = queue.pop(0)
            if queue:
                rm_prompt = construct_PRM_HRM_prompt_v2(node.question, "".join(node.previous_answer[:-1]),
                                                        node.previous_answer[-1])
                policy_model_prompt = construct_policy_model_prompt_for_PRM_HRM(node.question,
                                                                                "".join(node.previous_answer))
                prompts.append(
                    json.dumps({"policy_model_prompt": policy_model_prompt, "reward_model_prompt": rm_prompt}))

            for child in node.children:
                queue.append(child)
    return set(prompts)


def calculate_PRM_HRM_scores(model, tokenizer, prompts, N_batch=16):
    device = "cuda"
    current_step_score_pairs = []

    prompts = [json.loads(prompt) for prompt in prompts]
    N = len(prompts)
    for i in range(0, N, N_batch):
        temp_prompts = prompts[i:i + N_batch]

        rw_prompts = [item['reward_model_prompt'] for item in temp_prompts]

        inputs = tokenizer(rw_prompts, return_tensors="pt", max_length=4096, padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        positive_scores = logits[:, 0].tolist()

        batch_step_score_pairs = [{"score": score, "prompt": prompt['policy_model_prompt']} for score, prompt in
                                  zip(positive_scores, temp_prompts)]
        current_step_score_pairs.extend(batch_step_score_pairs)

    current_step_score_pairs.sort(reverse=True, key=lambda x: x['score'])
    return current_step_score_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="prm")
    parser.add_argument('--version', type=str, default="v1")
    parser.add_argument('--pkl_path', type=str, default="dataset/self_training")
    parser.add_argument('--N_batch', type=int, default=16)

    args = parser.parse_args()
    task = args.task
    version = args.version
    pkl_path = args.pkl_path
    N_batch = args.N_batch

    final_pairs = []

    rm_path = f"sf_best_model/{task}/{version}"
    data_path = f"dataset/self_training_{version}_policy/"
    os.makedirs(data_path, exist_ok=True)

    model, tokenizer = load_rm(rm_path)

    pkl_paths = load_all_pickle_file_paths(pkl_path)

    for pkl_path in tqdm.tqdm(pkl_paths, total=len(pkl_paths)):
        prompts = traverse_tree(pkl_path)
        current_pair = calculate_PRM_HRM_scores(model, tokenizer, prompts, N_batch)
        final_pairs.extend(current_pair)
    with open(os.path.join(data_path, f"{task}.jsonl"), "w") as f:
        for pair in final_pairs:
            f.write(json.dumps(pair) + "\n")
# CUDA_VISIBLE_DEVICES=0 nohup python self-training/assign_node_rm.py > logs/prm_v1_.log &
# CUDA_VISIBLE_DEVICES=0 nohup python self-training/assign_node_rm.py --task hrm --version v1 > logs/hrm_v1_.log &
