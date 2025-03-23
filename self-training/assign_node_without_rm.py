"""
Goal: Traverse the MCTS tree, filter out nodes with high Monte Carlo (MC) scores, and save their logits into a file.
These logits will later be used in the SFT policy model to compute the KL divergence against the reference model.
"""
import argparse
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def load_model(model_path):
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16
    )
    teacher_model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return teacher_model, tokenizer




def read_jsonl(path, threshold):
    prompts = []
    with open(path, 'r') as f:
        for line in f:
            dic = json.loads(line)
            score = dic.get('label', 0)
            if "Python" in dic['input']:
                continue
            if score >= threshold:
                prompts.append(dic['input'])
    return sorted(list(set(prompts)), reverse=True)



def write_jsonl(jsonl_path, prompts, version, task, model, tokenizer, start_idx, end_idx, train_or_eval):
    tensor_dir = f"dataset/temp_tensor/{version}_{task}_{start_idx}_{end_idx}_{train_or_eval}"
    device = "cuda"
    os.makedirs(tensor_dir, exist_ok=True)
    with open(jsonl_path, 'a') as f:
        for idx, prompt in tqdm(enumerate(prompts), total=len(prompts)):
            tensor_file = os.path.join(tensor_dir, f"logits_{idx}.pt")
            inputs = tokenizer([prompt], return_tensors="pt", max_length=4096, padding=True, truncation=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            logits = logits.type(torch.bfloat16)
            torch.save(logits, tensor_file)
            torch.cuda.empty_cache()

            record = {"input": prompt, "logits_path": tensor_file}
            f.write(json.dumps(record) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default="v1")
    parser.add_argument("--task", type=str, default="hrm")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--idx", type=int, default=0)

    args = parser.parse_args()

    version = args.version
    task = args.task
    threshold = float(args.threshold)
    slots = int(args.slots)
    idx = int(args.idx)

    save_train_path = f"dataset/self_training_{version}_policy_without_rm/can_use_{task}_train_with_logits.jsonl"
    save_test_path = f"dataset/self_training_{version}_policy_without_rm/can_use_{task}_eval_with_logits.jsonl"

    original_data_train_path = f"dataset/self_training_{version}_scoring/{task}_train.jsonl"
    original_data_eval_path = f"dataset/self_training_{version}_scoring/{task}_test.jsonl"

    train_prompts = read_jsonl(original_data_train_path, threshold)
    length_train_prompts = len(train_prompts)
    train_start_idx = int(length_train_prompts / slots * idx)
    train_end_idx = int(length_train_prompts / slots * (idx + 1))
    train_prompts = train_prompts[train_start_idx:train_end_idx]

    eval_prompts = read_jsonl(original_data_eval_path, threshold)
    length_eval_prompts = len(eval_prompts)
    eval_start_idx = int(length_eval_prompts / slots * idx)
    eval_end_idx = int(length_eval_prompts / slots * (idx + 1))
    eval_prompts = eval_prompts[eval_start_idx:eval_end_idx]

    print("train prompts:", len(train_prompts), "eval prompts:", len(eval_prompts))

    model, tokenizer = load_model("models/Qwen2.5-Math-7B-Instruct")

    write_jsonl(save_train_path, train_prompts, version, task, model, tokenizer, start_idx=train_start_idx,
                end_idx=train_end_idx, train_or_eval="train")
    write_jsonl(save_test_path, eval_prompts, version, task, model, tokenizer, start_idx=eval_start_idx,
                end_idx=eval_end_idx, train_or_eval="eval")
# CUDA_VISIBLE_DEVICES=0 nohup python self-training/assign_node_without_rm.py --version v1 --task hrm --slots 8 --idx 0 > logs/logits_hrm_v1_0.log &
# CUDA_VISIBLE_DEVICES=1 nohup python self-training/assign_node_without_rm.py --version v1 --task hrm --slots 8 --idx 1 > logs/logits_hrm_v1_1.log &
# CUDA_VISIBLE_DEVICES=2 nohup python self-training/assign_node_without_rm.py --version v1 --task hrm --slots 8 --idx 2 > logs/logits_hrm_v1_2.log &
# CUDA_VISIBLE_DEVICES=3 nohup python self-training/assign_node_without_rm.py --version v1 --task hrm --slots 8 --idx 3 > logs/logits_hrm_v1_3.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/assign_node_without_rm.py --version v1 --task hrm --slots 8 --idx 4 > logs/logits_hrm_v1_4.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/assign_node_without_rm.py --version v1 --task hrm --slots 8 --idx 5 > logs/logits_hrm_v1_5.log &
# CUDA_VISIBLE_DEVICES=6 nohup python self-training/assign_node_without_rm.py --version v1 --task hrm --slots 8 --idx 6 > logs/logits_hrm_v1_6.log &
# CUDA_VISIBLE_DEVICES=7 nohup python self-training/assign_node_without_rm.py --version v1 --task hrm --slots 8 --idx 7 > logs/logits_hrm_v1_7.log &
# _
# CUDA_VISIBLE_DEVICES=0 nohup python self-training/assign_node_without_rm.py --version v1 --task prm --slots 6 --idx 0 > logs/logits_prm_v1_0.log &
# CUDA_VISIBLE_DEVICES=1 nohup python self-training/assign_node_without_rm.py --version v1 --task prm --slots 6 --idx 1 > logs/logits_prm_v1_1.log &
# CUDA_VISIBLE_DEVICES=2 nohup python self-training/assign_node_without_rm.py --version v1 --task prm --slots 6 --idx 2 > logs/logits_prm_v1_2.log &
# CUDA_VISIBLE_DEVICES=3 nohup python self-training/assign_node_without_rm.py --version v1 --task prm --slots 6 --idx 3 > logs/logits_prm_v1_3.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/assign_node_without_rm.py --version v1 --task prm --slots 6 --idx 4 > logs/logits_prm_v1_4.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/assign_node_without_rm.py --version v1 --task prm --slots 6 --idx 5 > logs/logits_prm_v1_5.log &
