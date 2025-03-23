"""
PRM and HRM are trained from auto-labeled reasoning process in the PRM800K dataset. And evaluation in PRM800K dataset.
"""
import argparse
import json
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from llm_query import sequential_query, parallel_query
from construct_prompt import construct_policy_model_prompt_for_PRM_HRM, construct_PRM_HRM_prompt_v2
from grading.grader import grade_answer
from tqdm import tqdm
import gc

MAX_HEIGHT = 8


def load_rw_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                               num_labels=1,
                                                               attn_implementation="flash_attention_2",
                                                               torch_dtype=torch.float16,
                                                               )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.config.pad_token_id = model.config.eos_token_id
    device = torch.device("cuda")
    model.to(device)
    return model, tokenizer


def load_test_data(data_path='dataset/prm_dataset/phase1_test.jsonl'):
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


def calculate_PRM_HRM_scores(rm_model, tokenizer, question, N, current_steps, previous_steps="", N_batch=4):
    assert len(current_steps) == N, "The length of current steps must be equal to N"

    device = "cuda"
    current_step_score_pairs = []

    for i in range(0, N, N_batch):
        batch_steps = current_steps[i:i + N_batch]
        prompts = [construct_PRM_HRM_prompt_v2(question, previous_steps, current_step) for current_step in batch_steps]

        inputs = tokenizer(prompts, return_tensors="pt", max_length=4096, padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = rm_model(**inputs)
            logits = outputs.logits

        positive_scores = logits[:, 0].tolist()

        batch_step_score_pairs = [(score, step) for score, step in zip(positive_scores, batch_steps)]
        current_step_score_pairs.extend(batch_step_score_pairs)

    current_step_score_pairs.sort(reverse=True)
    return current_step_score_pairs


def extract_answer(text: str, placeholder="# Answer", end_placeholder='# END!'):
    text = text.lower()
    left_idx = text.rindex(placeholder.lower())
    length = len(placeholder)
    try:
        right_idx = text.rindex(end_placeholder.lower())
    except:
        right_idx = -1
    return text[left_idx + length:right_idx].strip()


def prm_hrm_best_of_n(model_path, host, port, model_name, api_key="", N=2,
                      test_data_path='dataset/prm_dataset/phase1_test.jsonl', task="prm",
                      version='v1'):
    def clear_memory():
        gc.collect()
        torch.cuda.empty_cache()

    def whether_contain_answer(text):
        return "# Answer" in text

    base_path = f"stats/{task}_{version}_{model_name}/"
    os.makedirs(base_path, exist_ok=True)

    saving_filename = os.path.join(base_path, f"{version}_{task}_{N}.txt")


    test_data_pairs = load_test_data(test_data_path)
    correct_cnt = 0
    total_cnt = len(test_data_pairs)

    final_answers = []
    answer_situations = []
    ground_truths = []

    prm_model, prm_tokenizer = load_rw_model(model_path)

    for question, ground_truth in tqdm(test_data_pairs, total=len(test_data_pairs)):
        clear_memory()

        previous_steps = ""
        ground_truths.append(ground_truth)
        found_answer = False
        for height in range(MAX_HEIGHT):
            candidates = []
            policy_model_prompt = construct_policy_model_prompt_for_PRM_HRM(question, previous_steps)
            print(f"---------\nheight{height}: policy model prompt")
            print(policy_model_prompt)
            print(f"---------\nPrompt finish!")
            try:
                intermediate_steps = parallel_query(host, port, model_name, policy_model_prompt, api_key, n=N)
            except Exception as e:
                intermediate_steps = []
                for _ in range(N):
                    try:
                        intermediate_step = sequential_query(host, port, model_name, policy_model_prompt, api_key)
                    except Exception as ee:
                        print(ee)
                        continue
                    intermediate_steps.append(intermediate_step)
            candidates.extend(intermediate_steps)

            if not candidates:
                break

            sorted_current_step_score_pairs = calculate_PRM_HRM_scores(prm_model, prm_tokenizer, question, N,
                                                                       candidates,
                                                                       previous_steps)
            _, current_step = sorted_current_step_score_pairs[0]
            print(f"Current step: {current_step}")
            previous_steps = previous_steps + current_step
            # print(f"Updated previous steps height{height}: ", previous_steps)

            print("---------")

            if whether_contain_answer(current_step):
                found_answer = True
                break
        print("-------最终的推理过程----------")
        print(previous_steps)
        if found_answer:
            answer = extract_answer(previous_steps)
        else:
            answer = "cannot find a correct answer."
        final_answers.append(answer)
        true_or_false = grade_answer(answer, ground_truth)
        answer_situations.append(true_or_false)
        if true_or_false:
            correct_cnt += 1
            print(f"find answer in {question}")

        with open(saving_filename, "a") as f:
            f.write(f"{answer}\t{ground_truth}\t{true_or_false}\n")

    print(f"N = {N}, the correct rate is {correct_cnt / total_cnt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="10086")
    parser.add_argument("--task", type=str, default="prm")
    parser.add_argument("--model_name", type=str, default="policy_model")
    parser.add_argument("--api_key", type=str, default="xxxx")
    parser.add_argument("--N", type=int, default=2)
    parser.add_argument("--test_data_path", type=str,
                        default='dataset/prm_dataset/phase1_test.jsonl')
    parser.add_argument("--version", type=str, default="v1")

    args = parser.parse_args()

    task = args.task
    host = args.host
    port = args.port
    model_name = args.model_name
    api_key = args.api_key
    version = args.version

    N = int(args.N)
    test_data_path = args.test_data_path
    model_path = f"sf_best_model/{task}/{version}"

    if task == "prm":
        prm_hrm_best_of_n(model_path, host, port, model_name, api_key, N, test_data_path, task, version)
    elif task == "hrm":
        prm_hrm_best_of_n(model_path, host, port, model_name, api_key, N, test_data_path, task, version)



# CUDA_VISIBLE_DEVICES=3 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 1   --task prm --model_name hrm_qwen7b_550 > logs/prm_1_v1_hrm_qwen7b_550.log &
# CUDA_VISIBLE_DEVICES=2 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 2   --task prm --model_name hrm_qwen7b_550 > logs/prm_2_v1_hrm_qwen7b_550.log &
# CUDA_VISIBLE_DEVICES=1 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 4   --task prm --model_name hrm_qwen7b_550 > logs/prm_4_v1_hrm_qwen7b_550.log &
# CUDA_VISIBLE_DEVICES=0 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 8   --task prm --model_name hrm_qwen7b_550 > logs/prm_8_v1_hrm_qwen7b_550.log &
# CUDA_VISIBLE_DEVICES=7 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 16  --task prm  --model_name hrm_qwen7b_550  > logs/prm_16_v1_hrm_qwen7b_550.log &
# CUDA_VISIBLE_DEVICES=6 nohup python self-training/best_of_n.py --port 10087 --version v1 --N 24  --task prm  --model_name hrm_qwen7b_550  > logs/prm_24_v1_hrm_qwen7b_550.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10087 --version v1 --N 32  --task prm  --model_name hrm_qwen7b_550  > logs/prm_32_v1_hrm_qwen7b_550.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10087 --version v1 --N 64  --task prm  --model_name hrm_qwen7b_550  > logs/prm_64_v1_hrm_qwen7b_550.log &
# CUDA_VISIBLE_DEVICES=3 nohup python self-training/best_of_n.py --port 10087 --version v1 --N 128  --task prm --model_name hrm_qwen7b_550  > logs/prm_128_v1_hrm_qwen7b_550.log &
# CUDA_VISIBLE_DEVICES=2 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 256  --task prm --model_name hrm_qwen7b_550  > logs/prm_256_v1_hrm_qwen7b_550.log &
# CUDA_VISIBLE_DEVICES=1 nohup python self-training/best_of_n.py --port 10087 --version v1 --N 512  --task prm --model_name hrm_qwen7b_550  > logs/prm_512_v1_hrm_qwen7b_550.log &


# CUDA_VISIBLE_DEVICES=0 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 1   --task hrm --model_name hrm_qwen7b_550 > logs/hrm_1_v1_hrm_qwen7b_550.log &
# CUDA_VISIBLE_DEVICES=1 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 2   --task hrm --model_name hrm_qwen7b_550 > logs/hrm_2_v1_hrm_qwen7b_550.log &
# CUDA_VISIBLE_DEVICES=2 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 4   --task hrm --model_name hrm_qwen7b_550 > logs/hrm_4_v1_hrm_qwen7b_550.log &
# CUDA_VISIBLE_DEVICES=3 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 8   --task hrm --model_name hrm_qwen7b_550 > logs/hrm_8_v1_hrm_qwen7b_550.log &
# CUDA_VISIBLE_DEVICES=0 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 16  --task hrm  --model_name hrm_qwen7b_550 > logs/hrm_16_v1_hrm_qwen7b_550.log &
# CUDA_VISIBLE_DEVICES=1 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 24  --task hrm  --model_name hrm_qwen7b_550  > logs/hrm_24_v1_hrm_qwen7b_550.log &
# CUDA_VISIBLE_DEVICES=2 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 32  --task hrm  --model_name hrm_qwen7b_550  > logs/hrm_32_v1_hrm_qwen7b_550.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 64  --task hrm  --model_name hrm_qwen7b_550  > logs/hrm_64_v1_hrm_qwen7b_550.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 128  --task hrm --model_name hrm_qwen7b_550  > logs/hrm_128_v1_hrm_qwen7b_550.log &
# CUDA_VISIBLE_DEVICES=6 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 256  --task hrm --model_name hrm_qwen7b_550  > logs/hrm_256_v1_hrm_qwen7b_550.log &
# CUDA_VISIBLE_DEVICES=7 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 512  --task hrm --model_name hrm_qwen7b_550  > logs/hrm_512_v1_hrm_qwen7b_550.log &





# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10087 --version v1 --N 1   --task prm --model_name hrm_qwen7b_600 > logs/prm_1_v1_hrm_qwen7b_600.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10087 --version v1 --N 2   --task prm --model_name hrm_qwen7b_600 > logs/prm_2_v1_hrm_qwen7b_600.log &
# CUDA_VISIBLE_DEVICES=6 nohup python self-training/best_of_n.py --port 10087 --version v1 --N 4   --task prm --model_name hrm_qwen7b_600 > logs/prm_4_v1_hrm_qwen7b_600.log &
# CUDA_VISIBLE_DEVICES=7 nohup python self-training/best_of_n.py --port 10087 --version v1 --N 8   --task prm --model_name hrm_qwen7b_600 > logs/prm_8_v1_hrm_qwen7b_600.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 16  --task prm --model_name hrm_qwen_7b  > logs/prm_16_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 24  --task prm --model_name hrm_qwen_7b  > logs/prm_24_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 32  --task prm --model_name hrm_qwen_7b  > logs/prm_32_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 64  --task prm --model_name hrm_qwen_7b  > logs/prm_64_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 128  --task prm --model_name hrm_qwen_7b  > logs/prm_128_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 256  --task prm --model_name hrm_qwen_7b  > logs/prm_256_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 512  --task prm --model_name hrm_qwen_7b  > logs/prm_512_v1_hrm_qwen_7b.log &


# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10087 --version v1 --N 1   --task hrm --model_name hrm_qwen7b_600 > logs/hrm_1_v1_hrm_qwen7b_600.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10087 --version v1 --N 2   --task hrm --model_name hrm_qwen7b_600 > logs/hrm_2_v1_hrm_qwen7b_600.log &
# CUDA_VISIBLE_DEVICES=6 nohup python self-training/best_of_n.py --port 10087 --version v1 --N 4   --task hrm --model_name hrm_qwen7b_600 > logs/hrm_4_v1_hrm_qwen7b_600.log &
# CUDA_VISIBLE_DEVICES=7 nohup python self-training/best_of_n.py --port 10087 --version v1 --N 8   --task hrm --model_name hrm_qwen7b_600 > logs/hrm_8_v1_hrm_qwen7b_600.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 16  --task hrm --model_name hrm_qwen_7b   > logs/hrm_16_v1_hrm_qwen7b_600.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 24  --task hrm --model_name hrm_qwen_7b  > logs/hrm_24_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 32  --task hrm --model_name hrm_qwen_7b  > logs/hrm_32_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 64  --task hrm --model_name hrm_qwen_7b  > logs/hrm_64_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 128  --task hrm --model_name hrm_qwen_7b  > logs/hrm_128_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 256  --task hrm --model_name hrm_qwen_7b  > logs/hrm_256_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 512  --task hrm --model_name hrm_qwen_7b  > logs/hrm_512_v1_hrm_qwen_7b.log &

# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 1  --model_name deepseek --task hrm > logs/hrm_1_v1_deepseek.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 2  --model_name deepseek --task hrm > logs/hrm_2_v1_deepseek.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 4  --model_name deepseek --task hrm > logs/hrm_4_v1_deepseek.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 8  --model_name deepseek --task hrm > logs/hrm_8_v1_deepseek.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 16 --model_name deepseek --task hrm > logs/hrm_16_v1_deepseek.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 24 --model_name deepseek --task hrm > logs/hrm_24_v1_deepseek.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 32 --model_name deepseek --task hrm > logs/hrm_32_v1_deepseek.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 64 --model_name deepseek --task hrm > logs/hrm_64_v1_deepseek.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 128 --model_name deepseek --task hrm > logs/hrm_128_v1_deepseek.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 256 --model_name deepseek --task hrm > logs/hrm_256_v1_deepseek.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 512 --model_name deepseek --task hrm > logs/hrm_512_v1_deepseek.log &


# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 1   --task prm --model_name deepseek > logs/prm_1_v1_deepseek.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 2   --task prm --model_name deepseek > logs/prm_2_v1_deepseek.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 4   --task prm --model_name deepseek > logs/prm_4_v1_deepseek.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 8   --task prm --model_name deepseek > logs/prm_8_v1_deepseek.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 16  --task prm --model_name deepseek  > logs/prm_16_v1_deepseek.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 24  --task prm --model_name deepseek  > logs/prm_24_v1_deepseek.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 32  --task prm --model_name deepseek  > logs/prm_32_v1_deepseek.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 64  --task prm --model_name deepseek  > logs/prm_64_v1_deepseek.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 128  --task prm --model_name deepseek  > logs/prm_128_v1_deepseek.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 256  --task prm --model_name deepseek  > logs/prm_256_v1_deepseek.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 512  --task prm --model_name deepseek  > logs/prm_512_v1_deepseek.log &


# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10087 --version v1 --N 1  --model_name qwen_72b --task hrm > logs/hrm_1_v1_qwen72b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10087 --version v1 --N 2  --model_name qwen_72b --task hrm > logs/hrm_2_v1_qwen72b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 4  --model_name qwen_72b --task hrm > logs/hrm_4_v1_qwen72b.log &
# CUDA_VISIBLE_DEVICES=1 nohup python self-training/best_of_n.py --port 10087 --version v1 --N 8  --model_name qwen_72b --task hrm > logs/hrm_8_v1_qwen72b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 16 --model_name qwen_72b --task hrm > logs/hrm_16_v1_qwen72b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 24 --model_name qwen_72b --task hrm > logs/hrm_24_v1_qwen72b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 32 --model_name qwen_72b --task hrm > logs/hrm_32_v1_qwen72b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10087 --version v1 --N 64 --model_name qwen_72b --task hrm > logs/hrm_64_v1_qwen72b.log &
# CUDA_VISIBLE_DEVICES=0 nohup python self-training/best_of_n.py --port 10087 --version v1 --N 128 --model_name qwen_72b --task hrm > logs/hrm_128_v1_qwen72b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 256 --model_name qwen_72b --task hrm > logs/hrm_256_v1_qwen72b.log &
# CUDA_VISIBLE_DEVICES=3 nohup python self-training/best_of_n.py --port 10087 --version v1 --N 512 --model_name qwen_72b --task hrm > logs/hrm_512_v1_qwen72b.log &


# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 1   --task prm --model_name qwen_72b > logs/prm_1_v1_qwen72b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 2   --task prm --model_name qwen_72b > logs/prm_2_v1_qwen72b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 4   --task prm --model_name qwen_72b > logs/prm_4_v1_qwen72b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 8   --task prm --model_name qwen_72b > logs/prm_8_v1_qwen72b.log &
# CUDA_VISIBLE_DEVICES=2 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 16  --task prm --model_name qwen_72b  > logs/prm_16_v1_qwen72b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 24  --task prm --model_name qwen_72b  > logs/prm_24_v1_qwen72b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 32  --task prm --model_name qwen_72b  > logs/prm_32_v1_qwen72b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 64  --task prm --model_name qwen_72b  > logs/prm_64_v1_qwen72b.log &
# CUDA_VISIBLE_DEVICES=6 nohup python self-training/best_of_n.py --port 10087 --version v1 --N 128  --task prm --model_name qwen_72b  > logs/prm_128_v1_qwen72b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --port 10086 --version v1 --N 256  --task prm --model_name qwen_72b  > logs/prm_256_v1_qwen72b.log &
# CUDA_VISIBLE_DEVICES=1 nohup python self-training/best_of_n.py --port 10087 --version v1 --N 512  --task prm --model_name qwen_72b  > logs/prm_512_v1_qwen72b.log &


# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 1   --task hrm --model_name qwen7b > logs/hrm_1_v1_qwen7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 2   --task hrm --model_name qwen7b > logs/hrm_2_v1_qwen7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 4   --task hrm --model_name qwen7b > logs/hrm_4_v1_qwen7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 8   --task hrm --model_name qwen7b > logs/hrm_8_v1_qwen7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 16  --task hrm --model_name qwen7b  > logs/hrm_16_v1_qwen7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 24  --task hrm --model_name qwen7b  > logs/hrm_24_v1_qwen7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 32  --task hrm --model_name qwen7b  > logs/hrm_32_v1_qwen7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 64  --task hrm --model_name qwen7b  > logs/hrm_64_v1_qwen7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 128  --task hrm --model_name qwen7b  > logs/hrm_128_v1_qwen7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 256  --task hrm --model_name qwen7b  > logs/hrm_256_v1_qwen7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 512  --task hrm --model_name qwen7b  > logs/hrm_512_v1_qwen7b.log &


# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 1   --task prm --model_name hrm_qwen_7b > logs/prm_1_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 2   --task prm --model_name hrm_qwen_7b > logs/prm_2_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 4   --task prm --model_name hrm_qwen_7b > logs/prm_4_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 8   --task prm --model_name hrm_qwen_7b > logs/prm_8_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 16  --task prm --model_name hrm_qwen_7b  > logs/prm_16_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 24  --task prm --model_name hrm_qwen_7b  > logs/prm_24_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 32  --task prm --model_name hrm_qwen_7b  > logs/prm_32_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 64  --task prm --model_name hrm_qwen_7b  > logs/prm_64_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 128  --task prm --model_name hrm_qwen_7b  > logs/prm_128_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 256  --task prm --model_name hrm_qwen_7b  > logs/prm_256_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 512  --task prm --model_name hrm_qwen_7b  > logs/prm_512_v1_hrm_qwen_7b.log &


# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 1   --task hrm --model_name hrm_qwen_7b > logs/hrm_1_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 2   --task hrm --model_name hrm_qwen_7b > logs/hrm_2_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 4   --task hrm --model_name hrm_qwen_7b > logs/hrm_4_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 8   --task hrm --model_name hrm_qwen_7b > logs/hrm_8_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 16  --task hrm --model_name hrm_qwen_7b  > logs/hrm_16_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 24  --task hrm --model_name hrm_qwen_7b  > logs/hrm_24_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 32  --task hrm --model_name hrm_qwen_7b  > logs/hrm_32_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 64  --task hrm --model_name hrm_qwen_7b  > logs/hrm_64_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 128  --task hrm --model_name hrm_qwen_7b  > logs/hrm_128_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 256  --task hrm --model_name hrm_qwen_7b  > logs/hrm_256_v1_hrm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 512  --task hrm --model_name hrm_qwen_7b  > logs/hrm_512_v1_hrm_qwen_7b.log &



# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 1   --task prm --model_name prm_qwen_7b > logs/prm_1_v1_prm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 2   --task prm --model_name prm_qwen_7b > logs/prm_2_v1_prm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 4   --task prm --model_name prm_qwen_7b > logs/prm_4_v1_prm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 8   --task prm --model_name prm_qwen_7b > logs/prm_8_v1_prm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 16  --task prm --model_name prm_qwen_7b  > logs/prm_16_v1_prm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 24  --task prm --model_name prm_qwen_7b  > logs/prm_24_v1_prm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 32  --task prm --model_name prm_qwen_7b  > logs/prm_32_v1_prm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 64  --task prm --model_name prm_qwen_7b  > logs/prm_64_v1_prm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 128  --task prm --model_name prm_qwen_7b  > logs/prm_128_v1_prm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 256  --task prm --model_name prm_qwen_7b  > logs/prm_256_v1_prm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 512  --task prm --model_name prm_qwen_7b  > logs/prm_512_v1_prm_qwen_7b.log &


# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 1   --task hrm --model_name prm_qwen_7b > logs/hrm_1_v1_prm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 2   --task hrm --model_name prm_qwen_7b > logs/hrm_2_v1_prm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 4   --task hrm --model_name prm_qwen_7b > logs/hrm_4_v1_prm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 8   --task hrm --model_name prm_qwen_7b > logs/hrm_8_v1_prm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 16  --task hrm --model_name prm_qwen_7b  > logs/hrm_16_v1_prm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 24  --task hrm --model_name prm_qwen_7b  > logs/hrm_24_v1_prm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 32  --task hrm --model_name prm_qwen_7b  > logs/hrm_32_v1_prm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 64  --task hrm --model_name prm_qwen_7b  > logs/hrm_64_v1_prm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 128  --task hrm --model_name prm_qwen_7b  > logs/hrm_128_v1_prm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 256  --task hrm --model_name prm_qwen_7b  > logs/hrm_256_v1_prm_qwen_7b.log &
# CUDA_VISIBLE_DEVICES=5 nohup python self-training/best_of_n.py --version v1 --N 512  --task hrm --model_name prm_qwen_7b  > logs/hrm_512_v1_prm_qwen_7b.log &

# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 1   --task prm --model_name qwen7b > logs/prm_1_v1_qwen7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 2   --task prm --model_name qwen7b > logs/prm_2_v1_qwen7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 4   --task prm --model_name qwen7b > logs/prm_4_v1_qwen7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 8   --task prm --model_name qwen7b > logs/prm_8_v1_qwen7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 16  --task prm --model_name qwen7b  > logs/prm_16_v1_qwen7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 24  --task prm --model_name qwen7b  > logs/prm_24_v1_qwen7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 32  --task prm --model_name qwen7b  > logs/prm_32_v1_qwen7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 64  --task prm --model_name qwen7b  > logs/prm_64_v1_qwen7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 128  --task prm --model_name qwen7b  > logs/prm_128_v1_qwen7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 256  --task prm --model_name qwen7b  > logs/prm_256_v1_qwen7b.log &
# CUDA_VISIBLE_DEVICES=4 nohup python self-training/best_of_n.py --version v1 --N 512  --task prm --model_name qwen7b  > logs/prm_512_v1_qwen7b.log &


# CUDA_VISIBLE_DEVICES=5 nohup python best_of_n.py --task prm --N 1  --repetition_penalty 1.0 > logs/prm_1.log &
# CUDA_VISIBLE_DEVICES=6 nohup python best_of_n.py --task prm --N 2  --repetition_penalty 1.0 > logs/prm_2.log &
# CUDA_VISIBLE_DEVICES=7 nohup python best_of_n.py --task prm --N 4  --repetition_penalty 1.0 > logs/prm_4.log &
# CUDA_VISIBLE_DEVICES=6 nohup python best_of_n.py --task prm --N 8  --repetition_penalty 1.0 > logs/prm_8.log &
# CUDA_VISIBLE_DEVICES=5 nohup python best_of_n.py --task prm --N 16 --repetition_penalty 1.0  > logs/prm_16.log &
# CUDA_VISIBLE_DEVICES=5 nohup python best_of_n.py --task prm --N 24 --repetition_penalty 1.0  > logs/prm_24.log &


# CUDA_VISIBLE_DEVICES=1 nohup python best_of_n.py --task prm --N 1  --repetition_penalty 1.0 > logs/new_prm_1.log &
# CUDA_VISIBLE_DEVICES=2 nohup python best_of_n.py --task prm --N 2  --repetition_penalty 1.0 > logs/new_prm_2.log &
# CUDA_VISIBLE_DEVICES=3 nohup python best_of_n.py --task prm --N 4  --repetition_penalty 1.0 > logs/new_prm_4.log &
# CUDA_VISIBLE_DEVICES=4 nohup python best_of_n.py --task prm --N 8  --repetition_penalty 1.0 > logs/new_prm_8.log &
# CUDA_VISIBLE_DEVICES=0 nohup python best_of_n.py --task prm --N 16 --repetition_penalty 1.0  > logs/new_prm_16.log &
# CUDA_VISIBLE_DEVICES=6 nohup python best_of_n.py --task prm --N 24 --repetition_penalty 1.0  > logs/new_prm_24.log &


# CUDA_VISIBLE_DEVICES=4 nohup python best_of_n.py --task hrm --N 1  --repetition_penalty 1.0 > logs/hrm_1.log  &
# CUDA_VISIBLE_DEVICES=3 nohup python best_of_n.py --task hrm --N 2  --repetition_penalty 1.0 > logs/hrm_2.log &
# CUDA_VISIBLE_DEVICES=2 nohup python best_of_n.py --task hrm --N 4  --repetition_penalty 1.0 > logs/hrm_4.log &
# CUDA_VISIBLE_DEVICES=1 nohup python best_of_n.py --task hrm --N 8  --repetition_penalty 1.0 > logs/hrm_8.log &
# CUDA_VISIBLE_DEVICES=0 nohup python best_of_n.py --task hrm --N 16 --repetition_penalty 1.0  > logs/hrm_16.log &
# CUDA_VISIBLE_DEVICES=1 nohup python best_of_n.py --task hrm --N 24 --repetition_penalty 1.0  > logs/hrm_24.log &


#
# CUDA_VISIBLE_DEVICES=0 nohup python best_of_n.py --task orm --N 1 --repetition_penalty 1.0 > logs/orm_1_rp_1.0.log  &
# CUDA_VISIBLE_DEVICES=0 nohup python best_of_n.py --task orm --N 1 --repetition_penalty 1.05 > logs/orm_1_rp_1.05.log  &
# CUDA_VISIBLE_DEVICES=1 nohup python best_of_n.py --task orm --N 1 --repetition_penalty 1.1 > logs/orm_1_rp_1.1.log  &
#
# CUDA_VISIBLE_DEVICES=2 nohup python best_of_n.py --task prm --N 1 --repetition_penalty 1.0 > logs/prm_1_rp_1.0.log  &
# CUDA_VISIBLE_DEVICES=3 nohup python best_of_n.py --task prm --N 1 --repetition_penalty 1.05 > logs/prm_1_rp_1.05.log  &
# CUDA_VISIBLE_DEVICES=4 nohup python best_of_n.py --task prm --N 1 --repetition_penalty 1.1 > logs/prm_1_rp_1.1.log  &
#
# CUDA_VISIBLE_DEVICES=5 nohup python best_of_n.py --task hrm --N 1 --repetition_penalty 1.0 > logs/hrm_1_rp_1.0.log  &
# CUDA_VISIBLE_DEVICES=6 nohup python best_of_n.py --task hrm --N 1 --repetition_penalty 1.05 > logs/hrm_1_rp_1.05.log  &
# CUDA_VISIBLE_DEVICES=7 nohup python best_of_n.py --task hrm --N 1 --repetition_penalty 1.1 > logs/hrm_1_rp_1.1.log  &


# CUDA_VISIBLE_DEVICES=4 nohup python best_of_n.py --task impunity_hrm --N 1  --repetition_penalty 1.0 > logs/impunity_hrm_1.log &
# CUDA_VISIBLE_DEVICES=4 nohup python best_of_n.py --task impunity_hrm --N 2  --repetition_penalty 1.0 > logs/impunity_hrm_2.log &
# CUDA_VISIBLE_DEVICES=4 nohup python best_of_n.py --task impunity_hrm --N 4  --repetition_penalty 1.0 > logs/impunity_hrm_4.log &
# CUDA_VISIBLE_DEVICES=4 nohup python best_of_n.py --task impunity_hrm --N 8  --repetition_penalty 1.0 > logs/impunity_hrm_8.log &
# CUDA_VISIBLE_DEVICES=4 nohup python best_of_n.py --task impunity_hrm --N 16 --repetition_penalty 1.0  > logs/impunity_hrm_16.log &
# CUDA_VISIBLE_DEVICES=5 nohup python best_of_n.py --task impunity_hrm --N 24 --repetition_penalty 1.0  > logs/impunity_hrm_24.log &


#

# CUDA_VISIBLE_DEVICES=0 nohup python best_of_n.py --model_name qwen7b_math_instruct --port 10086 --task orm --N 2  --repetition_penalty 1.0 > logs/7b_orm_2.log &
# CUDA_VISIBLE_DEVICES=1 nohup python best_of_n.py --model_name qwen7b_math_instruct --port 10086 --task orm --N 4  --repetition_penalty 1.0 > logs/7b_orm_4.log &
# CUDA_VISIBLE_DEVICES=2 nohup python best_of_n.py --model_name qwen7b_math_instruct --port 10086 --task orm --N 8  --repetition_penalty 1.0 > logs/7b_orm_8.log &
# CUDA_VISIBLE_DEVICES=3 nohup python best_of_n.py --model_name qwen7b_math_instruct --port 10086 --task orm --N 16 --repetition_penalty 1.0  > logs/7b_orm_16.log &
# CUDA_VISIBLE_DEVICES=4 nohup python best_of_n.py --model_name qwen7b_math_instruct --port 10086 --task orm --N 24 --repetition_penalty 1.0  > logs/7b_orm_24.log &
# CUDA_VISIBLE_DEVICES=4 nohup python best_of_n.py --model_name qwen7b_math_instruct --port 10086 --task orm --N 32 --repetition_penalty 1.0  > logs/7b_orm_32.log &


# CUDA_VISIBLE_DEVICES=4 nohup python best_of_n.py --model_name qwen7b_math_instruct --port 10086 --task prm --N 2  --repetition_penalty 1.0 > logs/7b_prm_2.log &
# CUDA_VISIBLE_DEVICES=3 nohup python best_of_n.py --model_name qwen7b_math_instruct --port 10086 --task prm --N 4  --repetition_penalty 1.0 > logs/7b_prm_4.log &
# CUDA_VISIBLE_DEVICES=2 nohup python best_of_n.py --model_name qwen7b_math_instruct --port 10086 --task prm --N 8  --repetition_penalty 1.0 > logs/7b_prm_8.log &
# CUDA_VISIBLE_DEVICES=1 nohup python best_of_n.py --model_name qwen7b_math_instruct --port 10086 --task prm --N 16 --repetition_penalty 1.0  > logs/7b_prm_16.log &
# CUDA_VISIBLE_DEVICES=0 nohup python best_of_n.py --model_name qwen7b_math_instruct --port 10086 --task prm --N 24 --repetition_penalty 1.0  > logs/7b_prm_24.log &
# CUDA_VISIBLE_DEVICES=2 nohup python best_of_n.py --model_name qwen7b_math_instruct --port 10086 --task prm --N 32 --repetition_penalty 1.0  > logs/7b_prm_32.log &


# CUDA_VISIBLE_DEVICES=5 nohup python best_of_n.py --model_name qwen7b_math_instruct_v2 --port 10087 --task hrm --N 2  --repetition_penalty 1.0 > logs/7b_hrm_2.log &
# CUDA_VISIBLE_DEVICES=6 nohup python best_of_n.py --model_name qwen7b_math_instruct_v2 --port 10087 --task hrm --N 4  --repetition_penalty 1.0 > logs/7b_hrm_4.log &
# CUDA_VISIBLE_DEVICES=7 nohup python best_of_n.py --model_name qwen7b_math_instruct_v2 --port 10087 --task hrm --N 8  --repetition_penalty 1.0 > logs/7b_hrm_8.log &
# CUDA_VISIBLE_DEVICES=0 nohup python best_of_n.py --model_name qwen7b_math_instruct_v2 --port 10087 --task hrm --N 16 --repetition_penalty 1.0  > logs/7b_hrm_16.log &
# CUDA_VISIBLE_DEVICES=1 nohup python best_of_n.py --model_name qwen7b_math_instruct_v2 --port 10087 --task hrm --N 24 --repetition_penalty 1.0  > logs/7b_hrm_24.log &
# CUDA_VISIBLE_DEVICES=5 nohup python best_of_n.py --model_name qwen7b_math_instruct_v2 --port 10087 --task hrm --N 32 --repetition_penalty 1.0  > logs/7b_hrm_32.log &


# CUDA_VISIBLE_DEVICES=5 nohup python best_of_n.py --model_name qwen7b_math_instruct_v2 --port 10087 --task impunity_hrm --N 2  --repetition_penalty 1.0 > logs/7b_impunity_hrm_2.log &
# CUDA_VISIBLE_DEVICES=6 nohup python best_of_n.py --model_name qwen7b_math_instruct_v2 --port 10087 --task impunity_hrm --N 4  --repetition_penalty 1.0 > logs/7b_impunity_hrm_4.log &
# CUDA_VISIBLE_DEVICES=7 nohup python best_of_n.py --model_name qwen7b_math_instruct_v2 --port 10087 --task impunity_hrm --N 8  --repetition_penalty 1.0 > logs/7b_impunity_hrm_8.log &
# CUDA_VISIBLE_DEVICES=6 nohup python best_of_n.py --model_name qwen7b_math_instruct_v2 --port 10087 --task impunity_hrm --N 16 --repetition_penalty 1.0  > logs/7b_impunity_hrm_16.log &
# CUDA_VISIBLE_DEVICES=5 nohup python best_of_n.py --model_name qwen7b_math_instruct_v2 --port 10087 --task impunity_hrm --N 24 --repetition_penalty 1.0  > logs/7b_impunity_hrm_24.log &
# CUDA_VISIBLE_DEVICES=6 nohup python best_of_n.py --model_name qwen7b_math_instruct_v2 --port 10087 --task impunity_hrm --N 32 --repetition_penalty 1.0  > logs/7b_impunity_hrm_32.log &
