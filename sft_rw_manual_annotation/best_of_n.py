import argparse

import json
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from llm_query import orm_query, prm_query, orm_parallel_query, prm_parallel_query
from construct_dataset.construct_prompt import construct_ORM_prompt, construct_PRM_HRM_prompt_v2, \
    construct_policy_model_prompt_for_ORM, construct_policy_model_prompt_for_PRM_HRM
import torch.nn.functional as F
from grading.grader import grade_answer
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt

MAX_HEIGHT = 100


def load_rw_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                               num_labels=2,
                                                               attn_implementation="flash_attention_2",
                                                               torch_dtype=torch.float16,
                                                               )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.config.pad_token_id = model.config.eos_token_id
    device = torch.device("cuda")
    model.to(device)
    return model, tokenizer


def load_test_data(data_path='dataset/prm_dataset/phase_test.jsonl'):
    question_answer_pairs = []
    with open(data_path) as f:
        for line in f.readlines():
            dic = json.loads(line)
            if dic['label']['finish_reason'] != "solution":
                continue

            question = dic['question']['problem']
            ground_truth = dic['question']['ground_truth_answer']
            question_answer_pairs.append((question, ground_truth))

    return question_answer_pairs[:1]


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()


def calculate_ORM_scores(model, tokenizer, question, answers, N, N_batch=4):
    assert len(answers) == N, "The length of answers must be equal to N"
    clear_memory()
    device = "cuda"

    answer_score_pairs = []

    for i in range(0, N, N_batch):
        batch_answers = answers[i:i + N_batch]
        prompts = [construct_ORM_prompt(question, answer) for answer in batch_answers]

        inputs = tokenizer(prompts, return_tensors="pt", max_length=4096, padding="max_length", truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)

        positive_scores = probs[:, 1].tolist()
        batch_answer_score_pairs = [(score, answer) for score, answer in zip(positive_scores, batch_answers)]
        answer_score_pairs.extend(batch_answer_score_pairs)
        print(f"length of answer pairs: {len(answer_score_pairs)}")
    answer_score_pairs.sort(reverse=True)
    return answer_score_pairs


def calculate_PRM_HRM_scores(model, tokenizer, question, N, current_steps, previous_steps="", N_batch=2):
    assert len(current_steps) == N, "The length of current steps must be equal to N"

    device = "cuda"
    current_step_score_pairs = []

    for i in range(0, N, N_batch):
        batch_steps = current_steps[i:i + N_batch]
        prompts = [construct_PRM_HRM_prompt_v2(question, previous_steps, current_step) for current_step in batch_steps]

        inputs = tokenizer(prompts, return_tensors="pt", max_length=4096, padding="max_length", truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)

        positive_scores = probs[:, 1].tolist()
        batch_step_score_pairs = [(score, step) for score, step in zip(positive_scores, batch_steps)]
        current_step_score_pairs.extend(batch_step_score_pairs)

        del inputs, outputs, logits, probs
        torch.cuda.empty_cache()

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


def orm_best_of_n(model_path, host, port, model_name, api_key="", N=2,
                  test_data_path='dataset/prm_dataset/phase_test.jsonl', repetition_penalty=1.0):
    test_data_pairs = load_test_data(test_data_path)
    correct_cnt = 0
    total_cnt = len(test_data_pairs)

    final_answers = []
    answer_situations = []
    ground_truths = []

    orm_model, orm_tokenizer = load_rw_model(model_path)

    for idx, (question, ground_truth) in enumerate(tqdm(test_data_pairs)):
        ground_truths.append(ground_truth)
        policy_model_prompt = construct_policy_model_prompt_for_ORM(question)

        print(policy_model_prompt)
        try:
            candidates = orm_parallel_query(host, port, model_name, policy_model_prompt, api_key, n=N)
        except Exception as e:
            candidates = []

            for _ in range(N):
                answer = orm_query(host, port, model_name, policy_model_prompt, api_key)
                print(f"for question {idx}, the {_} try for current answer: {answer}\n\n----")
                candidates.append(answer)
        for candidate in candidates:
            print(f"------{candidate}------\n\n")
        sorted_answer_score_pairs = calculate_ORM_scores(orm_model, orm_tokenizer, question, candidates, N)

        _, answer = sorted_answer_score_pairs[0]

        try:
            final_answer = extract_answer(answer)
        except Exception as e:
            print(e)
            final_answer = "cannot find a correct answer."

        final_answers.append(final_answer)

        true_or_false = grade_answer(final_answer, ground_truth)
        answer_situations.append(true_or_false)

        if true_or_false:
            correct_cnt += 1
            print(f"find right answer for question {idx}")

    base_path = f"stats/orm/"
    os.makedirs(base_path, exist_ok=True)

    with open(os.path.join(base_path, f"{model_name}_N_{N}_repetition_penalty_{repetition_penalty}.txt"), "w") as f:
        for final_answer, ground_truth, answer_situation in zip(final_answers, ground_truths, answer_situations):
            f.write(f"{final_answer}\t{ground_truth}\t{answer_situation}\n")
    print(f"N = {N}, the correct rate is {correct_cnt / total_cnt}")


def prm_hrm_best_of_n(model_path, host, port, model_name, api_key="", N=2,
                      test_data_path='dataset/prm_dataset/phase_test.jsonl', task="prm", repetition_penalty=1.0):
    def clear_memory():
        gc.collect()
        torch.cuda.empty_cache()

    def whether_contain_answer(text):
        return "# Answer" in text

    base_path = f"stats/{task}/"
    os.makedirs(base_path, exist_ok=True)

    saving_filename = os.path.join(base_path, f"{model_name}_N_{N}_repetition_penalty_{repetition_penalty}_{task}.txt")

    test_data_pairs = load_test_data(test_data_path)
    correct_cnt = 0
    total_cnt = len(test_data_pairs)

    final_answers = []
    answer_situations = []
    ground_truths = []

    prm_model, prm_tokenizer = load_rw_model(model_path)

    for question, ground_truth in tqdm(test_data_pairs):
        clear_memory()

        previous_steps = ""
        ground_truths.append(ground_truth)
        found_answer = False
        for height in range(MAX_HEIGHT):
            policy_model_prompt = construct_policy_model_prompt_for_PRM_HRM(question, previous_steps)
            print(f"---------\nheight{height}: policy model prompt")
            print(policy_model_prompt)
            print(f"---------\nPrompt finish!")

            try:
                candidates = prm_parallel_query(host, port, model_name, policy_model_prompt, api_key, n=N)
            except Exception as e:
                candidates = []

                for _ in range(N):
                    try:
                        intermediate_step = prm_query(host, port, model_name, policy_model_prompt, api_key)
                        # intermediate_step = intermediate_step.replace("\n\n", " ").replace("\n", " ")
                        candidates.append(intermediate_step)
                    except Exception as e:
                        print(e)
                        break
            if not candidates:
                break

            sorted_current_step_score_pairs = calculate_PRM_HRM_scores(prm_model, prm_tokenizer, question, N,
                                                                       candidates,
                                                                       previous_steps)
            _, current_step = sorted_current_step_score_pairs[0]

            # current_step = current_step.strip()

            print(f"current step height{height}: ", current_step)

            previous_steps = previous_steps + current_step
            print(f"Updated previous steps height{height}: ", previous_steps)

            print("---------")

            if whether_contain_answer(current_step):
                found_answer = True
                break
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
    # base_path = f"stats/{task}/"
    # os.makedirs(base_path, exist_ok=True)
    #
    # with open(os.path.join(base_path, f"N_{N}_repetition_penalty_{repetition_penalty}_new_prm.txt"), "w") as f:
    #     for final_answer, ground_truth, answer_situation in zip(final_answers, ground_truths, answer_situations):
    #         f.write(f"{final_answer}\t{ground_truth}\t{answer_situation}\n")
    # print(f"N = {N}, the correct rate is {correct_cnt / total_cnt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="10086")
    parser.add_argument("--task", type=str, default="orm")
    parser.add_argument("--model_name", type=str, default="policy_model")
    parser.add_argument("--api_key", type=str, default="xxxx")
    parser.add_argument("--N", type=int, default=2)
    parser.add_argument("--test_data_path", type=str, default='dataset/prm_dataset/phase1_test.jsonl')
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    args = parser.parse_args()
    model_path = f"best_model/{args.task}"
    task = args.task
    host = args.host
    port = args.port
    model_name = args.model_name
    api_key = args.api_key
    repetition_penalty = args.repetition_penalty

    N = int(args.N)
    test_data_path = args.test_data_path

    if task == "orm":
        orm_best_of_n(model_path, host, port, model_name, api_key, N, test_data_path, repetition_penalty)
    elif task == "prm":
        prm_hrm_best_of_n(model_path, host, port, model_name, api_key, N, test_data_path, task, repetition_penalty)
    elif task == "hrm":
        prm_hrm_best_of_n(model_path, host, port, model_name, api_key, N, test_data_path, task, repetition_penalty)
