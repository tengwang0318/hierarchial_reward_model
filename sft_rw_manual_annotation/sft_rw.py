from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import numpy as np
import argparse
from functools import partial
import torch



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def choose_task(task: str = 'orm'):
    if task == "orm" or task == "prm" or task == "hrm" or task == "impunity_hrm":
        data_path = f"dataset/phase1/{task}"
    else:
        raise ValueError("The task should be either 'orm' or 'prm' or 'hrm'.")
    train_data_path = os.path.join(data_path, 'train.jsonl')
    eval_data_path = os.path.join(data_path, 'eval.jsonl')

    print(f"train data path: {train_data_path}")
    print(f"eval data path: {eval_data_path}")

    train_dataset = load_dataset('json', data_files=train_data_path)
    eval_dataset = load_dataset('json', data_files=eval_data_path)

    train_dataset = train_dataset['train'].shuffle()
    eval_dataset = eval_dataset['train']

    # train_dataset = train_dataset.select([0, 1, 2, 3, 4, 5, 6, 7])
    # eval_dataset = eval_dataset.select([0, 1, 2, 3, 4, 5, 6, 7, ])

    return train_dataset, eval_dataset


def train(model_path, train_dataset, eval_dataset, task='orm', time=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                               num_labels=2,
                                                               attn_implementation="flash_attention_2",
                                                               torch_dtype=torch.float16,
                                                               )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.config.pad_token_id = model.config.eos_token_id
    model_name = model_path.split('/')[-1]
    output_dir = f"outputs/{task}/{model_name}/{time}"

    EVAL_STEP = 20

    training_args = TrainingArguments(

        output_dir=output_dir,
        learning_rate=1e-5,
        max_grad_norm=0.01,
        # warmup_steps=20,
        warmup_steps=10,
        per_device_train_batch_size=5,
        per_device_eval_batch_size=64,
        num_train_epochs=40,
        eval_strategy="steps",
        eval_steps=EVAL_STEP,
        save_steps=EVAL_STEP,
        save_strategy="steps",
        logging_steps=EVAL_STEP,
        logging_dir=f"./logs/{task}/{model_name}/{time}",

        load_best_model_at_end=True,
        fp16=True,
        gradient_accumulation_steps=32,


        greater_is_better=False,
        save_total_limit=1,
        metric_for_best_model="eval_loss",

        deepspeed="deepspeed_config/RM_config_stage2.json",
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer


def preprocess_function(examples, tokenizer):
    inputs = examples["input"]
    labels = examples["label"]
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=4096)
    model_inputs["labels"] = labels
    return model_inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/Qwen2.5-1.5B-math")
    parser.add_argument("--task", type=str, default='orm')

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, legacy=False)

    train_dataset, eval_dataset = choose_task(args.task)

    preprocess_with_tokenizer = partial(preprocess_function, tokenizer=tokenizer)

    train_dataset = train_dataset.map(preprocess_with_tokenizer, batched=True)
    eval_dataset = eval_dataset.map(preprocess_with_tokenizer, batched=True)

    train(model_path=args.model_path, train_dataset= train_dataset, eval_dataset=eval_dataset, task=args.task)
# nohup accelerate launch --config_file accelerate_config/1gpu.yaml --gpu_ids 0 fine_tune/train.py --task orm > logs/new_orm.log &
# nohup accelerate launch --config_file accelerate_config/3gpus.yaml --gpu_ids 2,3,4 --main_process_port 29501 fine_tune/train.py --task prm > logs/new_prm.log &
# nohup accelerate launch --config_file accelerate_config/4gpus.yaml --gpu_ids 1,5,6,7 --main_process_port 29502 fine_tune/train.py --task hrm > logs/new_hrm.log &
# nohup accelerate launch --config_file accelerate_config/3gpus.yaml --gpu_ids 0,1,2  fine_tune/train.py --task prm > logs/new_prm_v2.log &
