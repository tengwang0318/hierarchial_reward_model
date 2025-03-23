from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import argparse
from functools import partial
import torch


def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.squeeze(-1)  # Remove extra dimensions
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    return {"mse": mse, "mae": mae}


def choose_task(data_path, task: str = 'orm'):
    train_data_path = os.path.join(data_path, f'{task}_train.jsonl')
    eval_data_path = os.path.join(data_path, f'{task}_test.jsonl')

    print(f"train data path: {train_data_path}")
    print(f"eval data path: {eval_data_path}")

    train_dataset = load_dataset('json', data_files=train_data_path)
    eval_dataset = load_dataset('json', data_files=eval_data_path)

    train_dataset = train_dataset['train'].shuffle()
    eval_dataset = eval_dataset['train']

    # train_dataset = train_dataset.select([0, 1, 2, 3, 4, 5, 6, 7])
    # eval_dataset = eval_dataset.select([0, 1, 2, 3, 4, 5, 6, 7, ])

    return train_dataset, eval_dataset


def train(model_path, train_dataset, eval_dataset, idx=1, task='orm',
          time=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                               num_labels=1,
                                                               attn_implementation="flash_attention_2",
                                                               torch_dtype=torch.float16,
                                                               )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.config.pad_token_id = model.config.eos_token_id
    model_name = model_path.split('/')[-1]
    output_dir = f"outputs/self_training_v{idx}/{task}/{model_name}/{time}"

    EVAL_STEP = 50

    training_args = TrainingArguments(

        output_dir=output_dir,
        learning_rate=1e-5,
        max_grad_norm=0.01,
        # warmup_steps=20,
        warmup_steps=10,
        per_device_train_batch_size=5,
        per_device_eval_batch_size=64,
        num_train_epochs=20,
        eval_strategy="steps",
        eval_steps=EVAL_STEP,
        save_steps=EVAL_STEP,
        save_strategy="steps",
        logging_steps=EVAL_STEP,
        logging_dir=f"./self-train_logs_v{idx}/{task}/{model_name}/{time}",

        load_best_model_at_end=True,
        fp16=True,
        # tf32=True,
        gradient_accumulation_steps=16,

        # greater_is_better=False,
        # save_total_limit=3,
        # metric_for_best_model="eval_loss",

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
    parser.add_argument("--data_path", type=str, default="dataset/self_training_v1_scoring")
    parser.add_argument("--model_path", type=str, default="models/Qwen2.5-1.5B-math")
    parser.add_argument("--task", type=str, default='orm')
    parser.add_argument("--idx", type=int, default=1)

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, legacy=False)

    train_dataset, eval_dataset = choose_task(args.data_path, args.task)

    preprocess_with_tokenizer = partial(preprocess_function, tokenizer=tokenizer)

    train_dataset = train_dataset.map(preprocess_with_tokenizer, batched=True)
    eval_dataset = eval_dataset.map(preprocess_with_tokenizer, batched=True)

    train(model_path=args.model_path, train_dataset=train_dataset, idx=args.idx, eval_dataset=eval_dataset,
          task=args.task)
#  nohup accelerate launch --config_file accelerate_config/8gpus.yaml --gpu_ids 0,1,2,3,4,5,6,7 self-training/sft_rm.py --task prm --idx 1 > logs/self_training_prm.log &
# nohup accelerate launch --config_file accelerate_config/6gpus.yaml --gpu_ids 0,1,2,3,4,5 self-training/sft_rm.py --task prm --idx 1 > logs/self_training_prm.log &
# nohup accelerate launch --config_file accelerate_config/8gpus.yaml --gpu_ids 0,1,2,3,4,5,6,7 self-training/sft_rm.py --task hrm --idx 1 > logs/self_training_hrm_v1.log &