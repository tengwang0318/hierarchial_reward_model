import os
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch


def load_data_for_task(task: str = 'orm', version="v1"):
    if task in ["orm", "prm", "hrm"]:
        eval_data_path = f"dataset/self_training_{version}_scoring/{task}_test.jsonl"
    else:
        raise ValueError("The task should be either 'orm', 'prm', or 'hrm'.")

    print(f"eval data path: {eval_data_path}")

    eval_dataset = load_dataset('json', data_files=eval_data_path)
    eval_dataset = eval_dataset['train']

    return eval_dataset


def load_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                               num_labels=1,
                                                               attn_implementation="flash_attention_2",
                                                               torch_dtype=torch.float16,
                                                               )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer


def preprocess_function(examples, tokenizer):
    inputs = examples["input"]
    labels = examples["label"]
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=4096)
    model_inputs["labels"] = labels
    return model_inputs


def load_all_model_paths(base_path):
    paths = []
    for folder_name in os.listdir(base_path):
        temp_folder_path = os.path.join(base_path, folder_name)
        for model_filename in os.listdir(temp_folder_path):
            current_path = os.path.join(temp_folder_path, model_filename)
            paths.append(current_path)
    return paths


def main(task, version):
    _, tokenizer = load_model("models/Qwen2.5-1.5B-math")

    save_best_path = f"sf_best_model/{task}/{version}"
    os.makedirs(save_best_path, exist_ok=True)

    # Load data
    eval_dataset = load_data_for_task(task)

    # Load all models
    base_path = f"outputs/self_training_{version}/{task}/Qwen2.5-1.5B-math"

    all_model_paths = load_all_model_paths(base_path)
    print(f"all_model_paths: {all_model_paths}")

    # best_f1 = -1
    best_eval_loss = 1000
    best_model_path = None

    # Tokenize data
    preprocess_with_tokenizer = lambda examples: preprocess_function(examples, tokenizer)
    tokenized_eval_dataset = eval_dataset.map(preprocess_with_tokenizer, batched=True)

    for model_path in all_model_paths:
        print(f"Evaluating model: {model_path}")

        model, tokenizer = load_model(model_path)

        temp_model_path = model_path.replace(task, "temp" + task)
        training_args = TrainingArguments(
            output_dir=f"{temp_model_path}",
            per_device_eval_batch_size=32,
            per_device_train_batch_size=1,
            logging_dir=f"./logs/best/{task}",
            deepspeed="deepspeed_config/rm_config_stage2_py.json",
            fp16=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=tokenized_eval_dataset,
            tokenizer=tokenizer,
        )

        metrics = trainer.evaluate()
        eval_loss = metrics["eval_loss"]

        print(f"Model: {model_path}, Loss: {eval_loss}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_model_path = model_path

        if best_model_path:
            print(f"Best model found: {best_model_path} with F1: {best_eval_loss}")
        # model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
        # tokenizer = AutoTokenizer.from_pretrained(best_model_path)
        # model.save_pretrained(save_best_path)
        # tokenizer.save_pretrained(save_best_path)
        else:
            print("No models evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="orm")
    parser.add_argument("--version", type=str, default="v1")
    args = parser.parse_args()

    task = args.task
    version = args.version
    main(task, version)
