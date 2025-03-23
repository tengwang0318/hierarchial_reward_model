"""
Introduce log KL divergence to sft policy model, detailed analysis is shown in the paper.
"""
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
import os
from datetime import datetime
import argparse
from functools import partial
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader

accelerator = Accelerator()
device = accelerator.device


class SimpleDataCollator:
    def __init__(self):
        pass

    def __call__(self, features):
        batch = {
            "input_ids": torch.stack(
                [torch.tensor(f["input_ids"]) if not isinstance(f["input_ids"], torch.Tensor) else f["input_ids"] for f
                 in features]),
            "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) if not isinstance(f["attention_mask"],
                                                                                               torch.Tensor) else f[
                "attention_mask"] for f in features]),
            "labels": torch.stack(
                [torch.tensor(f["labels"]) if not isinstance(f["labels"], torch.Tensor) else f["labels"] for f in
                 features]),
            "logits_path": [f["logits_path"] for f in features],
        }
        return batch


def choose_task(data_path, task: str = 'orm'):
    train_data_path = os.path.join(data_path, f'can_use_{task}_train_with_logits.jsonl')
    eval_data_path = os.path.join(data_path, f'can_use_{task}_eval_with_logits.jsonl')

    print(f"train data path: {train_data_path}")
    print(f"eval data path: {eval_data_path}")

    train_dataset = load_dataset('json', data_files=train_data_path)
    eval_dataset = load_dataset('json', data_files=eval_data_path)

    train_dataset = train_dataset['train'].shuffle()
    eval_dataset = eval_dataset['train'].shuffle()

    eval_dataset = eval_dataset.select(list(range(int(len(eval_dataset) * 0.4))))
    # train_dataset = train_dataset.select([0, 1, 2, 3, 4, 5, 6, 7])
    # eval_dataset = eval_dataset.select([0, 1, 2, 3, 4, 5, 6, 7, ])
    return train_dataset, eval_dataset


def pad_or_truncate(logits, target_length=4096, padding_value=-1e9):
    # 确保输入的 logits 是可处理的形状
    if len(logits.size()) == 4:  # 例如 [1, 1, seq_len, 152064]
        logits = logits.squeeze(0)  # -> [1, seq_len, 152064]
    if len(logits.size()) == 3:  # 例如 [1, seq_len, 152064]
        logits = logits.squeeze(0)  # -> [seq_len, 152064]
    elif len(logits.size()) != 2:  # 其他意外形状
        raise ValueError(f"Unexpected logits shape: {logits.size()}")

    seq_len = logits.size(0)
    if seq_len < target_length:
        # 填充 0 到目标长度
        padding = (0, 0, 0, target_length - seq_len)  # (left, right, top, bottom)
        logits = F.pad(logits, padding, value=padding_value)
    elif seq_len > target_length:
        # 截断到目标长度
        logits = logits[:target_length, :]
    return logits.unsqueeze(0)  # 恢复为 [1, target_length, 152064]


def pad_or_truncate_for_multiple_batch(logits, target_length=4096, padding_value=-1e9):
    # [batch size, seq len, vocab size]
    seq_len = logits.size(1)
    if seq_len < target_length:
        # 填充 0 到目标长度
        padding = (0, 0, 0, target_length - seq_len)  # (left, right, top, bottom)
        logits = F.pad(logits, padding, value=padding_value)
    elif seq_len > target_length:
        # 截断到目标长度
        logits = logits[:, :target_length, :]
    return logits


def kl_divergence_loss_for_batch_one(student_logits, teacher_logits):
    # print("Original teacher_logits shape:", teacher_logits.size())
    # print("Original student_logits shape:", student_logits.size())

    # 处理 teacher_logits 的维度（从 4D 到 3D）
    if len(teacher_logits.size()) == 4:
        teacher_logits = teacher_logits.squeeze(1)  # [1, 1, seq_len, 152064] -> [1, seq_len, 152064]
    elif len(teacher_logits.size()) != 3:
        raise ValueError(f"Unexpected teacher_logits shape: {teacher_logits.size()}")
    # 获取序列长度
    student_seq_len = student_logits.size(1)  # 应为 4096
    teacher_seq_len = teacher_logits.size(1)  # 如 1599 without padding

    target_length = teacher_seq_len

    if student_seq_len < target_length:
        raise ValueError(f"The size of student seq length is wrong! The student seq length should be {student_seq_len}")
    elif student_seq_len > target_length:
        student_logits = student_logits[:, :target_length, :]

    kl_loss = F.kl_div(F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction='batchmean')
    return kl_loss


def kl_divergence_loss_for_batch_more(student_logits, teacher_logits):
    # bs = student_logits.size(0)
    kl_loss = F.kl_div(F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1),
                       reduction='batchmean')
    return kl_loss


def compute_loss(model, inputs, return_outputs=False):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    labels = inputs["labels"]
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    teacher_model_logits_paths = inputs['logits_path']

    teacher_logits_list = []
    seq_lengths = []
    for path in teacher_model_logits_paths:
        logits = torch.load(path, weights_only=True, map_location=device).to(device)
        seq_len = logits.size(1) if len(logits.size()) == 3 else logits.size(0)
        seq_lengths.append(seq_len)

    max_length = max(seq_lengths)

    for path in teacher_model_logits_paths:
        logits = torch.load(path, weights_only=True, map_location=device).to(device)
        adjusted_logits = pad_or_truncate(logits, target_length=max_length)
        teacher_logits_list.append(adjusted_logits)

    teacher_logits = torch.stack(teacher_logits_list)
    logits = outputs.logits
    loss = outputs.loss
    path = f"plots/data/{args.task}_{args.kl}"
    os.makedirs(path, exist_ok=True)

    if len(teacher_model_logits_paths) == 1:
        # print("training logit size()", logits.size())
        # print("training teacher size()", teacher_logits.size())
        kl_loss = kl_divergence_loss_for_batch_one(logits, teacher_logits)
        kl_loss_log = torch.log(1 + kl_loss)
        # kl_loss_log = kl_loss

        with open(os.path.join(path, "train.txt"), "a") as f:
            f.write(f"{kl_loss_log}/{loss}\n")
    else:
        # print(f"seq_lengths = {seq_lengths}")
        # print("Teacher logits size before squeeze:", teacher_logits.size())
        teacher_logits = teacher_logits.squeeze(1)
        # print("Teacher logits size after squeeze", teacher_logits.size())
        # print(f"max_length={max_length}")
        # print(f"logits size is {logits.size()}")
        logits = pad_or_truncate_for_multiple_batch(logits, max_length)
        logits = logits.squeeze(0)
        # print(f"logits size after padding is {logits.size()}")
        kl_loss = kl_divergence_loss_for_batch_more(logits, teacher_logits)
        # print(f"KL loss is {kl_loss}")
        kl_loss_log = torch.log(1 + kl_loss)
        # kl_loss_log = kl_loss
        with open(os.path.join(path, "eval.txt"), "a") as f:
            f.write(f"{kl_loss_log}/{loss}\n")

    total_loss = loss + kl_weight * kl_loss_log
    # print(f"Loss: {loss}, kl loss log: {kl_loss_log}, total loss: {total_loss}")
    return (total_loss, outputs) if return_outputs else total_loss


def compute_loss_kl_0(model, inputs, return_outputs=False):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    labels = inputs["labels"]
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    loss = outputs.loss
    path = f"plots/data/{args.task}_{args.kl}"
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "eval.txt"), "a") as f:
        f.write(f"{loss}\n")

    total_loss = loss

    return (total_loss, outputs) if return_outputs else total_loss


def preprocess_function(examples, tokenizer):
    inputs = examples["input"]
    logits_path = examples['logits_path']
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=4096, return_tensors="pt")
    input_ids = model_inputs["input_ids"].squeeze(0)
    labels = input_ids.clone()
    labels = torch.cat([labels[1:], torch.tensor([tokenizer.pad_token_id], dtype=labels.dtype)])
    model_inputs["input_ids"] = input_ids
    model_inputs["labels"] = labels
    model_inputs["attention_mask"] = model_inputs["attention_mask"].squeeze(0)
    model_inputs["logits_path"] = logits_path
    return model_inputs


class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # 将输入数据转移到模型所在的设备
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss, None, None)


def train(model_path, train_dataset, eval_dataset, idx=1, task='prm',
          time=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 attn_implementation="flash_attention_2",
                                                 torch_dtype=torch.float16,
                                                 )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.config.pad_token_id = model.config.eos_token_id
    model_name = model_path.split('/')[-1]
    output_dir = f"outputs/self_training_v{idx}_policy_model/{task}/{model_name}/KL_{kl_weight}_{time}"

    EVAL_STEP = 50

    training_args = TrainingArguments(

        output_dir=output_dir,
        learning_rate=2e-6,
        # max_grad_norm=0.1,
        # warmup_steps=20,
        warmup_steps=50,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        eval_strategy="steps",
        eval_steps=EVAL_STEP,
        save_steps=EVAL_STEP,
        save_strategy="steps",
        logging_steps=EVAL_STEP,
        logging_dir=f"./self-train_logs_v{idx}/{task}/{model_name}/{time}",

        load_best_model_at_end=True,
        fp16=True,
        gradient_accumulation_steps=16,

        greater_is_better=False,
        save_total_limit=5,
        metric_for_best_model="eval_loss",

        deepspeed="deepspeed_config/policy_model_72b.json",
        report_to="tensorboard",

        # gradient_checkpointing=True
    )

    data_collator = SimpleDataCollator()

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    if kl_weight > 0.0001:
        trainer.compute_loss = compute_loss
    else:
        trainer.compute_loss = compute_loss_kl_0

    trainer.train()
    return trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset/self_training_v1_policy_without_rm")
    parser.add_argument("--model_path", type=str,
                        default="models/Qwen2.5-Math-7B-Instruct")

    parser.add_argument("--task", type=str, default='prm')
    parser.add_argument("--idx", type=int, default=1)
    parser.add_argument("--kl", type=float, default=0.1)

    args = parser.parse_args()
    kl_weight = args.kl
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, legacy=False)

    train_dataset, eval_dataset = choose_task(args.data_path, args.task)
    # print("Train dataset sample 0:", train_dataset[0])
    # print("Keys in sample 0:", train_dataset[0].keys())
    preprocess_with_tokenizer = partial(preprocess_function, tokenizer=tokenizer)

    train_dataset = train_dataset.map(preprocess_with_tokenizer, batched=False, keep_in_memory=True)
    eval_dataset = eval_dataset.map(preprocess_with_tokenizer, batched=False, keep_in_memory=True)

    train(model_path=args.model_path, train_dataset=train_dataset, idx=args.idx, eval_dataset=eval_dataset,
          task=args.task)
#  nohup accelerate launch --config_file accelerate_config/8gpus.yaml --gpu_ids 0,1,2,3,4,5,6,7 self-training/sft_policy_model.py --task hrm --idx 1 > logs/self_training_v1_policy_model_hrm.log &
# nohup accelerate launch --config_file accelerate_config/6gpus.yaml --gpu_ids 0,1,2,3,4,5 self-training/sft_policy_model.py --task prm --idx 1 > logs/self_training_v1_policy_model_prm.log &
# nohup accelerate launch --config_file accelerate_config/6gpus.yaml --gpu_ids 0,1,2,3,4,5 self-training/sft_policy_model.py --task hrm --idx 1 > logs/self_training_v1_policy_model_hrm_kl_version.log &
# nohup accelerate launch --config_file accelerate_config/4gpus.yaml --main_process_port 29501 --gpu_ids 0,1,2,3 self-training/sft_policy_model.py --task hrm --idx 1 --kl 10 > logs/self_training_v1_policy_model_hrm_kl_10.log &
# nohup accelerate launch --config_file accelerate_config/4gpus.yaml --main_process_port 29502 --gpu_ids 4,5,6,7 self-training/sft_policy_model.py --task hrm --idx 1 --kl 0.5 > logs/self_training_v1_policy_model_hrm_kl_05.log &
# nohup accelerate launch --config_file accelerate_config/8gpus.yaml --main_process_port 29502 --gpu_ids 0,1,2,3,4,5,6,7 self-training/sft_policy_model.py --task hrm --idx 1 --kl 5 > logs/self_training_v1_policy_model_hrm_kl_5.log &
# nohup accelerate launch --config_file accelerate_config/8gpus.yaml --main_process_port 29500 --gpu_ids 0,1,2,3,4,5,6,7 self-training/sft_policy_model.py --task hrm --idx 1 --kl 1 > logs/self_training_v1_policy_model_hrm_kl_1.log &
# nohup accelerate launch --config_file accelerate_config/6gpus.yaml --main_process_port 29501 --gpu_ids 0,1,2,3,4,5 self-training/sft_policy_model.py --task prm --idx 1 --kl 1 > logs/self_training_v1_policy_model_prm_kl_1.log &

# nohup accelerate launch --config_file accelerate_config/6gpus.yaml --main_process_port 29500 --gpu_ids 0,1,2,3,4,5 self-training/sft_policy_model.py --task hrm --idx 1 --kl 0.5 > logs/self_training_v1_policy_model_hrm_kl_05.log &
# nohup accelerate launch --config_file accelerate_config/6gpus.yaml --main_process_port 29501 --gpu_ids 0,1,2,3,4,5 self-training/sft_policy_model.py --task prm --idx 1 --kl 0.5 > logs/self_training_v1_policy_model_prm_kl_05.log &
# nohup accelerate launch --config_file accelerate_config/8gpus.yaml --main_process_port 29500 --gpu_ids 0,1,2,3,4,5,6,7 self-training/sft_policy_model.py --task hrm --idx 1 --kl 0 > logs/self_training_v1_policy_model_prm_kl_0.log &


# nohup accelerate launch --config_file accelerate_config/8gpus.yaml --main_process_port 29500 --gpu_ids 0,1,2,3,4,5,6,7 self-training/sft_policy_model.py --task hrm --idx 1 --kl 0.001 > logs/self_training_v1_policy_model_hrm_kl_05_8gpus.log &


# nohup accelerate launch --config_file accelerate_config/4gpus.yaml --main_process_port 29501 --gpu_ids 0,1,2,3 self-training/sft_policy_model.py --task hrm --idx 1 --kl 10 > logs/self_training_v1_policy_model_hrm_kl_10.log &
# nohup accelerate launch --config_file accelerate_config/4gpus.yaml --main_process_port 29502 --gpu_ids 4,5,6,7 self-training/sft_policy_model.py --task hrm --idx 1 --kl 0.5 > logs/self_training_v1_policy_model_hrm_kl_05.log &
# nohup accelerate launch --config_file accelerate_config/4gpus.yaml --main_process_port 29503 --gpu_ids 1,2,3,4 self-training/sft_policy_model.py --task hrm --idx 1 --kl 0.001 > logs/self_training_v1_policy_model_hrm_kl_001.log &
