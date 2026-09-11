"""LoRA fine-tune a small local instruct model on the job-posting extraction
task. Runs against data/train.jsonl (build it first with `python -m src.dataset`).

Deliberately no 4-bit quantization: at 0.5B-1.5B parameters the base model
fits comfortably in bf16 on a 10GB RTX 3080, and bitsandbytes' Windows
support has historically been unreliable - not worth the risk for a model
this small (see requirements.txt).

Usage:
    python -m src.train_lora --base-model Qwen/Qwen2.5-1.5B-Instruct --epochs 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "lora-adapter"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_tokenized_dataset(records: list[dict], tokenizer, max_length: int = 1024) -> Dataset:
    """Each record becomes one training example: prompt tokens masked out of
    the loss (label=-100), completion tokens are what the model is actually
    trained to predict. Training on the prompt tokens too would waste
    capacity teaching the model to reproduce the (fixed, always-identical)
    schema instructions instead of the extraction itself."""

    def _tokenize(example: dict) -> dict:
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(
            example["completion"] + tokenizer.eos_token, add_special_tokens=False
        )["input_ids"]

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids

        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

    ds = Dataset.from_list(records)
    return ds.map(_tokenize, remove_columns=ds.column_names)


def collate(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(len(x["input_ids"]) for x in batch)

    def pad(seq: list[int], value: int) -> list[int]:
        return seq + [value] * (max_len - len(seq))

    return {
        "input_ids": torch.tensor([pad(x["input_ids"], pad_token_id) for x in batch]),
        "labels": torch.tensor([pad(x["labels"], -100) for x in batch]),
        "attention_mask": torch.tensor([pad(x["attention_mask"], 0) for x in batch]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train-path", default=str(DATA_DIR / "train.jsonl"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device found - this will be extremely slow on CPU.")

    print(f"Loading base model {args.base_model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    records = load_jsonl(Path(args.train_path))
    print(f"Loaded {len(records)} training examples from {args.train_path}")
    train_dataset = build_tokenized_dataset(records, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="epoch",
        bf16=device == "cuda",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
