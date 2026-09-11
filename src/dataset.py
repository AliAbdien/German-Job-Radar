"""Builds the instruction-tuning dataset (prompt -> JSON completion pairs)
from the seed postings, and splits into train/eval.

The prompt format matches how the fine-tuned model will actually be called
at inference time (src/infer.py) - same system instruction, same schema
description, same output format. Training on a different prompt shape than
you serve with is a common, avoidable way to quietly tank fine-tuning
results.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from src.schema import JobExtraction

SEED_PATH = Path(__file__).resolve().parent.parent / "data" / "seed_postings.jsonl"
TRAIN_OUT = Path(__file__).resolve().parent.parent / "data" / "train.jsonl"
EVAL_OUT = Path(__file__).resolve().parent.parent / "data" / "eval.jsonl"

SYSTEM_PROMPT = (
    "You extract structured data from German job postings. Given the posting "
    "text, output ONLY a JSON object matching this schema - no explanation, "
    "no markdown fences, just the JSON:\n\n"
    f"{json.dumps(JobExtraction.model_json_schema()['properties'], ensure_ascii=False, indent=2)}"
)


def build_prompt(posting_text: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nJob posting:\n{posting_text}\n\nJSON:"


def load_seed(path: Path = SEED_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_training_record(example: dict) -> dict:
    """One instruction-tuning record: {prompt, completion}."""
    # Re-validate + re-serialize through the schema so the completion is
    # always in canonical field order/formatting, not whatever order the
    # seed dict happened to be written in.
    label = JobExtraction.model_validate(example["label"])
    return {
        "prompt": build_prompt(example["text"]),
        "completion": label.model_dump_json(),
        "raw_text": example["text"],  # kept for eval-time readability, not used in training
    }


def build_and_split(eval_fraction: float = 0.2, seed: int = 13) -> tuple[list[dict], list[dict]]:
    examples = [to_training_record(e) for e in load_seed()]
    rng = random.Random(seed)
    rng.shuffle(examples)
    n_eval = max(1, int(len(examples) * eval_fraction))
    eval_set = examples[:n_eval]
    train_set = examples[n_eval:]
    return train_set, eval_set


def main() -> None:
    train_set, eval_set = build_and_split()
    with TRAIN_OUT.open("w", encoding="utf-8") as f:
        for ex in train_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with EVAL_OUT.open("w", encoding="utf-8") as f:
        for ex in eval_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"train: {len(train_set)} examples -> {TRAIN_OUT}")
    print(f"eval:  {len(eval_set)} examples -> {EVAL_OUT}")


if __name__ == "__main__":
    main()
