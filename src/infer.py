"""Run extraction on a job posting with either the base model or the LoRA
fine-tuned adapter - same code path for both, so the eval comparison
(src/eval.py) is apples-to-apples.

Usage:
    python -m src.infer --text "..." --adapter models/lora-adapter
    python -m src.infer --text "..."                     # base model only
"""
from __future__ import annotations

import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset import build_prompt
from src.schema import JobExtraction

_loaded_models: dict[str, tuple] = {}  # cache: key -> (model, tokenizer)


def load_model(base_model: str, adapter_path: str | None = None):
    key = f"{base_model}::{adapter_path or 'base'}"
    if key in _loaded_models:
        return _loaded_models[key]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    _loaded_models[key] = (model, tokenizer)
    return model, tokenizer


def _extract_json(text: str) -> dict:
    """Models occasionally wrap JSON in markdown fences or trail extra text
    despite instructions not to - pull out the first {...} block rather than
    assuming the whole completion is clean JSON."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    return json.loads(match.group(0))


def extract(
    posting_text: str,
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path: str | None = None,
    max_new_tokens: int = 400,
) -> JobExtraction:
    model, tokenizer = load_model(base_model, adapter_path)
    prompt = build_prompt(posting_text)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    parsed = _extract_json(generated)
    return JobExtraction.model_validate(parsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default=None)
    args = parser.parse_args()

    result = extract(args.text, base_model=args.base_model, adapter_path=args.adapter)
    print(result.model_dump_json(indent=2))
