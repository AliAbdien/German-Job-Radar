"""Field-level evaluation: how well does a model's extraction match the gold
label on the held-out eval set?

Different field types need different comparison logic - naive exact-string-match
on the whole JSON object would call two semantically-identical outputs "wrong"
over field ordering or minor wording. So each field type gets the comparison
that's actually meaningful for it:
  - enums/booleans (seniority, remote_policy, salary.stated, visa_sponsorship_mentioned): exact match
  - free text (job_title, company, location): case-insensitive containment
    (either string contains the other) - "ML Engineer" vs "Machine Learning
    Engineer" should count as close enough, exact string equality would not
  - lists (required_skills, nice_to_have_skills, required_languages): set
    precision/recall/F1, case-insensitive
  - salary numbers: exact match (None counts as a value - both being None
    when salary wasn't stated is correct, one stating a number when the
    other says unstated is wrong)

Usage:
    python -m src.eval --eval-path data/eval.jsonl --adapter models/lora-adapter
    python -m src.eval --eval-path data/eval.jsonl                    # base model only, no adapter
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.schema import JobExtraction

# `src.infer` pulls in torch/transformers/peft - a heavy dependency the
# scoring logic below doesn't need. Imported lazily inside `evaluate()` so
# `pytest tests/test_eval.py` (which only exercises the pure scoring
# functions) works in a plain CI environment with no GPU stack installed.

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _text_match(pred: str | None, gold: str | None) -> bool:
    if pred is None and gold is None:
        return True
    if pred is None or gold is None:
        return False
    p, g = pred.lower().strip(), gold.lower().strip()
    return p in g or g in p


def _list_f1(pred: list[str], gold: list[str]) -> float:
    pred_set = {s.lower().strip() for s in pred}
    gold_set = {s.lower().strip() for s in gold}
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_one(pred: JobExtraction, gold: JobExtraction) -> dict[str, float]:
    return {
        "job_title_match": float(_text_match(pred.job_title, gold.job_title)),
        "company_match": float(_text_match(pred.company, gold.company)),
        "location_match": float(_text_match(pred.location, gold.location)),
        "seniority_exact": float(pred.seniority == gold.seniority),
        "remote_policy_exact": float(pred.remote_policy == gold.remote_policy),
        "salary_stated_exact": float(pred.salary.stated == gold.salary.stated),
        "salary_min_exact": float(pred.salary.min_eur == gold.salary.min_eur),
        "salary_max_exact": float(pred.salary.max_eur == gold.salary.max_eur),
        "visa_exact": float(pred.visa_sponsorship_mentioned == gold.visa_sponsorship_mentioned),
        "required_skills_f1": _list_f1(pred.required_skills, gold.required_skills),
        "nice_to_have_skills_f1": _list_f1(pred.nice_to_have_skills, gold.nice_to_have_skills),
        "required_languages_f1": _list_f1(pred.required_languages, gold.required_languages),
    }


def evaluate(
    eval_path: Path,
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path: str | None = None,
) -> dict:
    from src.infer import extract

    with eval_path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    per_example_scores = []
    failures = 0
    for record in records:
        gold = JobExtraction.model_validate_json(record["completion"])
        try:
            pred = extract(record["raw_text"], base_model=base_model, adapter_path=adapter_path)
        except Exception as exc:  # noqa: BLE001 - a malformed generation is itself a real eval outcome
            failures += 1
            print(f"  [FAILED TO PARSE] {exc}")
            continue
        per_example_scores.append(score_one(pred, gold))

    n = len(per_example_scores)
    aggregate = {}
    if n > 0:
        keys = per_example_scores[0].keys()
        aggregate = {k: round(sum(s[k] for s in per_example_scores) / n, 3) for k in keys}
    aggregate["n_evaluated"] = n
    aggregate["n_parse_failures"] = failures
    aggregate["parse_success_rate"] = round(n / len(records), 3) if records else 0.0
    return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", default=str(DATA_DIR / "eval.jsonl"))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default=None, help="Path to a LoRA adapter; omit to evaluate the base model.")
    args = parser.parse_args()

    label = f"{args.base_model} + LoRA({args.adapter})" if args.adapter else f"{args.base_model} (base)"
    print(f"Evaluating: {label}")
    results = evaluate(Path(args.eval_path), base_model=args.base_model, adapter_path=args.adapter)
    print(json.dumps(results, indent=2))
