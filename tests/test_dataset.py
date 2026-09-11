from src.dataset import build_and_split, build_prompt, load_seed, to_training_record
from src.schema import JobExtraction


def test_seed_loads_and_validates():
    examples = load_seed()
    assert len(examples) >= 20
    for ex in examples:
        JobExtraction.model_validate(ex["label"])  # raises on invalid


def test_training_record_prompt_contains_posting_text():
    examples = load_seed()
    record = to_training_record(examples[0])
    assert examples[0]["text"] in record["prompt"]
    assert "prompt" in record and "completion" in record


def test_training_record_completion_is_valid_json():
    examples = load_seed()
    record = to_training_record(examples[0])
    JobExtraction.model_validate_json(record["completion"])


def test_build_prompt_is_deterministic():
    assert build_prompt("some posting") == build_prompt("some posting")


def test_train_eval_split_no_overlap_and_covers_all():
    train, eval_set = build_and_split(eval_fraction=0.2)
    train_texts = {r["raw_text"] for r in train}
    eval_texts = {r["raw_text"] for r in eval_set}
    assert not (train_texts & eval_texts)
    assert len(train) + len(eval_set) == len(load_seed())
