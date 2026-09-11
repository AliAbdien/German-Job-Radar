from src.eval import _list_f1, _text_match, score_one
from src.schema import JobExtraction


def test_text_match_both_none():
    assert _text_match(None, None) is True


def test_text_match_one_none():
    assert _text_match("Berlin", None) is False
    assert _text_match(None, "Berlin") is False


def test_text_match_substring_case_insensitive():
    assert _text_match("ML Engineer", "Machine Learning Engineer, ML Engineer") is True
    assert _text_match("berlin", "Berlin") is True


def test_text_match_mismatch():
    assert _text_match("Berlin", "Munich") is False


def test_list_f1_perfect():
    assert _list_f1(["Python", "SQL"], ["python", "sql"]) == 1.0


def test_list_f1_empty_both():
    assert _list_f1([], []) == 1.0


def test_list_f1_empty_one_side():
    assert _list_f1(["Python"], []) == 0.0
    assert _list_f1([], ["Python"]) == 0.0


def test_list_f1_partial_overlap():
    # 1 of 2 predicted correct (precision 0.5), 1 of 3 gold covered (recall 1/3)
    f1 = _list_f1(["Python", "Rust"], ["Python", "SQL", "Docker"])
    assert 0.3 < f1 < 0.45


def _make(**overrides) -> JobExtraction:
    base = dict(
        job_title="Junior Data Scientist",
        company="Beispiel GmbH",
        seniority="junior",
        required_skills=["Python", "SQL"],
        nice_to_have_skills=["Docker"],
        salary={"min_eur": 48000, "max_eur": 58000, "stated": True},
        location="München",
        remote_policy="hybrid",
        required_languages=["German (C1)"],
        visa_sponsorship_mentioned=False,
    )
    base.update(overrides)
    return JobExtraction.model_validate(base)


def test_score_one_perfect_match():
    gold = _make()
    pred = _make()
    scores = score_one(pred, gold)
    assert all(v == 1.0 for v in scores.values())


def test_score_one_salary_mismatch():
    gold = _make()
    pred = _make(salary={"min_eur": None, "max_eur": None, "stated": False})
    scores = score_one(pred, gold)
    assert scores["salary_stated_exact"] == 0.0
    assert scores["salary_min_exact"] == 0.0
    # Unrelated fields should still match
    assert scores["job_title_match"] == 1.0
