import pytest
from pydantic import ValidationError

from src.schema import JobExtraction


def test_minimal_valid_example():
    ex = JobExtraction.model_validate(
        {
            "job_title": "Data Scientist",
            "seniority": "mid",
            "salary": {"stated": False},
        }
    )
    assert ex.required_skills == []
    assert ex.remote_policy == "unspecified"


def test_invalid_seniority_rejected():
    with pytest.raises(ValidationError):
        JobExtraction.model_validate(
            {"job_title": "X", "seniority": "not-a-real-level", "salary": {"stated": False}}
        )


def test_missing_required_field_rejected():
    with pytest.raises(ValidationError):
        JobExtraction.model_validate({"seniority": "mid", "salary": {"stated": False}})
