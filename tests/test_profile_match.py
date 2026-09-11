from src.profile_match import load_profile, score_posting
from src.schema import JobExtraction

PROFILE = load_profile()


def _job(**overrides) -> JobExtraction:
    base = dict(
        job_title="Junior AI Engineer",
        seniority="junior",
        required_skills=["Python", "Machine Learning"],
        salary={"stated": False},
        remote_policy="unspecified",
        required_languages=[],
    )
    base.update(overrides)
    return JobExtraction.model_validate(base)


def test_strong_skill_and_seniority_match_scores_high():
    job = _job()
    result = score_posting(job, PROFILE)
    assert result["overall_score"] > 0.7


def test_unrelated_role_scores_lower_than_a_strong_match():
    strong = score_posting(_job(), PROFILE)
    weak = score_posting(
        _job(
            job_title="Vertriebsmitarbeiter Außendienst",
            seniority="unspecified",
            required_skills=["Sales Experience", "Driver's License Class B"],
            remote_policy="onsite",
            location="Bayern",
        ),
        PROFILE,
    )
    # The whole point of the ranking is separating a strong match from an
    # unrelated role by a wide margin, not by a hair.
    assert weak["overall_score"] < strong["overall_score"] - 0.3


def test_remote_role_gets_full_location_score():
    job = _job(remote_policy="remote")
    result = score_posting(job, PROFILE)
    assert result["sub_scores"]["location_fit"] == 1.0


def test_high_german_bar_penalizes_b2_profile():
    job = _job(required_languages=["German (C1)"])
    result = score_posting(job, PROFILE)
    assert result["sub_scores"]["language_fit"] < 1.0


def test_senior_role_penalizes_junior_profile():
    job = _job(seniority="lead")
    result = score_posting(job, PROFILE)
    assert result["sub_scores"]["seniority_fit"] < 0.5
