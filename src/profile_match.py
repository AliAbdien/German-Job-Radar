"""Score an extracted job posting against a candidate profile - turns "here's
a structured JSON extraction" into "here's whether this is worth your time".

Deliberately simple and transparent (a weighted sum of interpretable
sub-scores, not a black-box model) - the whole point of a personal job-search
tool is trusting *why* it ranked something, not just the number.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.schema import JobExtraction

PROFILE_PATH = Path(__file__).resolve().parent.parent / "data" / "ali_profile.json"

SENIORITY_ORDER = ["intern", "working_student", "junior", "mid", "senior", "lead"]

# How much worse a fit is for each rung of seniority mismatch (0 = exact match).
SENIORITY_MISMATCH_PENALTY = {0: 1.0, 1: 0.6, 2: 0.25}


def load_profile(path: Path = PROFILE_PATH) -> dict:
    with path.open() as f:
        return json.load(f)


def _skill_overlap_score(job: JobExtraction, profile: dict) -> float:
    profile_skills = {s.lower() for s in profile["skills"]}
    required = {s.lower() for s in job.required_skills}
    if not required:
        return 0.5  # no requirements stated - neutral, not a rejection
    matched = sum(1 for s in required if any(s in ps or ps in s for ps in profile_skills))
    return matched / len(required)


def _seniority_score(job: JobExtraction, profile: dict) -> float:
    if job.seniority == "unspecified":
        return 0.5
    if job.seniority not in SENIORITY_ORDER:
        return 0.5
    profile_idx = SENIORITY_ORDER.index(profile["seniority"])
    job_idx = SENIORITY_ORDER.index(job.seniority)
    distance = abs(profile_idx - job_idx)
    return SENIORITY_MISMATCH_PENALTY.get(distance, 0.0)


def _language_score(job: JobExtraction, profile: dict) -> float:
    if not job.required_languages:
        return 1.0
    ok = True
    for lang in job.required_languages:
        lang_lower = lang.lower()
        if "german" in lang_lower:
            # crude: reject only if a high bar (C1/C2/native/fluent) is stated
            # and profile's level is B2 - good enough for a first-pass filter.
            if any(tag in lang_lower for tag in ["c1", "c2", "native", "fluent", "verhandlungssicher"]):
                if profile["german_level"] not in ("C1", "C2", "native"):
                    ok = False
        if "english" in lang_lower and profile["english_level"] not in ("B2", "C1", "C2", "native"):
            ok = False
    return 1.0 if ok else 0.3


def _location_score(job: JobExtraction, profile: dict) -> float:
    if job.remote_policy == "remote":
        return 1.0
    if job.location is None:
        return 0.7  # unstated - don't penalize, but don't reward either
    profile_city = profile["location"].split(",")[0].lower()
    if job.location.lower() == profile_city:
        return 1.0
    if profile.get("willing_to_relocate_within_germany"):
        return 0.7 if job.remote_policy == "hybrid" else 0.5
    return 0.2


def score_posting(job: JobExtraction, profile: dict | None = None) -> dict:
    profile = profile or load_profile()
    sub_scores = {
        "skill_overlap": round(_skill_overlap_score(job, profile), 3),
        "seniority_fit": round(_seniority_score(job, profile), 3),
        "language_fit": round(_language_score(job, profile), 3),
        "location_fit": round(_location_score(job, profile), 3),
    }
    weights = {"skill_overlap": 0.4, "seniority_fit": 0.25, "language_fit": 0.2, "location_fit": 0.15}
    overall = sum(sub_scores[k] * weights[k] for k in weights)
    return {
        "job_title": job.job_title,
        "overall_score": round(overall, 3),
        "sub_scores": sub_scores,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--extractions", required=True, help="JSONL file of JobExtraction records to score.")
    args = parser.parse_args()

    profile = load_profile()
    scored = []
    with open(args.extractions) as f:
        for line in f:
            if not line.strip():
                continue
            job = JobExtraction.model_validate_json(line)
            scored.append(score_posting(job, profile))

    scored.sort(key=lambda s: s["overall_score"], reverse=True)
    for s in scored:
        print(f"{s['overall_score']:.2f}  {s['job_title']}  {s['sub_scores']}")
