"""The structured extraction target: what we want the model to pull out of a
raw German job posting. Kept deliberately close to what a real applicant
tracking workflow needs, not an academic NER label set.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SalaryRange(BaseModel):
    min_eur: Optional[int] = Field(None, description="Lower bound, annual gross EUR, if stated.")
    max_eur: Optional[int] = Field(None, description="Upper bound, annual gross EUR, if stated.")
    stated: bool = Field(..., description="Whether a salary was mentioned at all - most German postings omit it.")


class JobExtraction(BaseModel):
    job_title: str = Field(..., description="Normalized job title, e.g. 'Machine Learning Engineer'.")
    company: Optional[str] = Field(None, description="Company name if present in the posting text.")
    seniority: Literal["intern", "working_student", "junior", "mid", "senior", "lead", "unspecified"] = Field(
        ..., description="Inferred from title/requirements even when not stated explicitly."
    )
    required_skills: list[str] = Field(
        default_factory=list, description="Concrete required skills/technologies, e.g. ['Python', 'PyTorch', 'SQL']."
    )
    nice_to_have_skills: list[str] = Field(default_factory=list)
    salary: SalaryRange
    location: Optional[str] = Field(None, description="City or region, as stated.")
    remote_policy: Literal["remote", "hybrid", "onsite", "unspecified"] = "unspecified"
    required_languages: list[str] = Field(
        default_factory=list, description="Languages explicitly required, e.g. ['German (C1)', 'English (B2)']."
    )
    visa_sponsorship_mentioned: bool = Field(
        False, description="Whether the posting explicitly mentions visa/relocation support."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_title": "Junior Data Scientist",
                    "company": "Beispiel GmbH",
                    "seniority": "junior",
                    "required_skills": ["Python", "SQL", "Machine Learning"],
                    "nice_to_have_skills": ["PyTorch", "Docker"],
                    "salary": {"min_eur": 48000, "max_eur": 58000, "stated": True},
                    "location": "München",
                    "remote_policy": "hybrid",
                    "required_languages": ["German (C1)", "English (B2)"],
                    "visa_sponsorship_mentioned": False,
                }
            ]
        }
    }


EXTRACTION_JSON_SCHEMA = JobExtraction.model_json_schema()
