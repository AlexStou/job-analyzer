"""Data models for the job analyzer.

These Pydantic models do two jobs:
1. They are sent to the OpenAI API as the *structured output* schema, so the
   model is forced to answer with JSON that has exactly these fields.
2. They give us typed Python objects to work with after the call.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

Category = Literal[
    "programming_language",
    "framework_library",
    "ai_ml",
    "data",
    "cloud_devops",
    "tool",
    "soft_skill",
    "domain_knowledge",
    "spoken_language",
    "education",
    "other",
]

Importance = Literal["must_have", "nice_to_have"]
MatchStatus = Literal["met", "partial", "missing"]


# --------------------------------------------------------------------------
# Step 1: what the job posting asks for
# --------------------------------------------------------------------------
class Requirement(BaseModel):
    skill: str = Field(description="Short name of the skill or requirement, e.g. 'Python' or 'Docker'.")
    category: Category
    importance: Importance = Field(
        description="must_have if the posting says required/must/essential, nice_to_have if it says plus/bonus/preferred."
    )
    posting_quote: str = Field(description="The exact phrase from the job posting that states this requirement.")


class JobPosting(BaseModel):
    job_title: str
    company: Optional[str] = Field(description="Company name, or null if the posting does not say.")
    seniority: Literal["intern", "junior", "mid", "senior", "lead", "unspecified"]
    min_years_experience: Optional[int] = Field(description="Minimum years of experience asked for, or null.")
    location: Optional[str]
    work_mode: Literal["onsite", "hybrid", "remote", "unspecified"]
    requirements: list[Requirement]
    responsibilities: list[str]
    benefits: list[str]
    summary: str = Field(description="Two-sentence summary of the role.")


# --------------------------------------------------------------------------
# Step 2: how a CV compares with those requirements
# --------------------------------------------------------------------------
class RequirementMatch(BaseModel):
    skill: str = Field(description="Must repeat the skill name exactly as given in the requirements list.")
    status: MatchStatus
    cv_quote: Optional[str] = Field(
        description="Exact text copied from the CV that supports the status, or null if there is none."
    )
    explanation: str


class CVMatch(BaseModel):
    matches: list[RequirementMatch]
    strengths: list[str] = Field(description="Things in the CV that make the candidate stand out for this role.")
    suggestions: list[str] = Field(
        description="Concrete, honest actions to close the gaps (projects, courses, CV wording). Never invent experience."
    )
