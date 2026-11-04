"""LLM calls and the end-to-end analysis pipeline.

Pipeline:
    job posting text --(LLM call 1)--> JobPosting
    JobPosting + CV text --(LLM call 2)--> CVMatch
    JobPosting + CVMatch --(Python)--> MatchReport (score + grounding checks)

Every function takes a `provider` (see providers.py), so the same pipeline
runs on OpenAI or Anthropic Claude.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional, TypeVar

from pydantic import BaseModel

from .prompts import EXTRACT_JOB_INSTRUCTIONS, MATCH_CV_INSTRUCTIONS
from .schemas import CVMatch, JobPosting
from .providers import Provider
from .scoring import MatchReport, score_match

MAX_POSTING_CHARS = 15_000
MAX_CV_CHARS = 20_000

T = TypeVar("T", bound=BaseModel)


class AnalysisError(RuntimeError):
    pass


@dataclass
class CallStats:
    """Token usage and latency of one API call (useful for monitoring cost)."""

    step: str
    model: str
    input_tokens: int
    output_tokens: int
    seconds: float


@dataclass
class AnalysisResult:
    posting: JobPosting
    cv_match: Optional[CVMatch] = None
    report: Optional[MatchReport] = None
    stats: list[CallStats] = field(default_factory=list)


def _parse(provider: Provider, model: str, instructions: str, user_input: str, schema: type[T], step: str) -> tuple[T, CallStats]:
    """Send one request and return the parsed object plus usage stats."""
    start = time.perf_counter()
    result = provider.parse(model, instructions, user_input, schema)
    seconds = time.perf_counter() - start

    if result.parsed is None:  # e.g. the model refused or the output was cut off
        raise AnalysisError(f"The model returned no structured output during '{step}'.")

    stats = CallStats(
        step=step,
        model=model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        seconds=round(seconds, 2),
    )
    return result.parsed, stats


def extract_job_posting(provider: Provider, posting_text: str, model: str | None = None, language: str = "English") -> tuple[JobPosting, CallStats]:
    if not posting_text.strip():
        raise AnalysisError("The job posting is empty.")
    if len(posting_text) > MAX_POSTING_CHARS:
        raise AnalysisError(f"The job posting is too long ({len(posting_text)} characters, max {MAX_POSTING_CHARS}).")
    return _parse(
        provider,
        model or provider.default_model,
        EXTRACT_JOB_INSTRUCTIONS.format(language=language),
        f"<job_posting>\n{posting_text}\n</job_posting>",
        JobPosting,
        step="extract_job",
    )


def match_cv(provider: Provider, posting: JobPosting, cv_text: str, model: str | None = None, language: str = "English") -> tuple[CVMatch, CallStats]:
    if len(cv_text) > MAX_CV_CHARS:
        raise AnalysisError(f"The CV is too long ({len(cv_text)} characters, max {MAX_CV_CHARS}).")
    requirements = json.dumps(
        [{"skill": r.skill, "importance": r.importance, "category": r.category} for r in posting.requirements],
        ensure_ascii=False,
        indent=2,
    )
    user_input = f"<requirements>\n{requirements}\n</requirements>\n\n<cv>\n{cv_text}\n</cv>"
    return _parse(
        provider,
        model or provider.default_model,
        MATCH_CV_INSTRUCTIONS.format(language=language),
        user_input,
        CVMatch,
        step="match_cv",
    )


def analyze(provider: Provider, posting_text: str, cv_text: str = "", model: str | None = None, language: str = "English") -> AnalysisResult:
    """Run the full pipeline. The CV step is skipped when no CV text is given."""
    posting, stats1 = extract_job_posting(provider, posting_text, model, language)
    result = AnalysisResult(posting=posting, stats=[stats1])

    if cv_text.strip():
        cv_match, stats2 = match_cv(provider, posting, cv_text, model, language)
        result.cv_match = cv_match
        result.report = score_match(posting, cv_match, posting_text, cv_text)
        result.stats.append(stats2)

    return result
