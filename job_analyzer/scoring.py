"""Deterministic scoring and grounding checks.

The LLM decides *whether* each requirement is met. Python, not the LLM,
turns those decisions into a score, so the same inputs always give the
same number and the formula is easy to explain.

We also check that every quote the model returns really appears in the
source text. If the model says a skill is "met" but the quote it gives is
not in the CV, we downgrade it to "partial" and say so. This is a simple
guard against hallucinated evidence.
"""

import re
import unicodedata
from dataclasses import dataclass, field

from .schemas import CVMatch, JobPosting, RequirementMatch

IMPORTANCE_WEIGHT = {"must_have": 2.0, "nice_to_have": 1.0}
STATUS_CREDIT = {"met": 1.0, "partial": 0.5, "missing": 0.0}


def normalize(text: str) -> str:
    """Lower-case, strip accents and collapse whitespace/punctuation so quotes can be compared."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^\w]+", " ", text)
    return text.strip()


def quote_in_text(quote: str | None, text: str) -> bool:
    """True if `quote` appears in `text` (ignoring case, accents, spacing and punctuation)."""
    if not quote:
        return False
    q = normalize(quote)
    return bool(q) and q in normalize(text)


@dataclass
class ScoredRequirement:
    skill: str
    category: str
    importance: str
    status: str
    cv_quote: str | None
    explanation: str
    posting_quote: str
    posting_quote_verified: bool
    cv_quote_verified: bool
    downgraded: bool = False


@dataclass
class MatchReport:
    score: float  # 0-100
    must_have_coverage: float  # 0-100, share of must-haves that are met
    requirements: list[ScoredRequirement] = field(default_factory=list)
    missing_must_haves: list[str] = field(default_factory=list)


def _verify(match: RequirementMatch, cv_text: str) -> tuple[str, bool, bool, str]:
    """Return (status, cv_quote_verified, downgraded, explanation) after the grounding check."""
    verified = quote_in_text(match.cv_quote, cv_text)
    status, explanation, downgraded = match.status, match.explanation, False
    if status == "met" and not verified:
        status, downgraded = "partial", True
        explanation += " [Downgraded: the supporting quote was not found in the CV.]"
    return status, verified, downgraded, explanation


def score_match(posting: JobPosting, cv_match: CVMatch, posting_text: str, cv_text: str) -> MatchReport:
    """Combine the extracted requirements with the CV match into a scored report."""
    by_skill = {normalize(m.skill): m for m in cv_match.matches}

    rows: list[ScoredRequirement] = []
    earned = total = 0.0
    must_total = must_met = 0

    for req in posting.requirements:
        match = by_skill.get(normalize(req.skill))
        if match is None:
            # The model skipped this requirement: treat it as missing rather than ignore it.
            status, cv_quote, verified, downgraded = "missing", None, False, False
            explanation = "Not assessed by the model; counted as missing."
        else:
            cv_quote = match.cv_quote
            status, verified, downgraded, explanation = _verify(match, cv_text)

        weight = IMPORTANCE_WEIGHT[req.importance]
        total += weight
        earned += weight * STATUS_CREDIT[status]
        if req.importance == "must_have":
            must_total += 1
            must_met += status == "met"

        rows.append(
            ScoredRequirement(
                skill=req.skill,
                category=req.category,
                importance=req.importance,
                status=status,
                cv_quote=cv_quote,
                explanation=explanation,
                posting_quote=req.posting_quote,
                posting_quote_verified=quote_in_text(req.posting_quote, posting_text),
                cv_quote_verified=verified,
                downgraded=downgraded,
            )
        )

    return MatchReport(
        score=round(100 * earned / total, 1) if total else 0.0,
        must_have_coverage=round(100 * must_met / must_total, 1) if must_total else 100.0,
        requirements=rows,
        missing_must_haves=[r.skill for r in rows if r.importance == "must_have" and r.status == "missing"],
    )
