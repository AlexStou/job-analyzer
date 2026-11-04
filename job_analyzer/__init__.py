"""Job Analyzer: extract requirements from job postings and compare them with a CV."""

from .analyzer import AnalysisError, AnalysisResult, analyze, extract_job_posting, match_cv
from .schemas import CVMatch, JobPosting
from .scoring import MatchReport, score_match

__all__ = [
    "AnalysisError",
    "AnalysisResult",
    "CVMatch",
    "JobPosting",
    "MatchReport",
    "analyze",
    "extract_job_posting",
    "match_cv",
    "score_match",
]
