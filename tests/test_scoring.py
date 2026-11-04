from job_analyzer.schemas import CVMatch, JobPosting, Requirement, RequirementMatch
from job_analyzer.scoring import normalize, quote_in_text, score_match

POSTING = "Strong programming skills in Python. Experience with Docker is a plus."
CV = "Skills: Python, MATLAB. Thesis in time series analysis."


def make_posting() -> JobPosting:
    return JobPosting(
        job_title="Junior Developer",
        company=None,
        seniority="junior",
        min_years_experience=None,
        location=None,
        work_mode="unspecified",
        requirements=[
            Requirement(skill="Python", category="programming_language", importance="must_have",
                        posting_quote="Strong programming skills in Python"),
            Requirement(skill="Docker", category="cloud_devops", importance="nice_to_have",
                        posting_quote="Experience with Docker"),
        ],
        responsibilities=[],
        benefits=[],
        summary="",
    )


def test_normalize_ignores_case_accents_and_punctuation():
    assert normalize("Ανάλυση  Χρονοσειρών!") == normalize("αναλυση χρονοσειρων")


def test_quote_in_text():
    assert quote_in_text("skills: python", CV)
    assert not quote_in_text("Kubernetes", CV)
    assert not quote_in_text(None, CV)
    assert not quote_in_text("   ", CV)


def test_all_met_gives_100():
    match = CVMatch(
        matches=[
            RequirementMatch(skill="Python", status="met", cv_quote="Python, MATLAB", explanation=""),
            RequirementMatch(skill="Docker", status="met", cv_quote="time series", explanation=""),
        ],
        strengths=[],
        suggestions=[],
    )
    report = score_match(make_posting(), match, POSTING, CV)
    assert report.score == 100.0
    assert report.must_have_coverage == 100.0


def test_weighting_must_have_counts_double():
    # Python (weight 2) met, Docker (weight 1) missing -> 2 / 3
    match = CVMatch(
        matches=[
            RequirementMatch(skill="Python", status="met", cv_quote="Python", explanation=""),
            RequirementMatch(skill="Docker", status="missing", cv_quote=None, explanation=""),
        ],
        strengths=[],
        suggestions=[],
    )
    report = score_match(make_posting(), match, POSTING, CV)
    assert report.score == round(100 * 2 / 3, 1)
    assert report.missing_must_haves == []


def test_unverified_quote_is_downgraded():
    match = CVMatch(
        matches=[
            RequirementMatch(skill="Python", status="met", cv_quote="5 years of Python at Google", explanation=""),
            RequirementMatch(skill="Docker", status="missing", cv_quote=None, explanation=""),
        ],
        strengths=[],
        suggestions=[],
    )
    report = score_match(make_posting(), match, POSTING, CV)
    python_row = report.requirements[0]
    assert python_row.status == "partial"
    assert python_row.downgraded
    assert report.must_have_coverage == 0.0


def test_requirement_skipped_by_model_counts_as_missing():
    match = CVMatch(
        matches=[RequirementMatch(skill="python", status="met", cv_quote="Python", explanation="")],
        strengths=[],
        suggestions=[],
    )
    report = score_match(make_posting(), match, POSTING, CV)
    docker_row = report.requirements[1]
    assert docker_row.status == "missing"
    assert report.requirements[0].status == "met"  # skill names matched case-insensitively


def test_posting_quotes_are_verified():
    match = CVMatch(matches=[], strengths=[], suggestions=[])
    report = score_match(make_posting(), match, POSTING, CV)
    assert all(r.posting_quote_verified for r in report.requirements)
