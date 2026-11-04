"""Streamlit front end for the Job Analyzer.

Run locally with:
    streamlit run app.py
"""

import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from job_analyzer import AnalysisResult, analyze, score_match
from job_analyzer.file_loader import extract_text
from job_analyzer.providers import PROVIDERS
from job_analyzer.schemas import CVMatch, JobPosting

load_dotenv()

EXAMPLES = Path(__file__).parent / "examples"
LANGUAGES = {"English": "English", "Ελληνικά": "Greek"}
STATUS_ICON = {"met": "✅ met", "partial": "🟡 partial", "missing": "❌ missing"}

st.set_page_config(page_title="Job Analyzer", page_icon="🧭", layout="wide")


def get_api_key(env_name: str) -> str | None:
    """Look for the key in Streamlit secrets (for deployment), then in the environment / .env file."""
    try:
        key = st.secrets.get(env_name)
    except Exception:  # noqa: BLE001 - no secrets file when running locally
        key = None
    return key or os.getenv(env_name)


def load_demo() -> tuple[AnalysisResult, str, str]:
    """Build an AnalysisResult from the precomputed example, without calling the API."""
    data = json.loads((EXAMPLES / "demo_result.json").read_text(encoding="utf-8"))
    posting_text = (EXAMPLES / "job_posting_ai_engineer.txt").read_text(encoding="utf-8")
    cv = (EXAMPLES / "cv_sample.txt").read_text(encoding="utf-8")
    posting, cv_match = JobPosting(**data["posting"]), CVMatch(**data["cv_match"])
    result = AnalysisResult(posting=posting, cv_match=cv_match, report=score_match(posting, cv_match, posting_text, cv))
    return result, posting_text, cv


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    provider_id = st.radio("Provider", list(PROVIDERS), horizontal=True,
                           format_func=lambda p: PROVIDERS[p][0].name)
    provider_cls, key_env = PROVIDERS[provider_id]
    api_key = get_api_key(key_env) or st.text_input(f"{provider_cls.name} API key", type="password",
                                                    help="Used only for this session, never stored.")
    demo_mode = st.toggle("Demo mode (no API calls)", value=not api_key,
                          help="Shows a precomputed analysis of the example posting and CV.")
    models = [provider_cls.default_model] + [m for m in provider_cls.models if m != provider_cls.default_model]
    model = st.selectbox("Model", models, disabled=demo_mode)
    language = LANGUAGES[st.radio("Output language", list(LANGUAGES), horizontal=True, disabled=demo_mode)]
    st.divider()
    st.caption(f"The job posting and CV are sent to the {provider_cls.name} API for analysis and are not stored by this app.")

# ---------------------------------------------------------------- inputs
st.title("🧭 Job Analyzer")
st.write("Paste a job posting to extract its requirements. Add your CV to see how well you match and what to improve.")

if "posting_text" not in st.session_state:
    st.session_state.posting_text = ""
if "cv_text" not in st.session_state:
    st.session_state.cv_text = ""

left, right = st.columns(2)
with left:
    st.subheader("1. Job posting")
    if st.button("Load example posting and CV"):
        st.session_state.posting_text = (EXAMPLES / "job_posting_ai_engineer.txt").read_text(encoding="utf-8")
        st.session_state.cv_text = (EXAMPLES / "cv_sample.txt").read_text(encoding="utf-8")
    posting_text = st.text_area("Job posting text", key="posting_text", height=320, label_visibility="collapsed",
                           placeholder="Paste the full job posting here (English or Greek)…")
with right:
    st.subheader("2. Your CV (optional)")
    upload = st.file_uploader("Upload CV", type=["pdf", "docx", "txt"], label_visibility="collapsed")
    if upload is not None and st.session_state.get("upload_id") != upload.file_id:
        st.session_state.upload_id = upload.file_id  # only re-read when a new file is uploaded
        try:
            st.session_state.cv_text = extract_text(upload.name, upload.getvalue())
        except Exception as exc:  # noqa: BLE001 - show any parsing problem (bad PDF, wrong type) to the user
            st.error(f"Could not read the file: {exc}")
    cv_text = st.text_area("CV text", key="cv_text", height=250, label_visibility="collapsed",
                           placeholder="…or paste your CV text here.")

run = st.button("Analyze", type="primary", width="stretch")

# ---------------------------------------------------------------- run
if run:
    if demo_mode:
        st.session_state.result, *_ = load_demo()
        st.info("Demo mode: showing the precomputed analysis of the example posting and CV.")
    elif not api_key:
        st.warning(f"Add a {provider_cls.name} API key in the sidebar, or turn on demo mode.")
    else:
        try:
            with st.spinner("Analyzing…"):
                st.session_state.result = analyze(provider_cls(api_key=api_key), posting_text, cv_text, model, language)
        except Exception as exc:  # noqa: BLE001 - API, network and validation errors are all shown to the user
            st.error(f"Analysis failed: {exc}")

result: AnalysisResult | None = st.session_state.get("result")
if result is None:
    st.stop()

# ---------------------------------------------------------------- results
p = result.posting
st.divider()
st.header(f"{p.job_title}" + (f" · {p.company}" if p.company else ""))
st.write(p.summary)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Seniority", p.seniority)
c2.metric("Experience", f"{p.min_years_experience}+ years" if p.min_years_experience is not None else "not stated")
c3.metric("Work mode", p.work_mode)
c4.metric("Location", p.location or "not stated")

report = result.report
if report:
    st.subheader("Match")
    m1, m2, m3 = st.columns(3)
    m1.metric("Match score", f"{report.score:.0f} / 100", help="Must-haves count double. Met = 1, partial = 0.5, missing = 0.")
    m2.metric("Must-haves met", f"{report.must_have_coverage:.0f}%")
    m3.metric("Missing must-haves", len(report.missing_must_haves))
    st.progress(report.score / 100)

    rows = [
        {
            "Requirement": r.skill,
            "Importance": "must have" if r.importance == "must_have" else "nice to have",
            "Status": STATUS_ICON[r.status],
            "Evidence in CV": r.cv_quote or "—",
            "Explanation": r.explanation,
        }
        for r in report.requirements
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=36 * (len(rows) + 1) + 3)

    downgraded = [r.skill for r in report.requirements if r.downgraded]
    if downgraded:
        st.warning("Grounding check: the model's evidence for "
                   + ", ".join(downgraded) + " was not found in the CV, so it was downgraded to 'partial'.")

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**Strengths**")
        for item in result.cv_match.strengths:
            st.markdown(f"- {item}")
    with s2:
        st.markdown("**Suggestions**")
        for item in result.cv_match.suggestions:
            st.markdown(f"- {item}")
else:
    st.subheader("Requirements")
    st.dataframe(pd.DataFrame([r.model_dump() for r in p.requirements]), hide_index=True, width="stretch")

with st.expander("Responsibilities and benefits"):
    st.markdown("**Responsibilities**\n" + "\n".join(f"- {x}" for x in p.responsibilities))
    st.markdown("**Benefits**\n" + "\n".join(f"- {x}" for x in p.benefits))

if result.stats:
    with st.expander("API usage"):
        st.dataframe(pd.DataFrame([asdict(s) for s in result.stats]), hide_index=True)

export = {"posting": p.model_dump()}
if report:
    export["cv_match"] = result.cv_match.model_dump()
    export["report"] = asdict(report)
st.download_button("Download report (JSON)", json.dumps(export, indent=2, ensure_ascii=False),
                   file_name="job_analysis.json", mime="application/json")
