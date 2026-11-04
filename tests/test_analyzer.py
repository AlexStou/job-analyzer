"""Tests for the pipeline using a fake OpenAI client, so no API key or network is needed."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from job_analyzer import AnalysisError, analyze
from job_analyzer.providers import AnthropicProvider, OpenAIProvider
from job_analyzer.schemas import CVMatch, JobPosting

EXAMPLES = Path(__file__).parent.parent / "examples"
DEMO = json.loads((EXAMPLES / "demo_result.json").read_text(encoding="utf-8"))
POSTING = (EXAMPLES / "job_posting_ai_engineer.txt").read_text(encoding="utf-8")
CV = (EXAMPLES / "cv_sample.txt").read_text(encoding="utf-8")


class FakeResponses:
    def __init__(self):
        self.calls = []

    def parse(self, *, model, instructions, input, text_format):
        self.calls.append({"model": model, "instructions": instructions, "input": input, "schema": text_format})
        data = DEMO["posting"] if text_format is JobPosting else DEMO["cv_match"]
        return SimpleNamespace(
            output_parsed=text_format(**data),
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )


class FakeClient:
    """Mimics openai.OpenAI: only client.responses.parse is used."""

    def __init__(self):
        self.responses = FakeResponses()


class FakeMessages:
    def __init__(self):
        self.calls = []

    def parse(self, *, model, max_tokens, system, messages, output_format):
        self.calls.append({"model": model, "system": system, "messages": messages, "schema": output_format})
        data = DEMO["posting"] if output_format is JobPosting else DEMO["cv_match"]
        return SimpleNamespace(
            parsed_output=output_format(**data),
            usage=SimpleNamespace(input_tokens=80, output_tokens=40),
        )


class FakeAnthropicClient:
    """Mimics anthropic.Anthropic: only client.messages.parse is used."""

    def __init__(self):
        self.messages = FakeMessages()


def openai_provider():
    client = FakeClient()
    return OpenAIProvider(client=client), client


def test_full_pipeline_with_cv():
    provider, client = openai_provider()
    result = analyze(provider, POSTING, CV, model="test-model", language="Greek")

    assert [c["schema"] for c in client.responses.calls] == [JobPosting, CVMatch]
    assert "Greek" in client.responses.calls[0]["instructions"]
    assert "<job_posting>" in client.responses.calls[0]["input"]
    assert "<cv>" in client.responses.calls[1]["input"]
    assert result.report is not None
    assert 0 <= result.report.score <= 100
    assert len(result.stats) == 2


def test_pipeline_without_cv_skips_matching():
    provider, client = openai_provider()
    result = analyze(provider, POSTING, "")
    assert len(client.responses.calls) == 1
    assert result.report is None


def test_demo_example_downgrades_unsupported_sql_claim():
    result = analyze(openai_provider()[0], POSTING, CV)
    sql = next(r for r in result.report.requirements if r.skill == "SQL")
    assert sql.status == "partial" and sql.downgraded


def test_empty_ad_raises():
    with pytest.raises(AnalysisError):
        analyze(openai_provider()[0], "   ")


def test_missing_parsed_output_raises():
    provider, client = openai_provider()
    client.responses.parse = lambda **kwargs: SimpleNamespace(output_parsed=None, usage=None)
    with pytest.raises(AnalysisError):
        analyze(provider, POSTING)


def test_anthropic_provider_runs_same_pipeline():
    client = FakeAnthropicClient()
    result = analyze(AnthropicProvider(client=client), POSTING, CV, language="Greek")
    calls = client.messages.calls
    assert [c["schema"] for c in calls] == [JobPosting, CVMatch]
    assert calls[0]["model"] == AnthropicProvider.default_model
    assert "Greek" in calls[0]["system"]
    assert calls[1]["messages"][0]["content"].count("<cv>") == 1
    assert result.report is not None and len(result.stats) == 2
