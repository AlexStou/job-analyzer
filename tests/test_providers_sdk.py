"""Check that the real SDKs accept our schemas and build valid requests.

The HTTP layer is replaced with a mock, so no API key or network is needed.
"""

import importlib
import json
from pathlib import Path

import httpx
from anthropic import Anthropic
from openai import OpenAI

from job_analyzer.providers import AnthropicProvider, OpenAIProvider
from job_analyzer.schemas import JobPosting

POSTING = json.loads((Path(__file__).parent.parent / "examples" / "demo_result.json").read_text(encoding="utf-8"))["posting"]


def _anthropic_http():
    """Recent anthropic SDKs use the `httpx2` package; older ones use `httpx`."""
    try:
        return importlib.import_module("httpx2")
    except ImportError:
        return httpx


def test_anthropic_sdk_request_and_parsing():
    http = _anthropic_http()
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return http.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
            "content": [{"type": "text", "text": json.dumps(POSTING)}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 20},
        })

    client = Anthropic(api_key="test", http_client=http.Client(transport=http.MockTransport(handler)))
    result = AnthropicProvider(client=client).parse("claude-haiku-5-5", "instructions", "<job_posting>x</job_posting>", JobPosting)

    assert isinstance(result.parsed, JobPosting)
    assert result.parsed.job_title == POSTING["job_title"]
    assert result.input_tokens == 10
    assert seen["body"]["system"] == "instructions"
    assert "output_config" in seen["body"] or "output_format" in seen["body"]  # schema was sent


def test_openai_sdk_request_and_parsing():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "resp_1", "object": "response", "created_at": 0, "model": "gpt-5.4-mini", "status": "completed",
            "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": json.dumps(POSTING), "annotations": []}]}],
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "usage": {"input_tokens": 12, "output_tokens": 30, "total_tokens": 42,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
        })

    client = OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    result = OpenAIProvider(client=client).parse("gpt-5.4-mini", "instructions", "<job_posting>x</job_posting>", JobPosting)

    assert isinstance(result.parsed, JobPosting)
    assert result.output_tokens == 30
    assert seen["body"]["text"]["format"]["type"] == "json_schema"
