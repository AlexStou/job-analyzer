"""LLM providers: one small class per API, all with the same `parse` method.

The rest of the code only calls `provider.parse(...)`, so adding a provider
means adding one class here. Both providers return a validated Pydantic
object, so the pipeline does not care which API produced it.
"""

import os
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class ParseResult:
    parsed: BaseModel | None
    input_tokens: int
    output_tokens: int


class Provider(Protocol):
    name: str
    default_model: str
    models: list[str]

    def parse(self, model: str, instructions: str, user_input: str, schema: type[T]) -> ParseResult: ...


class OpenAIProvider:
    """OpenAI Responses API with Structured Outputs (`responses.parse`)."""

    name = "OpenAI"
    default_model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    models = ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-4.1-mini"]

    def __init__(self, api_key: str | None = None, client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self.client = client

    def parse(self, model, instructions, user_input, schema):
        response = self.client.responses.parse(
            model=model,
            instructions=instructions,
            input=user_input,
            text_format=schema,
        )
        usage = getattr(response, "usage", None)
        return ParseResult(
            parsed=response.output_parsed,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )


class AnthropicProvider:
    """Anthropic Claude Messages API with structured outputs (`messages.parse`)."""

    name = "Anthropic (Claude)"
    default_model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
    models = ["claude-haiku-5-5", "claude-sonnet-5-5"]

    def __init__(self, api_key: str | None = None, client=None):
        if client is None:
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key)
        self.client = client

    def parse(self, model, instructions, user_input, schema):
        response = self.client.messages.parse(
            model=model,
            max_tokens=8000,
            system=instructions,
            messages=[{"role": "user", "content": user_input}],
            output_format=schema,
        )
        usage = getattr(response, "usage", None)
        return ParseResult(
            parsed=response.parsed_output,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )


PROVIDERS = {
    "openai": (OpenAIProvider, "OPENAI_API_KEY"),
    "anthropic": (AnthropicProvider, "ANTHROPIC_API_KEY"),
}
