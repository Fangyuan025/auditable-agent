"""LLM clients.

`OpenAICompatibleLLM` talks to any OpenAI-compatible /chat/completions server
(LM Studio, Ollama, llama.cpp server, vLLM, or a cloud endpoint).

`MockLLM` replays a scripted list of responses so the agent loop, tracing,
validation, and recovery logic are all unit-testable with zero model calls.
"""
from __future__ import annotations

import time
from typing import Protocol

from openai import OpenAI

from agent.config import settings


class LLM(Protocol):
    def complete(self, messages: list[dict]) -> tuple[str, float]:
        """Return (assistant_text, latency_seconds)."""
        ...


class OpenAICompatibleLLM:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, temperature: float | None = None):
        self.model = model or settings.model
        self.temperature = settings.temperature if temperature is None else temperature
        self._client = OpenAI(base_url=base_url or settings.base_url,
                              api_key=api_key or settings.api_key)

    def complete(self, messages: list[dict]) -> tuple[str, float]:
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=settings.max_tokens,
        )
        latency = time.perf_counter() - t0
        return resp.choices[0].message.content or "", latency


class MockLLM:
    """Deterministic scripted LLM for tests and CI (no server, no cost)."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict]) -> tuple[str, float]:
        self.calls.append(messages)
        if not self._responses:
            raise RuntimeError("MockLLM ran out of scripted responses")
        return self._responses.pop(0), 0.0
