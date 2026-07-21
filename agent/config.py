"""Provider-agnostic configuration.

Any OpenAI-compatible endpoint works — local ones cost $0:

  LM Studio          AGENT_BASE_URL=http://localhost:1234/v1
  Ollama             AGENT_BASE_URL=http://localhost:11434/v1
  llama.cpp server   AGENT_BASE_URL=http://localhost:8080/v1
  (any cloud OpenAI-compatible endpoint works the same way)

Local servers ignore the API key, but the client requires one — "local" is fine.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    base_url: str = os.getenv("AGENT_BASE_URL", "http://localhost:1234/v1")
    api_key: str = os.getenv("AGENT_API_KEY", "local")
    model: str = os.getenv("AGENT_MODEL", "qwen3-4b-instruct")
    temperature: float = float(os.getenv("AGENT_TEMPERATURE", "0.1"))
    # hard cap on completion length: one JSON action never needs more, and
    # unbounded generation lets a rambling model stall the whole eval
    max_tokens: int = int(os.getenv("AGENT_MAX_TOKENS", "1000"))
    max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "12"))
    # retries for malformed model output (schema-invalid JSON)
    max_format_retries: int = int(os.getenv("AGENT_FORMAT_RETRIES", "2"))
    trace_dir: str = os.getenv("AGENT_TRACE_DIR", "traces")
    # --- injection defenses (see agent/guard.py); on by default, set "0" to
    # reproduce the undefended baseline the v0.3 campaign measured ----------
    egress_guard: bool = os.getenv("AGENT_EGRESS_GUARD", "1") != "0"
    provenance_tags: bool = os.getenv("AGENT_PROVENANCE_TAGS", "1") != "0"
    # optional separate grader for `judge` checks — defaults to the model
    # under test (a stated limitation) unless overridden
    judge_model: str = os.getenv("AGENT_JUDGE_MODEL", "")
    judge_base_url: str = os.getenv("AGENT_JUDGE_BASE_URL", "")


settings = Settings()
