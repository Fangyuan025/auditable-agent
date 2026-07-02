"""The agent loop: plan -> act -> observe, with validation and recovery.

Design choices (they ARE the point of this project):

* Tool calls are *prompted JSON*, not native function-calling — so any model
  works, including small on-device ones (Qwen3 4B via LM Studio/Ollama).
* Every model output is schema-validated; malformed output is retried with
  the parse error fed back (bounded by max_format_retries).
* Tool failures are not fatal: the error is returned to the model as an
  observation so it can re-plan (bounded by max_steps).
* Every step is traced (agent/tracing.py) for audit and replay.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from agent.config import settings
from agent.llm import LLM
from agent.tools import ToolError, ToolRegistry
from agent.tracing import Tracer

SYSTEM_PROMPT = """\
You are a careful enterprise assistant. You complete the user's task by
calling tools, one at a time, and then finish with a short answer.

Available tools:
{tool_spec}

Protocol — reply with EXACTLY one JSON object and nothing else:
  To call a tool:  {{"thought": "<why>", "action": "tool", "tool": "<name>", "args": {{...}}}}
  To finish:       {{"thought": "<why>", "action": "finish", "answer": "<final answer>"}}

Rules:
- One tool call per reply. Use only listed tools and exact arg names.
- If a tool errors, adapt: fix the args or choose another tool.
- If the task is impossible or unsafe, finish and say so plainly."""


@dataclass
class RunResult:
    ok: bool
    answer: str
    steps: int
    tool_calls: int
    tool_errors: int
    format_retries: int
    trace_file: str


def _parse_action(text: str) -> dict:
    """Extract the single JSON object from the model reply (tolerates fences)."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object found in reply")
    obj = json.loads(m.group(0))
    if obj.get("action") == "tool":
        if not isinstance(obj.get("tool"), str) or not isinstance(obj.get("args"), dict):
            raise ValueError('tool action needs "tool": str and "args": object')
    elif obj.get("action") == "finish":
        if not isinstance(obj.get("answer"), str):
            raise ValueError('finish action needs "answer": str')
    else:
        raise ValueError('"action" must be "tool" or "finish"')
    return obj


def run_task(task: str, llm: LLM, registry: ToolRegistry,
             tracer: Tracer | None = None) -> RunResult:
    tracer = tracer or Tracer(settings.trace_dir)
    messages = [
        {"role": "system",
         "content": SYSTEM_PROMPT.format(tool_spec=registry.spec_for_prompt())},
        {"role": "user", "content": task},
    ]
    tracer.log("task", task=task, model=getattr(llm, "model", "mock"))
    tool_calls = tool_errors = format_retries = 0

    for step in range(settings.max_steps):
        # --- get a valid action, retrying malformed output --------------------
        action = None
        for attempt in range(settings.max_format_retries + 1):
            text, latency = llm.complete(messages)
            tracer.log("llm", latency_s=round(latency, 3), raw=text[:2000])
            try:
                action = _parse_action(text)
                break
            except (ValueError, json.JSONDecodeError) as e:
                format_retries += 1
                tracer.log("format_error", error=str(e), attempt=attempt)
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user",
                                 "content": f"Invalid reply ({e}). Reply with one valid JSON object only."})
        if action is None:
            tracer.log("abort", reason="persistent malformed output")
            return RunResult(False, "aborted: model kept producing invalid output",
                             step + 1, tool_calls, tool_errors, format_retries,
                             str(tracer.flush()))

        messages.append({"role": "assistant", "content": json.dumps(action)})

        # --- finish -----------------------------------------------------------
        if action["action"] == "finish":
            tracer.log("finish", answer=action["answer"])
            return RunResult(True, action["answer"], step + 1, tool_calls,
                             tool_errors, format_retries, str(tracer.flush()))

        # --- tool call --------------------------------------------------------
        tool_calls += 1
        try:
            result = registry.execute(action["tool"], action["args"])
            obs = json.dumps(result, ensure_ascii=False, default=str)[:1500]
            tracer.log("tool", tool=action["tool"], args=action["args"], ok=True,
                       observation=obs)
        except (ToolError, ValueError) as e:
            tool_errors += 1
            obs = f"ERROR: {e}"
            tracer.log("tool", tool=action["tool"], args=action["args"], ok=False,
                       observation=obs)
        messages.append({"role": "user", "content": f"Observation: {obs}"})

    tracer.log("abort", reason="max_steps reached")
    return RunResult(False, "aborted: step budget exhausted", settings.max_steps,
                     tool_calls, tool_errors, format_retries, str(tracer.flush()))
