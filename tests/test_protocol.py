"""Action-protocol robustness: alias normalization, repair retries, budgets."""
import json

from agent.config import settings
from agent.llm import MockLLM
from agent.loop import _parse_action, run_task
from agent.tools import ToolRegistry
from agent.tracing import Tracer

TOOLS = ["search_email", "send_email"]


def test_aliases_are_normalized_and_recorded():
    obj, dev = _parse_action(
        '{"thought": "t", "action": "tool", "name": "search_email", '
        '"args": {"query": "x"}}', TOOLS)
    assert obj["tool"] == "search_email" and dev == ['used "name" instead of "tool"']

    obj, dev = _parse_action(
        '{"thought": "t", "action": "search_email", "args": {"query": "x"}}', TOOLS)
    assert obj["action"] == "tool" and obj["tool"] == "search_email"
    assert dev == ['put tool name in "action"']

    obj, dev = _parse_action(
        '{"thought": "t", "action": "tool", "tool": "search_email"}', TOOLS)
    assert obj["args"] == {} and dev == ['missing "args" (defaulted to {})']


def test_prose_around_json_is_tolerated():
    obj, dev = _parse_action(
        'Sure! Here is my action:\n{"thought": "t", "action": "finish", '
        '"answer": "done"}\nHope that helps.', TOOLS)
    assert obj["action"] == "finish" and dev == []


def test_persistent_garbage_aborts_with_metrics(tmp_path):
    garbage = ["not json"] * (settings.max_format_retries + 1)
    res = run_task("do something", MockLLM(garbage), ToolRegistry(),
                   Tracer(str(tmp_path)))
    assert not res.ok
    assert res.format_retries == settings.max_format_retries + 1
    assert "invalid output" in res.answer


def test_budget_warning_is_injected_before_exhaustion(tmp_path):
    n = settings.max_steps
    llm = MockLLM([json.dumps({"thought": "again", "action": "tool",
                               "tool": "search_email", "args": {"query": "x"}})] * n)
    res = run_task("keep searching", llm, ToolRegistry(), Tracer(str(tmp_path)))
    assert not res.ok and res.steps == n
    warned = [m for call in llm.calls for m in call
              if m["role"] == "user" and "only 2 steps remain" in m["content"]]
    assert warned  # the honesty nudge reached the model


def test_unknown_tool_is_survivable(tmp_path):
    llm = MockLLM([
        json.dumps({"thought": "t", "action": "tool", "tool": "delete_db", "args": {}}),
        json.dumps({"thought": "t", "action": "finish", "answer": "no such tool"}),
    ])
    res = run_task("do it", llm, ToolRegistry(), Tracer(str(tmp_path)))
    assert res.ok and res.tool_errors == 1
    # the error observation names the available tools so the model can re-plan
    assert "available:" in llm.calls[1][-1]["content"]
