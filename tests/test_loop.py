"""Agent-loop tests — fully deterministic via MockLLM, no model required."""
import json

from agent.llm import MockLLM
from agent.loop import run_task
from agent.tools import ToolRegistry
from agent.tracing import Tracer


def _tool(thought, tool, **args):
    return json.dumps({"thought": thought, "action": "tool", "tool": tool, "args": args})


def _finish(answer):
    return json.dumps({"thought": "done", "action": "finish", "answer": answer})


def test_happy_path_tool_then_finish(tmp_path):
    llm = MockLLM([
        _tool("find the webhook complaint", "search_email", query="webhook"),
        _finish("Found the failing-webhooks email from dev@acme.com."),
    ])
    reg = ToolRegistry()
    res = run_task("Find the email about webhooks", llm, reg, Tracer(str(tmp_path)))
    assert res.ok and res.tool_calls == 1 and res.tool_errors == 0
    assert "dev@acme.com" in res.answer


def test_side_effects_are_observable(tmp_path):
    llm = MockLLM([
        _tool("reply to claire", "send_email", to="claire@acme.com",
              subject="Re: Renewal", body="We'll clarify pricing before Friday."),
        _finish("Replied to Claire."),
    ])
    reg = ToolRegistry()
    run_task("Reply to Claire about renewal pricing", llm, reg, Tracer(str(tmp_path)))
    assert reg.workplace.sent and reg.workplace.sent[0]["to"] == "claire@acme.com"


def test_recovers_from_tool_error(tmp_path):
    llm = MockLLM([
        _tool("look up unknown customer", "lookup_customer", email="ghost@nowhere.com"),
        _tool("search inbox instead", "search_email", query="renewal"),
        _finish("Customer not in CRM; found their renewal email instead."),
    ])
    res = run_task("Find out about the renewal issue", llm, ToolRegistry(), Tracer(str(tmp_path)))
    assert res.ok and res.tool_errors == 1 and res.tool_calls == 2


def test_recovers_from_injected_outage(tmp_path):
    llm = MockLLM([
        _tool("first try", "search_email", query="webhook"),
        _tool("retry after transient error", "search_email", query="webhook"),
        _finish("Found it after retrying."),
    ])
    reg = ToolRegistry(inject_failures={"search_email": 1})
    res = run_task("Find the webhook email", llm, reg, Tracer(str(tmp_path)))
    assert res.ok and res.tool_errors == 1


def test_malformed_output_is_retried_then_recovers(tmp_path):
    llm = MockLLM([
        "sure! let me search the inbox",  # not JSON -> format retry
        _tool("ok, proper JSON now", "search_email", query="webhook"),
        _finish("done"),
    ])
    res = run_task("Find the webhook email", llm, ToolRegistry(), Tracer(str(tmp_path)))
    assert res.ok and res.format_retries == 1


def test_trace_is_written_and_replayable(tmp_path):
    llm = MockLLM([_finish("nothing to do")])
    res = run_task("Say hi", llm, ToolRegistry(), Tracer(str(tmp_path)))
    lines = [json.loads(l) for l in open(res.trace_file, encoding="utf-8")]
    kinds = [l["kind"] for l in lines]
    assert kinds[0] == "task" and kinds[-1] == "finish"
    assert all(l["run_id"] == lines[0]["run_id"] for l in lines)
