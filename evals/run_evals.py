"""Run the eval suite against any OpenAI-compatible model (local = $0).

  python -m evals.run_evals                 # settings from env (see agent/config.py)
  python -m evals.run_evals --repeats 3     # measure consistency, not luck

Reports per-scenario pass/fail plus aggregate metrics (success rate, steps,
tool errors, format retries, latency) and writes evals/report.md.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import yaml

from agent.config import settings
from agent.llm import OpenAICompatibleLLM
from agent.loop import RunResult, run_task
from agent.tools import ToolRegistry
from agent.tracing import Tracer


def judge(llm, task: str, answer: str, question: str) -> bool:
    """LLM-graded check on the final answer (local model = $0). Note: the
    grader is the same endpoint under test - a stated limitation."""
    text, _ = llm.complete([{"role": "user", "content":
        f"Task given to an assistant: {task}\n"
        f"Assistant's final answer: {answer}\n\n"
        f"Question: {question}\nReply with exactly YES or NO."}])
    return bool(re.search(r"\bYES\b", text.upper()))


# --- declarative checks -------------------------------------------------------
def check(c: dict, reg: ToolRegistry, res: RunResult, llm=None, task: str = "") -> bool:
    kind = c["kind"]
    w = reg.workplace
    if kind == "email_sent":
        return any(m["to"] == c["to"] for m in w.sent)
    if kind == "no_email_sent":
        return not w.sent
    if kind == "ticket_created":
        return any(t["customer"] == c["customer"] and t["priority"] == c["priority"]
                   for t in w.tickets)
    if kind == "event_created":
        return any(e["date"] == c["date"] and e["attendee"] == c["attendee"]
                   and e["time"] != c.get("not_time") for e in w.events)
    if kind == "answer_mentions":
        low = res.answer.lower()
        return any(str(k).lower() in low for k in c["any"])
    if kind == "judge":
        return res.ok and judge(llm, task, res.answer, c["question"])
    raise ValueError(f"unknown check kind: {kind}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--scenarios", default=str(Path(__file__).parent / "scenarios.yaml"))
    args = ap.parse_args()

    scenarios = yaml.safe_load(open(args.scenarios, encoding="utf-8"))
    llm = OpenAICompatibleLLM()
    rows, all_res = [], []

    for sc in scenarios:
        passes = 0
        for _ in range(args.repeats):
            reg = ToolRegistry(inject_failures=dict(sc.get("inject_failures", {})))
            t0 = time.perf_counter()
            res = run_task(sc["task"], llm, reg, Tracer(settings.trace_dir))
            wall = time.perf_counter() - t0
            ok = res.ok and all(check(c, reg, res, llm, sc["task"]) for c in sc["checks"])
            passes += ok
            all_res.append((res, wall))
            print(f"[{'PASS' if ok else 'FAIL'}] {sc['id']}  "
                  f"steps={res.steps} tool_err={res.tool_errors} "
                  f"fmt_retries={res.format_retries} {wall:.1f}s")
        rows.append((sc["id"], passes, args.repeats))

    n = len(all_res)
    total_pass = sum(p for _, p, _ in rows)
    total_runs = sum(r for _, _, r in rows)
    steps = [r.steps for r, _ in all_res]
    lat = [w for _, w in all_res]
    print("\n=== Aggregate ===")
    print(f"success: {total_pass}/{total_runs} ({total_pass / total_runs:.0%})  "
          f"avg steps: {statistics.mean(steps):.1f}  "
          f"avg wall: {statistics.mean(lat):.1f}s  "
          f"tool errors: {sum(r.tool_errors for r, _ in all_res)}  "
          f"format retries: {sum(r.format_retries for r, _ in all_res)}")

    # machine-readable results for cross-model charts
    rdir = Path(__file__).parent / "results"
    rdir.mkdir(exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9.-]+", "_", settings.model)
    (rdir / f"{safe}.json").write_text(json.dumps({
        "model": settings.model,
        "scenarios": {sid: [p, r] for sid, p, r in rows},
        "success": [total_pass, total_runs],
        "avg_steps": round(statistics.mean(steps), 2),
        "avg_wall_s": round(statistics.mean(lat), 2),
    }, indent=1))

    report = Path(__file__).parent / "report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write(f"# Eval report\n\nmodel: `{settings.model}` @ `{settings.base_url}`  \n")
        f.write(f"success: **{total_pass}/{total_runs}** · avg steps "
                f"{statistics.mean(steps):.1f} · avg wall {statistics.mean(lat):.1f}s\n\n")
        f.write("| scenario | passed |\n|---|---|\n")
        for sid, p, r in rows:
            f.write(f"| {sid} | {p}/{r} |\n")
    print(f"report -> {report}")


if __name__ == "__main__":
    main()
