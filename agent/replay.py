"""Replay a run trace in the terminal:  python -m agent.replay traces/<id>.jsonl

Also available as `auditable-agent replay <trace>` (or `--last` for the most
recent trace).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from agent.config import settings

C = {"task": "\033[1;36m", "llm": "\033[90m", "tool": "\033[1;33m",
     "schema_deviation": "\033[35m", "format_error": "\033[31m",
     "budget_warning": "\033[41m", "policy_block": "\033[1;35m",
     "finish": "\033[1;32m", "abort": "\033[1;31m"}
R = "\033[0m"


def latest_trace(trace_dir: str | None = None) -> Path:
    d = Path(trace_dir or settings.trace_dir)
    traces = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not traces:
        raise SystemExit(f"no traces in {d}/")
    return traces[-1]


def render(path: str | Path) -> None:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        k, c = e["kind"], C.get(e["kind"], "")
        if k == "task":
            print(f"{c}■ TASK{R} {e['task'].strip()}  (model: {e['model']})")
        elif k == "llm":
            raw = e["raw"].strip().replace("\n", " ")
            print(f"{c}  · llm {e['latency_s']}s  {raw[:140]}{R}")
        elif k == "tool":
            mark = "✓" if e["ok"] else "✗"
            print(f"{c}  {mark} {e['tool']}({json.dumps(e['args'], ensure_ascii=False)})"
                  f" → {str(e['observation'])[:120]}{R}")
        elif k == "policy_block":
            print(f"{c}  ⛔ {e['tool']} blocked: {e['reason']}{R}")
        elif k == "finish":
            print(f"{c}■ FINISH{R} {e['answer'][:300]}")
        elif k == "abort":
            print(f"{c}■ ABORT{R} {e['reason']}")
        else:
            rest = {x: y for x, y in e.items() if x not in ("run_id", "ts", "step", "kind")}
            print(f"{c}  ! {k}: {rest}{R}")


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] == "--last":
        render(latest_trace())
    else:
        render(args[0])


if __name__ == "__main__":
    main()
