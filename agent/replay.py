"""Replay a run trace in the terminal:  python -m agent.replay traces/<id>.jsonl"""
from __future__ import annotations

import json
import sys

C = {"task": "\033[1;36m", "llm": "\033[90m", "tool": "\033[1;33m",
     "schema_deviation": "\033[35m", "format_error": "\033[31m",
     "budget_warning": "\033[41m", "finish": "\033[1;32m", "abort": "\033[1;31m"}
R = "\033[0m"


def main() -> None:
    path = sys.argv[1]
    for line in open(path, encoding="utf-8"):
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
        elif k == "finish":
            print(f"{c}■ FINISH{R} {e['answer'][:300]}")
        elif k == "abort":
            print(f"{c}■ ABORT{R} {e['reason']}")
        else:
            print(f"{c}  ! {k}: { {x: y for x, y in e.items() if x not in ('run_id','ts','step','kind')} }{R}")


if __name__ == "__main__":
    main()
