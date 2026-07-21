"""One entry point for the whole toolkit.

  auditable-agent run "find the webhook email and open a ticket"
  auditable-agent replay traces/<id>.jsonl     # or: replay --last
  auditable-agent eval --repeats 3
  auditable-agent chart

`run` executes a single task against the configured endpoint (see
agent/config.py for AGENT_* env vars) and prints the answer, run metrics,
and the trace file — which `replay --last` then renders step by step.
"""
from __future__ import annotations

import argparse
import sys


def _cmd_run(args: argparse.Namespace) -> None:
    from agent.config import settings
    from agent.llm import OpenAICompatibleLLM
    from agent.loop import run_task
    from agent.tools import ToolRegistry

    res = run_task(args.task, OpenAICompatibleLLM(), ToolRegistry())
    print(f"\n{'─' * 60}\nanswer: {res.answer}")
    print(f"ok={res.ok} steps={res.steps} tool_calls={res.tool_calls} "
          f"tool_errors={res.tool_errors} format_retries={res.format_retries} "
          f"policy_blocks={res.policy_blocks}")
    print(f"model:  {settings.model} @ {settings.base_url}")
    print(f"trace:  {res.trace_file}   (replay: auditable-agent replay --last)")


def _cmd_replay(args: argparse.Namespace) -> None:
    from agent import replay

    replay.main(["--last"] if args.last or not args.trace else [args.trace])


def _cmd_eval(args: argparse.Namespace, extra: list[str]) -> None:
    from evals.run_evals import main as eval_main

    eval_main(extra)


def _cmd_chart(args: argparse.Namespace) -> None:
    try:
        from evals.chart import main as chart_main
    except ImportError:
        raise SystemExit(
            "matplotlib is required for charts:  pip install 'auditable-agent[viz]'"
        ) from None
    chart_main()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="auditable-agent", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run one task against the configured model")
    p_run.add_argument("task", help="natural-language task for the agent")

    p_replay = sub.add_parser("replay", help="pretty-print a JSONL trace")
    p_replay.add_argument("trace", nargs="?", help="path to traces/<id>.jsonl")
    p_replay.add_argument("--last", action="store_true", help="replay the newest trace")

    sub.add_parser("eval", help="run the eval suite (flags passed through)",
                   add_help=False)
    sub.add_parser("chart", help="render the cross-model heatmap")

    argv = sys.argv[1:] if argv is None else argv
    # `eval` owns its flags — split them off before parsing so argparse
    # doesn't reject e.g. `auditable-agent eval --repeats 3`.
    if argv and argv[0] == "eval":
        args, extra = ap.parse_args(["eval"]), argv[1:]
    else:
        args, extra = ap.parse_args(argv), []

    {"run": _cmd_run, "replay": _cmd_replay, "chart": _cmd_chart,
     "eval": lambda a: _cmd_eval(a, extra)}[args.cmd](args)


if __name__ == "__main__":
    main()
