"""Render the cross-model pass-rate heatmap from evals/results/*.json:
python -m evals.chart  ->  evals/results/heatmap.png"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # render headless, before pyplot import
import matplotlib.pyplot as plt  # noqa: E402


def _label(run: dict) -> str:
    name = run["model"].replace("-uncensored-hauhaucs-aggressive", "\n(uncensored)")
    if not run.get("defenses", True):
        name += "\n[no defenses]"
    return name


def main() -> None:
    rdir = Path(__file__).parent / "results"
    runs = [json.loads(p.read_text()) for p in sorted(rdir.glob("*.json"))]
    if not runs:
        raise SystemExit("no results - run evals first")
    scen: list[str] = []
    for r in runs:  # union, keeping first-seen order (older runs had fewer scenarios)
        scen += [s for s in r["scenarios"] if s not in scen]
    mat = np.array([[r["scenarios"][s][0] / r["scenarios"][s][1]
                     if s in r["scenarios"] else np.nan for r in runs]
                    for s in scen])
    fig, ax = plt.subplots(figsize=(1.6 + 1.7 * len(runs), 0.42 * len(scen) + 1.4))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(runs)), [_label(r) for r in runs], fontsize=8)
    ax.set_yticks(range(len(scen)), scen, fontsize=8)
    for i, s in enumerate(scen):
        for j, r in enumerate(runs):
            if s in r["scenarios"]:
                p, n = r["scenarios"][s]
                ax.text(j, i, f"{p}/{n}", ha="center", va="center", fontsize=8)
    ax.set_title("Scenario pass rate by model (same harness)", fontsize=10)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    out = rdir / "heatmap.png"
    fig.savefig(out, dpi=160)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
