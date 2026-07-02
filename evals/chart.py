"""Render the cross-model pass-rate heatmap from evals/results/*.json:
python -m evals.chart  ->  evals/results/heatmap.png"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    rdir = Path(__file__).parent / "results"
    runs = [json.loads(p.read_text()) for p in sorted(rdir.glob("*.json"))]
    if not runs:
        raise SystemExit("no results - run evals first")
    scen = list(runs[0]["scenarios"])
    mat = np.array([[r["scenarios"][s][0] / r["scenarios"][s][1] for r in runs]
                    for s in scen])
    fig, ax = plt.subplots(figsize=(1.6 + 1.7 * len(runs), 0.42 * len(scen) + 1.4))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(runs)),
                  [r["model"].replace("-uncensored-hauhaucs-aggressive", "\n(uncensored)")
                   for r in runs], fontsize=8)
    ax.set_yticks(range(len(scen)), scen, fontsize=8)
    for i in range(len(scen)):
        for j in range(len(runs)):
            p, n = runs[j]["scenarios"][scen[i]]
            ax.text(j, i, f"{p}/{n}", ha="center", va="center", fontsize=8)
    ax.set_title("Scenario pass rate by model (same harness)", fontsize=10)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    out = rdir / "heatmap.png"
    fig.savefig(out, dpi=160)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
