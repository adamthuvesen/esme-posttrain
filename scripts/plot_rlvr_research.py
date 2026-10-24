#!/usr/bin/env python3
"""Render the RLVR research figures and their summary data from heldout_large CompletionSets.

Reads the CompletionSets that `scripts/modal_emit_completions.py` wrote (one directory
per label, 200 `heldout_large` tasks) and writes:

- `studies/rlvr-research-summary.json`: every number the figures and the research
  report use, so they stay citable from a clean clone.
- `assets/fig-reward-shape-tradeoff.svg`: validity vs distinct problems solved, per arm.
- `assets/fig-learning-curves.svg`: validity and problems solved over training steps,
  real reward vs random-reward placebo.
- `assets/fig-pass-at-k.svg`: unbiased pass@k for k = 1..256, Chat vs RL.

    uv run --with plotly --with kaleido python scripts/plot_rlvr_research.py \
        --completions runs/gpu-emit/batch1

It imports the repo's scorer, so it runs in the project environment.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import plotly.graph_objects as go
from plot_run_telemetry import (
    BLUE,
    CARD_HEIGHT,
    CARD_WIDTH,
    FONT_FAMILY,
    GREEN,
    RED,
    REFERENCE_GREY,
    SUBTITLE_COLOR,
    card_layout,
    styled_axis,
    trace_label,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from esme_posttrain.studies.report import _read_jsonl, _score_completion_rows  # noqa: E402

TASKS = 200
CURVE_SEEDS = (215, 216, 217)
CURVE_STEPS = (20, 40, 60, 80, 100, 120, 160, 200, 240)
ARMS = {
    # arm key: (display name, seeds)
    "placebo": ("Random-reward placebo", (214, 215, 216, 217, 218, 219)),
    "real": ("Full recipe", (214, 215, 216, 217, 218, 219)),
    "nostab": ("No stability techniques", (215, 216, 217)),
    "stdnorm": ("Std-normalized advantages", (215, 216, 217)),
    "binary": ("Binary exact-only reward", (215, 216, 217)),
}
PASS_K = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def score(directory: Path) -> tuple[dict[str, float | int], dict[str, dict[str, float]]]:
    metrics, per_problem = _score_completion_rows(_read_jsonl(directory / "completions.jsonl"))
    if metrics["problem_count"] != TASKS:
        raise ValueError(f"{directory} has {metrics['problem_count']} problems, expected {TASKS}")
    return metrics, per_problem


def unbiased_pass_at_k(samples: int, correct: int, k: int) -> float:
    # Chen et al. (2021): 1 - C(n - c, k) / C(n, k).
    if samples - correct < k:
        return 1.0
    return 1.0 - math.comb(samples - correct, k) / math.comb(samples, k)


def summarize(root: Path) -> dict[str, object]:
    chat, _ = score(root / "heldout_large-chat")
    arms: dict[str, object] = {}
    for key, (name, seeds) in ARMS.items():
        runs = []
        for seed in seeds:
            metrics, per_problem = score(root / f"heldout_large-{key}-seed{seed}-step240")
            runs.append(
                {
                    "seed": seed,
                    "valid_expression_rate": metrics["valid_expression_rate"],
                    "exact_solve_rate": metrics["exact_solve_rate"],
                    "problems_solved": metrics["any_exact_solved"],
                    "solved_ids": sorted(
                        k for k, v in per_problem.items() if v["any_exact_solve_rate"]
                    ),
                }
            )
        union = set().union(*(set(run["solved_ids"]) for run in runs))
        arms[key] = {
            "name": name,
            "runs": [{k: v for k, v in run.items() if k != "solved_ids"} for run in runs],
            "mean_valid_expression_rate": statistics.fmean(
                r["valid_expression_rate"] for r in runs
            ),
            "mean_problems_solved": statistics.fmean(r["problems_solved"] for r in runs),
            "problems_solved_by_any_seed": len(union),
            "problems_solved_by_every_seed": len(
                set.intersection(*(set(run["solved_ids"]) for run in runs))
            ),
        }
    curves: dict[str, object] = {}
    for key in ("real", "placebo"):
        points = []
        for step in CURVE_STEPS:
            per_seed = [
                score(root / f"heldout_large-{key}-seed{s}-step{step}")[0] for s in CURVE_SEEDS
            ]
            points.append(
                {
                    "step": step,
                    "valid_expression_rate": statistics.fmean(
                        m["valid_expression_rate"] for m in per_seed
                    ),
                    "problems_solved": statistics.fmean(m["any_exact_solved"] for m in per_seed),
                }
            )
        curves[key] = points
    pass_at_k: dict[str, object] = {}
    for key, label in (
        ("chat", "passk-heldout_large-chat"),
        ("real", "passk-heldout_large-real-seed214-step240"),
    ):
        metrics, per_problem = score(root / label)
        samples = int(metrics["samples_per_problem"])
        correct = [round(v["exact_solve_rate"] * samples) for v in per_problem.values()]
        pass_at_k[key] = {
            "samples_per_problem": samples,
            "pass_at_k": {
                str(k): statistics.fmean(unbiased_pass_at_k(samples, c, k) for c in correct)
                for k in PASS_K
            },
        }
    return {
        "schema_version": 1,
        "task_set": "heldout_large",
        "tasks": TASKS,
        "decoding": "n=16 (pass@k: n=256), temperature 1.0, 12 new tokens, seed 0",
        "chat": {
            "valid_expression_rate": chat["valid_expression_rate"],
            "problems_solved": chat["any_exact_solved"],
        },
        "arms_step240": arms,
        "learning_curves": curves,
        "pass_at_k": pass_at_k,
    }


def tradeoff_figure(summary: dict) -> go.Figure:
    arms = summary["arms_step240"]
    colors = {
        "placebo": REFERENCE_GREY,
        "real": BLUE,
        "nostab": "#ab63fa",
        "stdnorm": GREEN,
        "binary": RED,
    }
    figure = go.Figure(
        layout=card_layout(
            title="Reward shape trades validity for coverage",
            subtitle="Step 240, 200 held-out Countdown tasks, 16 samples each, one point per seed",
            conclusion=(
                "The graded reward makes ~98% of answers valid, and every seed solves the same"
                " ~13 problems.<br>A binary exact-only reward is valid far less often but solves"
                " about twice as many problems. Stability techniques change nothing."
            ),
        )
    )
    for key, arm in arms.items():
        figure.add_trace(
            go.Scatter(
                x=[100 * run["valid_expression_rate"] for run in arm["runs"]],
                y=[run["problems_solved"] for run in arm["runs"]],
                mode="markers",
                marker={"color": colors[key], "size": 11, "opacity": 0.85},
                name=arm["name"],
            )
        )
    labels = [
        trace_label(
            arm["name"],
            100 * arm["mean_valid_expression_rate"] + 2.5,
            arm["mean_problems_solved"]
            + (1.6 if key == "nostab" else -1.6 if key == "stdnorm" else 0),
            colors[key],
        )
        for key, arm in arms.items()
    ]
    labels[-1]["xanchor"] = "left"
    figure.update_layout(
        xaxis=styled_axis(title={"text": "valid expressions (% of samples)"}, range=[-5, 135]),
        yaxis=styled_axis(title={"text": "distinct problems solved (of 200)"}, rangemode="tozero"),
        annotations=list(figure.layout.annotations) + labels,
    )
    return figure


def curves_figure(summary: dict) -> go.Figure:
    curves = summary["learning_curves"]
    figure = go.Figure(
        layout=card_layout(
            title="Coverage peaks early, then contracts",
            subtitle="Mean of seeds 215-217 at each saved checkpoint, 200 held-out tasks",
            conclusion=(
                "With real reward, problems solved peak at step 40 (19 of 200) and then fall to 13"
                " as validity rises to 98%.<br>The random-reward placebo stays at the base model"
                " on both measures."
            ),
        )
    )
    chat = summary["chat"]
    for key, color, name in (("real", BLUE, "Full recipe"), ("placebo", REFERENCE_GREY, "Placebo")):
        steps = [0] + [p["step"] for p in curves[key]]
        figure.add_trace(
            go.Scatter(
                x=steps,
                y=[100 * chat["valid_expression_rate"]]
                + [100 * p["valid_expression_rate"] for p in curves[key]],
                mode="lines+markers",
                line={"color": color, "width": 2.5},
                name=f"{name}: valid (left)",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=steps,
                y=[chat["problems_solved"]] + [p["problems_solved"] for p in curves[key]],
                mode="lines+markers",
                line={"color": color, "width": 2.5, "dash": "dot"},
                name=f"{name}: problems solved (right)",
                yaxis="y2",
            )
        )
    figure.update_layout(
        xaxis=styled_axis(title={"text": "GRPO step (0 = Esme-214M-Chat)"}),
        yaxis=styled_axis(title={"text": "valid expressions (% of samples)"}, range=[0, 105]),
        yaxis2=styled_axis(
            title={"text": "distinct problems solved (of 200)"},
            overlaying="y",
            side="right",
            range=[0, 21],
            showgrid=False,
        ),
        showlegend=True,
        legend={
            "x": 0.98,
            "y": 0.45,
            "yanchor": "top",
            "xanchor": "right",
            "font": {"family": FONT_FAMILY, "size": 12, "color": SUBTITLE_COLOR},
        },
    )
    return figure


def pass_at_k_figure(summary: dict) -> go.Figure:
    data = summary["pass_at_k"]
    figure = go.Figure(
        layout=card_layout(
            title="RL wins at small k, the base model at large k",
            subtitle="Unbiased pass@k on 200 held-out tasks, 256 samples per task, temperature 1.0",
            conclusion=(
                "RL solves its few problems on almost every sample, so it leads up to k = 64."
                "<br>With 256 samples the base model solves 28 distinct problems,"
                " the RL model 19."
            ),
        )
    )
    for key, color, name in (
        ("chat", REFERENCE_GREY, "Esme-214M-Chat"),
        ("real", BLUE, "Esme-214M-RL"),
    ):
        figure.add_trace(
            go.Scatter(
                x=list(PASS_K),
                y=[100 * data[key]["pass_at_k"][str(k)] for k in PASS_K],
                mode="lines+markers",
                line={"color": color, "width": 2.5},
                name=name,
            )
        )
    figure.update_layout(
        xaxis=styled_axis(
            title={"text": "k (samples per task)"}, type="log", tickvals=list(PASS_K)
        ),
        yaxis=styled_axis(title={"text": "pass@k (% of tasks)"}, rangemode="tozero"),
        showlegend=True,
        legend={
            "x": 0.02,
            "y": 0.98,
            "font": {"family": FONT_FAMILY, "size": 12, "color": SUBTITLE_COLOR},
        },
    )
    return figure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the RLVR research figures and summary.")
    parser.add_argument("--completions", type=Path, default=Path("runs/gpu-emit/batch1"))
    parser.add_argument("--summary", type=Path, default=Path("studies/rlvr-research-summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    args = parser.parse_args(argv)
    try:
        summary = summarize(args.completions)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name, builder in (
        ("fig-reward-shape-tradeoff.svg", tradeoff_figure),
        ("fig-learning-curves.svg", curves_figure),
        ("fig-pass-at-k.svg", pass_at_k_figure),
    ):
        builder(summary).write_image(
            args.output_dir / name, format="svg", width=CARD_WIDTH, height=CARD_HEIGHT
        )
        print(args.output_dir / name)
    print(args.summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
