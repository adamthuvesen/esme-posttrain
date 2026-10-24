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

from esme_posttrain.studies.report import (  # noqa: E402
    _parse_countdown_key,
    _read_jsonl,
    _score_completion_rows,
    _score_countdown_sample,
)

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


def sum_task_ids() -> set[str]:
    path = REPO_ROOT / "data" / "rl" / "countdown_heldout" / "heldout_large.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {row["task_id"] for row in rows if row["target"] == sum(row["numbers"])}


def score(directory: Path, *, samples: int = 16) -> dict[str, object]:
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    expected = {"n": samples, "temperature": 1.0, "max_new_tokens": 12, "seed": 0}
    actual = {key: provenance["sampling"][key] for key in expected}
    if actual != expected or provenance["dataset"]["config"] != "heldout_large":
        raise ValueError(f"{directory} was sampled with {actual}, expected {expected}")
    rows = _read_jsonl(directory / "completions.jsonl")
    metrics, per_problem = _score_completion_rows(rows)
    if metrics["problem_count"] != TASKS:
        raise ValueError(f"{directory} has {metrics['problem_count']} problems, expected {TASKS}")
    # A valid expression with only "+" uses every supplied number once, so it is their sum.
    sum_template = 0
    for row in rows:
        target, numbers = _parse_countdown_key(row["problem"]["gold_answer"])
        for sample in row["samples"]:
            expression = sample[7:-1]
            if (
                set(expression) <= set("0123456789+ ")
                and _score_countdown_sample(
                    sample, numbers=numbers, target=target
                ).is_valid_expression
            ):
                sum_template += 1
    return {
        "valid_expression_rate": metrics["valid_expression_rate"],
        "exact_solve_rate": metrics["exact_solve_rate"],
        "problems_solved": metrics["any_exact_solved"],
        "sum_template_rate": sum_template / metrics["sample_count"],
        "solved_ids": {k for k, v in per_problem.items() if v["any_exact_solve_rate"]},
        "per_problem": per_problem,
    }


def unbiased_pass_at_k(samples: int, correct: int, k: int) -> float:
    # Chen et al. (2021): 1 - C(n - c, k) / C(n, k).
    if samples - correct < k:
        return 1.0
    return 1.0 - math.comb(samples - correct, k) / math.comb(samples, k)


def summarize(root: Path) -> dict[str, object]:
    sums = sum_task_ids()

    def public(result: dict[str, object]) -> dict[str, object]:
        solved = result["solved_ids"]
        assert isinstance(solved, set)
        return {
            "valid_expression_rate": result["valid_expression_rate"],
            "exact_solve_rate": result["exact_solve_rate"],
            "problems_solved": result["problems_solved"],
            "sum_template_rate": result["sum_template_rate"],
            "solved_sum_tasks": len(solved & sums),
            "solved_other_tasks": len(solved - sums),
        }

    chat = score(root / "heldout_large-chat")
    real_union: set[str] = set()
    arms: dict[str, object] = {}
    for key, (name, seeds) in ARMS.items():
        results = [score(root / f"heldout_large-{key}-seed{s}-step240") for s in seeds]
        solved_sets = [r["solved_ids"] for r in results]
        union = set().union(*solved_sets)
        if key == "real":
            real_union = union
        arms[key] = {
            "name": name,
            "runs": [{"seed": s, **public(r)} for s, r in zip(seeds, results, strict=True)],
            "mean_valid_expression_rate": statistics.fmean(
                float(r["valid_expression_rate"]) for r in results
            ),
            "mean_problems_solved": statistics.fmean(float(r["problems_solved"]) for r in results),
            "mean_sum_template_rate": statistics.fmean(
                float(r["sum_template_rate"]) for r in results
            ),
            "problems_solved_by_any_seed": len(union),
            "problems_solved_by_every_seed": len(set.intersection(*solved_sets)),
            "solved_by_any_seed_and_no_full_recipe_seed": len(union - real_union),
        }
    curves: dict[str, object] = {}
    for key in ("real", "placebo"):
        points = []
        for step in CURVE_STEPS:
            per_seed = [
                public(score(root / f"heldout_large-{key}-seed{s}-step{step}")) for s in CURVE_SEEDS
            ]
            points.append(
                {"step": step}
                | {
                    field: statistics.fmean(float(m[field]) for m in per_seed)
                    for field in per_seed[0]
                }
            )
        curves[key] = points
    pass_at_k: dict[str, object] = {}
    solved_at_256: dict[str, set[str]] = {}
    for key, label in (
        ("chat", "passk-heldout_large-chat"),
        ("real", "passk-heldout_large-real-seed214-step240"),
    ):
        result = score(root / label, samples=256)
        per_problem = result["per_problem"]
        assert isinstance(per_problem, dict)
        correct = [round(v["exact_solve_rate"] * 256) for v in per_problem.values()]
        solved = result["solved_ids"]
        assert isinstance(solved, set)
        solved_at_256[key] = solved
        pass_at_k[key] = {
            "samples_per_problem": 256,
            "problems_solved": len(solved),
            "solved_sum_tasks": len(solved & sums),
            "pass_at_k": {
                str(k): statistics.fmean(unbiased_pass_at_k(256, c, k) for c in correct)
                for k in PASS_K
            },
        }
    pass_at_k["overlap"] = {
        "both": len(solved_at_256["chat"] & solved_at_256["real"]),
        "chat_only": len(solved_at_256["chat"] - solved_at_256["real"]),
        "rl_only": len(solved_at_256["real"] - solved_at_256["chat"]),
    }
    return {
        "schema_version": 1,
        "task_set": "heldout_large",
        "tasks": TASKS,
        "sum_tasks": len(sums),
        "decoding": "n=16 (pass@k: n=256), temperature 1.0, 12 new tokens, seed 0",
        "chat": public(chat),
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
            title="The graded reward collapses to one template",
            subtitle="Step 240, 200 held-out tasks, 16 samples each, one point per seed",
            conclusion=(
                "With the graded reward, ~98% of samples just add all the numbers. That solves the"
                " 13 tasks whose target is the sum, and nothing else.<br>A binary exact-only reward"
                " adds the numbers half as often and solves 15-18 other tasks."
            ),
        )
    )
    for key, arm in arms.items():
        figure.add_trace(
            go.Scatter(
                x=[100 * run["sum_template_rate"] for run in arm["runs"]],
                y=[run["solved_other_tasks"] for run in arm["runs"]],
                mode="markers",
                marker={"color": colors[key], "size": 11, "opacity": 0.85},
                name=arm["name"],
            )
        )
    offsets = {"real": 0.3, "nostab": 1.4, "stdnorm": -1.6}
    labels = [
        trace_label(
            arm["name"],
            100 * arm["mean_sum_template_rate"] + 2.5,
            statistics.fmean(run["solved_other_tasks"] for run in arm["runs"])
            + offsets.get(key, 0.0),
            colors[key],
        )
        for key, arm in arms.items()
    ]
    figure.update_layout(
        xaxis=styled_axis(title={"text": "samples that add all the numbers (%)"}, range=[-5, 135]),
        yaxis=styled_axis(
            title={"text": "problems solved beyond the 13 sum tasks"}, range=[-3, 20]
        ),
        annotations=list(figure.layout.annotations) + labels,
    )
    return figure


def curves_figure(summary: dict) -> go.Figure:
    curves = summary["learning_curves"]
    chat = summary["chat"]
    figure = go.Figure(
        layout=card_layout(
            title="GRPO converges on adding the numbers",
            subtitle="Full recipe, mean of seeds 215-217 per saved checkpoint, 200 held-out tasks",
            conclusion=(
                "By step 40 the model solves 7 tasks that are not sums. As the sum template takes"
                " over the samples, those solves disappear.<br>The random-reward placebo"
                " (not shown) stays at the base model on every measure."
            ),
        )
    )
    steps = [0] + [point["step"] for point in curves["real"]]
    series = (
        (
            "sum_template_rate",
            100,
            BLUE,
            "solid",
            "samples that add all the numbers, % (left)",
            "y",
        ),
        ("solved_sum_tasks", 1, GREEN, "dot", "sum tasks solved (right)", "y2"),
        ("solved_other_tasks", 1, RED, "dot", "other tasks solved (right)", "y2"),
    )
    for field, scale, color, dash, name, axis in series:
        figure.add_trace(
            go.Scatter(
                x=steps,
                y=[scale * chat[field]] + [scale * point[field] for point in curves["real"]],
                mode="lines+markers",
                line={"color": color, "width": 2.5, "dash": dash},
                name=name,
                yaxis=axis,
            )
        )
    figure.update_layout(
        xaxis=styled_axis(title={"text": "GRPO step (0 = Esme-214M-Chat)"}),
        yaxis=styled_axis(title={"text": "% of samples"}, range=[0, 105]),
        yaxis2=styled_axis(
            title={"text": "problems solved (of 200)"},
            overlaying="y",
            side="right",
            range=[0, 14],
            dtick=2,
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
                "RL leads up to k = 64 because it nearly always adds the numbers, which solves the"
                " 13 sum tasks.<br>With 256 samples the base model solves 28 tasks (19 not sums),"
                " the RL model 19 (6 not sums)."
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
