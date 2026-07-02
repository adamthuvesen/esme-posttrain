#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "plotly>=6.1",
#   "kaleido>=1.0",
# ]
# ///
"""Render README training-telemetry figures from post-training run artifacts.

Reads the GRPO run's `rollouts.jsonl` + `metrics.jsonl` + `best-checkpoint.json`
and the DPO run's `metrics.jsonl` + `best-checkpoint.json` (downloaded read-only
from the run Modal volumes) and exports three static SVG cards into `assets/`:

- fig-grpo-training-dynamics.svg: reward mean +-1 std, valid-expression and
  exact-solve rates over the 240 GRPO steps, best checkpoint marked.
- fig-grpo-countdown-evidence.svg: Chat-vs-RL acceptance metrics and the
  width-2 held-out transfer panel (bars from the tracked evidence docs).
- fig-dpo-training-dynamics.svg: train preference accuracy and margin over the
  960 DPO steps, accepted checkpoint marked.

    uv run scripts/plot_run_telemetry.py --output-dir assets --json

Fetch the inputs from the run Modal volumes (read-only):

    uv run --with modal==1.5.0 modal volume get esme-posttrain-esme-rlvr-countdown \
        esme-214m-rlvr-countdown-grpo-v2-ccb6287-1/rollouts.jsonl \
        runs/esme-214m-rlvr-countdown-grpo-v2-ccb6287-1/
    uv run --with modal==1.5.0 modal volume get esme-posttrain-esme-rlvr-countdown \
        esme-214m-rlvr-countdown-grpo-v2-ccb6287-1/metrics.jsonl \
        runs/esme-214m-rlvr-countdown-grpo-v2-ccb6287-1/
    uv run --with modal==1.5.0 modal volume get esme-posttrain-esme-rlvr-countdown \
        esme-214m-rlvr-countdown-grpo-v2-ccb6287-1/best-checkpoint.json \
        runs/esme-214m-rlvr-countdown-grpo-v2-ccb6287-1/
    uv run --with modal==1.5.0 modal volume get esme-posttrain-esme-chat-dpo \
        esme-214m-chat-dpo-full/metrics.jsonl runs/esme-214m-chat-dpo-full/
    uv run --with modal==1.5.0 modal volume get esme-posttrain-esme-chat-dpo \
        esme-214m-chat-dpo-full/best-checkpoint.json runs/esme-214m-chat-dpo-full/
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go

# Style contract: match esme-pretrain/scripts/plot_run_telemetry.py (which in
# turn matches grpo-decomp/results/fig-gsm8k-decomposition.svg).
FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
TITLE_COLOR = "#1f2937"
SUBTITLE_COLOR = "#6b7280"
AXIS_COLOR = "#cfd5df"
GRID_COLOR = "#eef1f6"
BORDER_COLOR = "#d9dde7"
TICK_COLOR = "#6b7280"
BLUE = "#636efa"
GREEN = "#00cc96"
RED = "#ef553b"
REFERENCE_GREY = "#9aa1ad"
BLUE_BAND = "rgba(99, 110, 250, 0.16)"
CARD_WIDTH = 920
CARD_HEIGHT = 560

# Rollouts per GRPO step; mirrors the run config (8 prompts x group size 16).
GRPO_ROLLOUTS_PER_STEP = 128
# Tolerance for cross-checking float32 aggregates logged by the trainer.
METRIC_TOLERANCE = 5e-4

# Evidence-card values are transcribed from the tracked evidence docs; the
# sources are the acceptance table in docs/rlvr-countdown-lite-grpo.md and the
# width-stratified table in docs/rlvr-countdown-heldout-transfer.md.
ACCEPTANCE_METRICS = [
    # (metric label, Esme-214M-Chat %, Esme-214M-RL %)
    ("valid-expression", 5.83, 99.38),
    ("exact-solve", 0.73, 16.35),
    ("pass@32", 13.33, 16.67),
]
HELDOUT_WIDTH2_METRICS = [
    # 2-number stratum, held-out fresh set (5 tasks), same acceptance protocol.
    ("valid-expression", 10.00, 98.75),
    ("exact-solve", 3.12, 38.75),
]


@dataclass(frozen=True)
class GrpoTelemetry:
    steps: list[int]
    reward_mean: list[float]
    reward_std: list[float]
    valid_expression_rate: list[float]
    exact_solve_rate: list[float]
    best_step: int
    best_reward_mean: float
    rollout_rows: int


@dataclass(frozen=True)
class DpoTelemetry:
    train_steps: list[int]
    preference_accuracy: list[float]
    margin: list[float]
    accepted_step: int
    accepted_eval_accuracy: float
    train_rows: int
    eval_rows: int


def load_grpo_telemetry(run_dir: Path) -> GrpoTelemetry:
    rollouts_path = run_dir / "rollouts.jsonl"
    metrics_path = run_dir / "metrics.jsonl"
    best_path = run_dir / "best-checkpoint.json"
    for path in (rollouts_path, metrics_path, best_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing run artifact: {path}")

    per_step: dict[int, list[dict]] = {}
    rollout_rows = 0
    for line_number, line in enumerate(rollouts_path.read_text().splitlines(), start=1):
        record = json.loads(line)
        for key in ("step", "reward", "is_valid_expression", "is_exact_solve"):
            if key not in record:
                raise ValueError(f"{rollouts_path}:{line_number}: rollout record missing {key}")
        per_step.setdefault(record["step"], []).append(record)
        rollout_rows += 1

    steps = sorted(per_step)
    if steps != list(range(1, len(steps) + 1)):
        raise ValueError(f"{rollouts_path}: expected contiguous steps from 1, got {steps[:5]}...")
    for step, group in per_step.items():
        if len(group) != GRPO_ROLLOUTS_PER_STEP:
            raise ValueError(
                f"{rollouts_path}: step {step} has {len(group)} rollouts,"
                f" expected {GRPO_ROLLOUTS_PER_STEP}"
            )

    reward_mean = [statistics.mean(r["reward"] for r in per_step[s]) for s in steps]
    reward_std = [statistics.pstdev(r["reward"] for r in per_step[s]) for s in steps]
    valid_rate = [statistics.mean(r["is_valid_expression"] for r in per_step[s]) for s in steps]
    exact_rate = [statistics.mean(r["is_exact_solve"] for r in per_step[s]) for s in steps]

    # Cross-check the derived per-step aggregates against every logged
    # metrics.jsonl record so a rollout/metrics divergence fails loudly.
    checked = 0
    for line_number, line in enumerate(metrics_path.read_text().splitlines(), start=1):
        record = json.loads(line)
        step = record["step"]
        if step not in per_step:
            raise ValueError(f"{metrics_path}:{line_number}: step {step} absent from rollouts")
        index = step - 1
        for key, derived in (
            ("train/reward_mean", reward_mean[index]),
            ("train/reward_std", reward_std[index]),
            ("train/valid_expression_rate", valid_rate[index]),
            ("train/exact_solve_rate", exact_rate[index]),
        ):
            if abs(record[key] - derived) > METRIC_TOLERANCE:
                raise ValueError(
                    f"{metrics_path}:{line_number}: {key}={record[key]:.6f}"
                    f" disagrees with rollouts-derived {derived:.6f}"
                )
        checked += 1
    if checked == 0:
        raise ValueError(f"{metrics_path}: no metric records to cross-check")

    best = json.loads(best_path.read_text())
    if best["selected_metric_name"] != "train/reward_mean":
        raise ValueError(f"{best_path}: unexpected selection metric {best['selected_metric_name']}")
    best_step = best["selected_step"]
    best_value = best["selected_metric_value"]
    if abs(reward_mean[best_step - 1] - best_value) > METRIC_TOLERANCE:
        raise ValueError(
            f"{best_path}: selected reward {best_value:.6f} disagrees with"
            f" rollouts-derived step-{best_step} mean {reward_mean[best_step - 1]:.6f}"
        )

    return GrpoTelemetry(
        steps=steps,
        reward_mean=reward_mean,
        reward_std=reward_std,
        valid_expression_rate=valid_rate,
        exact_solve_rate=exact_rate,
        best_step=best_step,
        best_reward_mean=best_value,
        rollout_rows=rollout_rows,
    )


def load_dpo_telemetry(run_dir: Path) -> DpoTelemetry:
    metrics_path = run_dir / "metrics.jsonl"
    best_path = run_dir / "best-checkpoint.json"
    for path in (metrics_path, best_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing run artifact: {path}")

    train_steps: list[int] = []
    accuracy: list[float] = []
    margin: list[float] = []
    eval_accuracy_by_step: dict[int, float] = {}
    for line_number, line in enumerate(metrics_path.read_text().splitlines(), start=1):
        record = json.loads(line)
        if record["event"] == "train":
            train_steps.append(record["step"])
            accuracy.append(record["train/preference_accuracy"])
            margin.append(record["train/margin"])
        elif record["event"] == "eval":
            eval_accuracy_by_step[record["step"]] = record["eval/preference_accuracy"]
        else:
            raise ValueError(f"{metrics_path}:{line_number}: unknown event {record['event']}")
    if not train_steps or not eval_accuracy_by_step:
        raise ValueError(f"{metrics_path}: expected both train and eval records")
    if train_steps != sorted(train_steps):
        raise ValueError(f"{metrics_path}: train steps are not monotonically increasing")

    best = json.loads(best_path.read_text())
    if best["selected_metric"] != "eval/preference_accuracy":
        raise ValueError(f"{best_path}: unexpected selection metric {best['selected_metric']}")
    accepted_step = best["selected_step"]
    accepted_value = best["selected_metric_value"]
    if accepted_step not in eval_accuracy_by_step:
        raise ValueError(f"{best_path}: selected step {accepted_step} has no eval record")
    if abs(eval_accuracy_by_step[accepted_step] - accepted_value) > METRIC_TOLERANCE:
        raise ValueError(
            f"{best_path}: selected accuracy {accepted_value:.6f} disagrees with"
            f" metrics.jsonl eval {eval_accuracy_by_step[accepted_step]:.6f}"
        )

    return DpoTelemetry(
        train_steps=train_steps,
        preference_accuracy=accuracy,
        margin=margin,
        accepted_step=accepted_step,
        accepted_eval_accuracy=accepted_value,
        train_rows=len(train_steps),
        eval_rows=len(eval_accuracy_by_step),
    )


def rounded_border_path(radius_px: float) -> str:
    """Rounded-rect border in paper coordinates (plotly paths have no arc command)."""
    rx = radius_px / CARD_WIDTH
    ry = radius_px / CARD_HEIGHT
    x0, x1 = 0.5 / CARD_WIDTH, 1 - 0.5 / CARD_WIDTH
    y0, y1 = 0.5 / CARD_HEIGHT, 1 - 0.5 / CARD_HEIGHT
    return (
        f"M {x0 + rx},{y0} L {x1 - rx},{y0} Q {x1},{y0} {x1},{y0 + ry} "
        f"L {x1},{y1 - ry} Q {x1},{y1} {x1 - rx},{y1} "
        f"L {x0 + rx},{y1} Q {x0},{y1} {x0},{y1 - ry} "
        f"L {x0},{y0 + ry} Q {x0},{y0} {x0 + rx},{y0} Z"
    )


def card_layout(title: str, subtitle: str, conclusion: str) -> go.Layout:
    return go.Layout(
        width=CARD_WIDTH,
        height=CARD_HEIGHT,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": FONT_FAMILY, "size": 12, "color": TICK_COLOR},
        margin={"l": 84, "r": 84, "t": 118, "b": 96},
        showlegend=False,
        shapes=[
            {
                "type": "path",
                "path": rounded_border_path(radius_px=8),
                "xref": "paper",
                "yref": "paper",
                "line": {"color": BORDER_COLOR, "width": 1},
                "layer": "above",
            }
        ],
        annotations=[
            {
                "text": f"<b>{title}</b>",
                "xref": "paper",
                "yref": "paper",
                "x": -0.045,
                "y": 1.24,
                "xanchor": "left",
                "showarrow": False,
                "font": {"family": FONT_FAMILY, "size": 22, "color": TITLE_COLOR},
            },
            {
                "text": subtitle,
                "xref": "paper",
                "yref": "paper",
                "x": -0.045,
                "y": 1.135,
                "xanchor": "left",
                "showarrow": False,
                "font": {"family": FONT_FAMILY, "size": 14, "color": SUBTITLE_COLOR},
            },
            {
                "text": conclusion,
                "xref": "paper",
                "yref": "paper",
                "x": -0.045,
                "y": -0.13,
                "xanchor": "left",
                "yanchor": "top",
                "align": "left",
                "showarrow": False,
                "font": {"family": FONT_FAMILY, "size": 12, "color": SUBTITLE_COLOR},
            },
        ],
    )


def styled_axis(**overrides: object) -> dict[str, object]:
    axis: dict[str, object] = {
        "showgrid": True,
        "gridcolor": GRID_COLOR,
        "gridwidth": 1,
        "zeroline": False,
        "showline": True,
        "linecolor": AXIS_COLOR,
        "linewidth": 1,
        "ticks": "outside",
        "tickcolor": AXIS_COLOR,
        "tickfont": {"family": FONT_FAMILY, "size": 12, "color": TICK_COLOR},
        "title": {"font": {"family": FONT_FAMILY, "size": 13, "color": "#374151"}},
    }
    axis.update(overrides)
    return axis


def trace_label(text: str, x: float, y: float, color: str) -> dict[str, object]:
    return {
        "text": text,
        "x": x,
        "y": y,
        "showarrow": False,
        "xanchor": "left",
        "font": {"family": FONT_FAMILY, "size": 13, "color": color},
    }


def build_grpo_dynamics_figure(telemetry: GrpoTelemetry) -> go.Figure:
    total_steps = telemetry.steps[-1]
    band_upper = [m + s for m, s in zip(telemetry.reward_mean, telemetry.reward_std, strict=True)]
    band_lower = [m - s for m, s in zip(telemetry.reward_mean, telemetry.reward_std, strict=True)]

    first_reward = telemetry.reward_mean[0]
    last_window_mean = statistics.mean(telemetry.reward_mean[-20:])

    figure = go.Figure(
        layout=card_layout(
            title="Esme-214M-RL: GRPO training dynamics on Countdown-Lite",
            subtitle=(
                f"Reward mean +-1 std across {GRPO_ROLLOUTS_PER_STEP} rollouts/step -"
                f" {total_steps} steps, one A100"
            ),
            conclusion=(
                f"Conclusion: reward climbs from {first_reward:.2f} to a best of"
                f" {telemetry.best_reward_mean:.2f} at step {telemetry.best_step} with no"
                f" collapse<br>(last-20-step mean {last_window_mean:.2f}); the best and final"
                f" checkpoints score identically on the held-out eval."
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=telemetry.steps,
            y=band_upper,
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=telemetry.steps,
            y=band_lower,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=BLUE_BAND,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=telemetry.steps,
            y=telemetry.reward_mean,
            mode="lines",
            name="reward mean",
            line={"color": BLUE, "width": 2.4},
        )
    )
    figure.update_layout(
        xaxis=styled_axis(title={"text": "GRPO step"}, range=[0, total_steps * 1.02]),
        yaxis=styled_axis(
            title={"text": "reward"},
            range=[-0.02, 1.12],
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
        ),
    )
    figure.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=telemetry.best_step,
        x1=telemetry.best_step,
        y0=0,
        y1=1,
        line={"color": REFERENCE_GREY, "width": 1, "dash": "dash"},
    )
    figure.add_annotation(
        {
            "text": (
                f"best checkpoint - step {telemetry.best_step}"
                f" (reward {telemetry.best_reward_mean:.2f})"
            ),
            "x": telemetry.best_step,
            "yref": "paper",
            "y": 0.97,
            "showarrow": False,
            "xanchor": "left",
            "xshift": 6,
            "font": {"family": FONT_FAMILY, "size": 12, "color": REFERENCE_GREY},
        }
    )
    return figure


def evidence_bar_pair(metrics: list[tuple[str, float, float]], xaxis: str) -> tuple[go.Bar, go.Bar]:
    labels = [label for label, _, _ in metrics]
    chat = [chat_pct for _, chat_pct, _ in metrics]
    rl = [rl_pct for _, _, rl_pct in metrics]
    text_font = {"family": FONT_FAMILY, "size": 12}
    return (
        go.Bar(
            x=labels,
            y=chat,
            name="Esme-214M-Chat",
            marker={"color": REFERENCE_GREY},
            text=[f"{value:.2f}" for value in chat],
            textposition="outside",
            textfont={**text_font, "color": SUBTITLE_COLOR},
            xaxis=xaxis,
        ),
        go.Bar(
            x=labels,
            y=rl,
            name="Esme-214M-RL",
            marker={"color": BLUE},
            text=[f"<b>{value:.2f}</b>" for value in rl],
            textposition="outside",
            textfont={**text_font, "color": TITLE_COLOR},
            xaxis=xaxis,
        ),
    )


def build_countdown_evidence_figure() -> go.Figure:
    figure = go.Figure(
        layout=card_layout(
            title="Esme-214M-RL vs Esme-214M-Chat: verifier-scored evidence",
            subtitle=(
                "Acceptance eval (30 tasks x 32 samples) vs 2-number held-out fresh tasks -"
                " same protocol, max_new_tokens=12"
            ),
            conclusion=(
                "Conclusion: GRPO lifts valid expressions 5.83% -> 99.38% and exact solves"
                " 0.73% -> 16.35% in-distribution,<br>and keeps nearly the whole gain on"
                " never-seen 2-number tasks - transfer, not memorization."
            ),
        )
    )
    chat_bars, rl_bars = evidence_bar_pair(ACCEPTANCE_METRICS, "x")
    figure.add_trace(chat_bars)
    figure.add_trace(rl_bars)
    chat_bars2, rl_bars2 = evidence_bar_pair(HELDOUT_WIDTH2_METRICS, "x2")
    figure.add_trace(chat_bars2)
    figure.add_trace(rl_bars2)

    panel_caption_font = {"family": FONT_FAMILY, "size": 13, "color": "#374151"}
    figure.update_layout(
        barmode="group",
        bargap=0.32,
        bargroupgap=0.08,
        xaxis=styled_axis(domain=[0.0, 0.44], showgrid=False),
        xaxis2=styled_axis(domain=[0.58, 1.0], showgrid=False),
        yaxis=styled_axis(
            title={"text": "% of samples"},
            range=[0, 108],
            tickvals=[0, 25, 50, 75, 100],
        ),
        annotations=list(figure.layout.annotations)
        + [
            {
                "text": "<b>A - acceptance eval (in-distribution)</b>",
                "xref": "paper",
                "yref": "paper",
                "x": 0.22,
                "y": 1.035,
                "xanchor": "center",
                "showarrow": False,
                "font": panel_caption_font,
            },
            {
                "text": "<b>B - held-out fresh, 2-number stratum</b>",
                "xref": "paper",
                "yref": "paper",
                "x": 0.79,
                "y": 1.035,
                "xanchor": "center",
                "showarrow": False,
                "font": panel_caption_font,
            },
            {
                "text": (
                    f"<span style='color:{REFERENCE_GREY}'>&#9632;</span> Esme-214M-Chat"
                    f"   <span style='color:{BLUE}'>&#9632;</span> Esme-214M-RL"
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.115,
                "xanchor": "center",
                "showarrow": False,
                "font": {"family": FONT_FAMILY, "size": 12, "color": SUBTITLE_COLOR},
            },
        ],
    )
    return figure


def build_dpo_training_figure(telemetry: DpoTelemetry) -> go.Figure:
    total_steps = telemetry.train_steps[-1]
    figure = go.Figure(
        layout=card_layout(
            title="Esme-214M-Chat: DPO preference training",
            subtitle=(
                f"Train preference accuracy and margin over {total_steps} steps -"
                " UltraFeedback-binarized pairs, beta 0.5"
            ),
            conclusion=(
                "Conclusion: the preference margin climbs steadily while train accuracy"
                f" saturates; the accepted checkpoint<br>is step {telemetry.accepted_step},"
                " where held-out preference accuracy peaks at"
                f" {telemetry.accepted_eval_accuracy * 100:.1f}%."
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=telemetry.train_steps,
            y=telemetry.preference_accuracy,
            mode="lines",
            name="train preference accuracy",
            line={"color": BLUE, "width": 2.4},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=telemetry.train_steps,
            y=telemetry.margin,
            yaxis="y2",
            mode="lines",
            name="train margin",
            line={"color": GREEN, "width": 2},
        )
    )
    figure.update_layout(
        xaxis=styled_axis(title={"text": "DPO step"}, range=[0, total_steps * 1.02]),
        yaxis=styled_axis(
            title={"text": "preference accuracy"},
            range=[0.4, 1.05],
            tickvals=[0.4, 0.6, 0.8, 1.0],
        ),
        yaxis2=styled_axis(
            title={"text": "margin"},
            overlaying="y",
            side="right",
            range=[0, 2.1],
            tickvals=[0, 0.5, 1.0, 1.5, 2.0],
            showgrid=False,
        ),
    )
    figure.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=telemetry.accepted_step,
        x1=telemetry.accepted_step,
        y0=0,
        y1=1,
        line={"color": REFERENCE_GREY, "width": 1, "dash": "dash"},
    )
    figure.add_annotation(
        {
            "text": (
                f"accepted - step {telemetry.accepted_step}"
                f" (held-out acc {telemetry.accepted_eval_accuracy * 100:.1f}%)"
            ),
            "x": telemetry.accepted_step,
            "yref": "paper",
            "y": 0.06,
            "showarrow": False,
            "xanchor": "right",
            "xshift": -6,
            "font": {"family": FONT_FAMILY, "size": 12, "color": REFERENCE_GREY},
        }
    )
    figure.add_annotation(trace_label("train preference accuracy", total_steps * 0.03, 0.98, BLUE))
    figure.add_annotation(trace_label("train margin (right axis)", total_steps * 0.45, 0.6, GREEN))
    return figure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render README telemetry SVGs for the post-training runs."
    )
    parser.add_argument(
        "--grpo-run-dir",
        type=Path,
        default=Path("runs/esme-214m-rlvr-countdown-grpo-v2-ccb6287-1"),
        help="GRPO run directory with rollouts.jsonl, metrics.jsonl, best-checkpoint.json.",
    )
    parser.add_argument(
        "--dpo-run-dir",
        type=Path,
        default=Path("runs/esme-214m-chat-dpo-full"),
        help="DPO run directory with metrics.jsonl and best-checkpoint.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    try:
        grpo = load_grpo_telemetry(args.grpo_run_dir)
        dpo = load_dpo_telemetry(args.dpo_run_dir)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "grpo_training_dynamics": args.output_dir / "fig-grpo-training-dynamics.svg",
        "grpo_countdown_evidence": args.output_dir / "fig-grpo-countdown-evidence.svg",
        "dpo_training_dynamics": args.output_dir / "fig-dpo-training-dynamics.svg",
    }
    build_grpo_dynamics_figure(grpo).write_image(
        outputs["grpo_training_dynamics"], format="svg", width=CARD_WIDTH, height=CARD_HEIGHT
    )
    build_countdown_evidence_figure().write_image(
        outputs["grpo_countdown_evidence"], format="svg", width=CARD_WIDTH, height=CARD_HEIGHT
    )
    build_dpo_training_figure(dpo).write_image(
        outputs["dpo_training_dynamics"], format="svg", width=CARD_WIDTH, height=CARD_HEIGHT
    )

    summary = {
        "grpo_run_dir": str(args.grpo_run_dir),
        "grpo_steps": len(grpo.steps),
        "grpo_rollout_rows": grpo.rollout_rows,
        "grpo_best_step": grpo.best_step,
        "grpo_best_reward_mean": grpo.best_reward_mean,
        "dpo_run_dir": str(args.dpo_run_dir),
        "dpo_train_rows": dpo.train_rows,
        "dpo_eval_rows": dpo.eval_rows,
        "dpo_accepted_step": dpo.accepted_step,
        "dpo_accepted_eval_accuracy": dpo.accepted_eval_accuracy,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    if args.as_json:
        print(json.dumps(summary, indent=2))
    else:
        for key, path in outputs.items():
            print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
