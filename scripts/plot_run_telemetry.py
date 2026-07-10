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
and the SFT and DPO runs' `metrics.jsonl` + `best-checkpoint.json` (downloaded
read-only from the run Modal volumes) and exports four static SVG cards into
`assets/`:

- fig-sft-training-dynamics.svg: train loss and held-out response loss over the
  7500 SFT steps, accepted (early-stopped) checkpoint marked.
- fig-dpo-training-dynamics.svg: train and held-out preference accuracy over
  the 960 DPO steps, accepted checkpoint marked.
- fig-grpo-training-dynamics.svg: reward mean +-1 std, valid-expression and
  exact-solve rates over the 240 GRPO steps, best checkpoint marked.
- fig-grpo-countdown-evidence.svg: Chat-vs-RL sample-level acceptance metrics
  and the unseen-2-number transfer panel (bars from the tracked evidence docs).

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
from plotly.subplots import make_subplots

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
# The evidence card has two side-by-side panels plus a reader-facing conclusion,
# so it needs more vertical air than the single-panel training cards.
EVIDENCE_CARD_WIDTH = 1120
EVIDENCE_CARD_HEIGHT = 700
# The DPO card is a two-panel 1x2 (train | held-out zoomed), so it is wider.
DPO_CARD_WIDTH = 1180
DPO_CARD_HEIGHT = 600

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
    eval_steps: list[int]
    eval_preference_accuracy: list[float]
    accepted_step: int
    accepted_eval_accuracy: float
    train_rows: int
    eval_rows: int


@dataclass(frozen=True)
class SftTelemetry:
    train_steps: list[int]
    train_loss: list[float]
    eval_steps: list[int]
    eval_response_loss: list[float]
    accepted_step: int
    accepted_response_loss: float
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
    eval_accuracy_by_step: dict[int, float] = {}
    for line_number, line in enumerate(metrics_path.read_text().splitlines(), start=1):
        record = json.loads(line)
        if record["event"] == "train":
            train_steps.append(record["step"])
            accuracy.append(record["train/preference_accuracy"])
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

    eval_steps = [step for step in sorted(eval_accuracy_by_step) if step > 0]

    return DpoTelemetry(
        train_steps=train_steps,
        preference_accuracy=accuracy,
        eval_steps=eval_steps,
        eval_preference_accuracy=[eval_accuracy_by_step[s] for s in eval_steps],
        accepted_step=accepted_step,
        accepted_eval_accuracy=accepted_value,
        train_rows=len(train_steps),
        eval_rows=len(eval_accuracy_by_step),
    )


def load_sft_telemetry(run_dir: Path) -> SftTelemetry:
    metrics_path = run_dir / "metrics.jsonl"
    best_path = run_dir / "best-checkpoint.json"
    for path in (metrics_path, best_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing run artifact: {path}")

    train_steps: list[int] = []
    train_loss: list[float] = []
    response_loss_by_step: dict[int, float] = {}
    for line_number, line in enumerate(metrics_path.read_text().splitlines(), start=1):
        record = json.loads(line)
        if record["event"] == "train":
            train_steps.append(record["step"])
            train_loss.append(record["train/loss"])
        elif record["event"] == "eval":
            response_loss_by_step[record["step"]] = record["eval/matched/response_loss"]
        else:
            raise ValueError(f"{metrics_path}:{line_number}: unknown event {record['event']}")
    if not train_steps or not response_loss_by_step:
        raise ValueError(f"{metrics_path}: expected both train and eval records")
    if train_steps != sorted(train_steps):
        raise ValueError(f"{metrics_path}: train steps are not monotonically increasing")

    best = json.loads(best_path.read_text())
    if best["selected_metric"] != "eval/matched/response_loss":
        raise ValueError(f"{best_path}: unexpected selection metric {best['selected_metric']}")
    accepted_step = best["selected_step"]
    accepted_value = best["selected_metric_value"]
    if accepted_step not in response_loss_by_step:
        raise ValueError(f"{best_path}: selected step {accepted_step} has no eval record")
    if abs(response_loss_by_step[accepted_step] - accepted_value) > METRIC_TOLERANCE:
        raise ValueError(
            f"{best_path}: selected response loss {accepted_value:.6f} disagrees with"
            f" metrics.jsonl eval {response_loss_by_step[accepted_step]:.6f}"
        )
    if accepted_value != min(response_loss_by_step.values()):
        raise ValueError(
            f"{best_path}: selected response loss {accepted_value:.6f} is not the"
            f" minimum eval value {min(response_loss_by_step.values()):.6f}"
        )

    eval_steps = sorted(response_loss_by_step)
    return SftTelemetry(
        train_steps=train_steps,
        train_loss=train_loss,
        eval_steps=eval_steps,
        eval_response_loss=[response_loss_by_step[s] for s in eval_steps],
        accepted_step=accepted_step,
        accepted_response_loss=accepted_value,
        train_rows=len(train_steps),
        eval_rows=len(eval_steps),
    )


def rounded_border_path(
    radius_px: float, width: int = CARD_WIDTH, height: int = CARD_HEIGHT
) -> str:
    """Rounded-rect border in paper coordinates (plotly paths have no arc command)."""
    rx = radius_px / width
    ry = radius_px / height
    x0, x1 = 0.5 / width, 1 - 0.5 / width
    y0, y1 = 0.5 / height, 1 - 0.5 / height
    return (
        f"M {x0 + rx},{y0} L {x1 - rx},{y0} Q {x1},{y0} {x1},{y0 + ry} "
        f"L {x1},{y1 - ry} Q {x1},{y1} {x1 - rx},{y1} "
        f"L {x0 + rx},{y1} Q {x0},{y1} {x0},{y1 - ry} "
        f"L {x0},{y0 + ry} Q {x0},{y0} {x0 + rx},{y0} Z"
    )


def rounded_panel_border_path(
    x0: float,
    x1: float,
    *,
    radius_px: float = 8,
    width: int = CARD_WIDTH,
    height: int = CARD_HEIGHT,
) -> str:
    """Rounded border for one subplot domain in paper coordinates."""
    px = 0.5 / width
    py = 0.5 / height
    left = x0 + px
    right = x1 - px
    bottom = py
    top = 1 - py
    rx = min(radius_px / width, (right - left) / 2)
    ry = min(radius_px / height, (top - bottom) / 2)
    return (
        f"M {left + rx},{bottom} L {right - rx},{bottom} "
        f"Q {right},{bottom} {right},{bottom + ry} "
        f"L {right},{top - ry} Q {right},{top} {right - rx},{top} "
        f"L {left + rx},{top} Q {left},{top} {left},{top - ry} "
        f"L {left},{bottom + ry} Q {left},{bottom} {left + rx},{bottom} Z"
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


def evidence_card_layout(title: str, subtitle: str, conclusion: str) -> go.Layout:
    return go.Layout(
        width=EVIDENCE_CARD_WIDTH,
        height=EVIDENCE_CARD_HEIGHT,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": FONT_FAMILY, "size": 13, "color": TICK_COLOR},
        margin={"l": 104, "r": 72, "t": 166, "b": 154},
        showlegend=False,
        shapes=[
            {
                "type": "path",
                "path": rounded_border_path(
                    radius_px=8, width=EVIDENCE_CARD_WIDTH, height=EVIDENCE_CARD_HEIGHT
                ),
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
                "x": -0.05,
                "y": 1.27,
                "xanchor": "left",
                "showarrow": False,
                "font": {"family": FONT_FAMILY, "size": 26, "color": TITLE_COLOR},
            },
            {
                "text": subtitle,
                "xref": "paper",
                "yref": "paper",
                "x": -0.05,
                "y": 1.18,
                "xanchor": "left",
                "showarrow": False,
                "font": {"family": FONT_FAMILY, "size": 16, "color": SUBTITLE_COLOR},
            },
            {
                "text": conclusion,
                "xref": "paper",
                "yref": "paper",
                "x": -0.05,
                "y": -0.22,
                "xanchor": "left",
                "yanchor": "top",
                "align": "left",
                "showarrow": False,
                "font": {"family": FONT_FAMILY, "size": 13.5, "color": SUBTITLE_COLOR},
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
    # Anchor the label away from the nearest plot edge so a best step late in
    # the run does not push the text off the card.
    late_best = telemetry.best_step > total_steps * 0.7
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
            "xanchor": "right" if late_best else "left",
            "xshift": -6 if late_best else 6,
            "font": {"family": FONT_FAMILY, "size": 12, "color": REFERENCE_GREY},
        }
    )
    return figure


def evidence_bar_pair(metrics: list[tuple[str, float, float]], xaxis: str) -> tuple[go.Bar, go.Bar]:
    labels = [label for label, _, _ in metrics]
    chat = [chat_pct for _, chat_pct, _ in metrics]
    rl = [rl_pct for _, _, rl_pct in metrics]
    text_font = {"family": FONT_FAMILY, "size": 13}
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
        layout=evidence_card_layout(
            title="Esme-214M-RL vs Esme-214M-Chat: verifier-scored evidence",
            subtitle=(
                "Countdown-Lite acceptance eval (30 tasks x 32 samples) and unseen"
                " 2-number transfer, same protocol, max_new_tokens=12"
            ),
            conclusion=(
                "Conclusion: GRPO lifts valid expressions from 5.83% to 99.38%"
                " in-distribution and preserves the gain on unseen 2-number tasks"
                " (10.00% -> 98.75%).<br>Exact solves rise too: 0.73% -> 16.35%"
                " in-distribution, 3.12% -> 38.75% on unseen 2-number tasks."
            ),
        )
    )
    chat_bars, rl_bars = evidence_bar_pair(ACCEPTANCE_METRICS, "x")
    figure.add_trace(chat_bars)
    figure.add_trace(rl_bars)
    chat_bars2, rl_bars2 = evidence_bar_pair(HELDOUT_WIDTH2_METRICS, "x2")
    figure.add_trace(chat_bars2)
    figure.add_trace(rl_bars2)

    panel_caption_font = {"family": FONT_FAMILY, "size": 15, "color": "#374151"}
    figure.update_layout(
        barmode="group",
        bargap=0.36,
        bargroupgap=0.1,
        xaxis=styled_axis(domain=[0.0, 0.46], showgrid=False),
        xaxis2=styled_axis(domain=[0.54, 1.0], showgrid=False),
        yaxis=styled_axis(
            title={"text": "% of samples"},
            title_standoff=16,
            range=[0, 112],
            tickvals=[0, 25, 50, 75, 100],
            tickfont={"family": FONT_FAMILY, "size": 13, "color": TICK_COLOR},
        ),
        annotations=list(figure.layout.annotations)
        + [
            {
                "text": "<b>A. Acceptance eval (in-distribution)</b>",
                "xref": "paper",
                "yref": "paper",
                "x": 0.22,
                "y": 1.03,
                "xanchor": "center",
                "yanchor": "bottom",
                "showarrow": False,
                "font": panel_caption_font,
            },
            {
                "text": "<b>B. Unseen 2-number tasks</b>",
                "xref": "paper",
                "yref": "paper",
                "x": 0.79,
                "y": 1.03,
                "xanchor": "center",
                "yanchor": "bottom",
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
                "y": -0.14,
                "xanchor": "center",
                "showarrow": False,
                "font": {"family": FONT_FAMILY, "size": 13, "color": SUBTITLE_COLOR},
            },
        ],
    )
    figure.update_xaxes(tickfont={"family": FONT_FAMILY, "size": 13, "color": TICK_COLOR})
    return figure


def _dpo_panel_annotation(
    figure: go.Figure,
    *,
    xref: str,
    yref: str,
    color: str,
    total_steps: int,
    start_x: int,
    start_v: float,
    acc_v: float,
    delta_pp: float,
    accepted_step: int,
    delta_y: float,
    caption_y: float,
    acc_x: int,
    acc_yshift: int,
    acc_anchor: str,
    bottom_x: float,
    approx: bool,
) -> None:
    """Symmetric per-panel labels: start value, value at accepted step, delta, accepted caption.

    Each panel is annotated the same way so the two read as parallel standalone graphs. The
    delta + caption sit in the clear bottom-left corner so neither crowds the dashed accepted
    line; the accepted-value label can be nudged (``acc_x``/``acc_yshift``/``acc_anchor``) off
    the line on the jagged train panel. ``approx`` tags the windowed train read with a tilde.
    """
    tilde = "~" if approx else ""
    figure.add_annotation(
        text=f"{start_v * 100:.1f}%",
        x=start_x,
        y=start_v,
        xref=xref,
        yref=yref,
        yshift=-16,
        showarrow=False,
        font={"family": FONT_FAMILY, "size": 12, "color": SUBTITLE_COLOR},
    )
    figure.add_annotation(
        text=f"<b>{tilde}{acc_v * 100:.1f}%</b>",
        x=acc_x,
        y=acc_v,
        xref=xref,
        yref=yref,
        xanchor=acc_anchor,
        yshift=acc_yshift,
        showarrow=False,
        font={"family": FONT_FAMILY, "size": 12.5, "color": color},
    )
    figure.add_annotation(
        text=f"<b>+{delta_pp:.1f} pp</b>  step {start_x}→{accepted_step}",
        x=bottom_x,
        y=delta_y,
        xref=xref,
        yref=yref,
        xanchor="left",
        showarrow=False,
        font={"family": FONT_FAMILY, "size": 12.5, "color": "#374151"},
    )
    figure.add_annotation(
        text=f"accepted · step {accepted_step}",
        x=bottom_x,
        y=caption_y,
        xref=xref,
        yref=yref,
        xanchor="left",
        showarrow=False,
        font={"family": FONT_FAMILY, "size": 11.5, "color": REFERENCE_GREY},
    )


def build_dpo_training_figure(telemetry: DpoTelemetry) -> go.Figure:
    """Two framed standalone panels in one card: train accuracy | held-out zoomed.

    Greedy single-axis plotting compressed the held-out gain into a flat sliver dwarfed by the
    train curve. Splitting the axes (train 0.4-1.05, held-out zoomed 0.60-0.70) makes the modest
    but real held-out improvement legible while the train panel owns the overfitting story that
    motivates checkpoint selection.
    """
    total_steps = telemetry.train_steps[-1]
    accepted = telemetry.accepted_step

    start_acc = telemetry.eval_preference_accuracy[0]
    start_step = telemetry.eval_steps[0]
    delta_eval_pp = (telemetry.accepted_eval_accuracy - start_acc) * 100

    # Windowed so the train label lands on the local level at the accepted step, not a spike.
    train_start = telemetry.preference_accuracy[0]
    window = [
        v
        for s, v in zip(telemetry.train_steps, telemetry.preference_accuracy, strict=True)
        if abs(s - accepted) <= 40
    ]
    train_at_accepted = statistics.mean(window)
    delta_train_pp = (train_at_accepted - train_start) * 100

    left_domain = (0.0, 0.45)
    right_domain = (0.55, 1.0)
    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=False,
        horizontal_spacing=0.10,
        subplot_titles=(
            "Train preference accuracy (overfits to ~0.95)",
            "Held-out preference accuracy, zoomed",
        ),
    )
    figure.add_trace(
        go.Scatter(
            x=telemetry.train_steps,
            y=telemetry.preference_accuracy,
            mode="lines",
            line={"color": BLUE, "width": 1.9},
            opacity=0.6,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=telemetry.eval_steps,
            y=telemetry.eval_preference_accuracy,
            mode="lines",
            line={"color": GREEN, "width": 2.8},
        ),
        row=1,
        col=2,
    )

    figure.update_layout(
        width=DPO_CARD_WIDTH,
        height=DPO_CARD_HEIGHT,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=False,
        font={"family": FONT_FAMILY, "size": 12, "color": TICK_COLOR},
        margin={"l": 72, "r": 72, "t": 132, "b": 96},
        # Equal domains + symmetric margins: both panels the same width, pair centred. Use two
        # panel borders so the top/bottom strokes do not connect through the center gap.
        xaxis=styled_axis(
            title={"text": "DPO step"}, range=[0, total_steps * 1.02], domain=list(left_domain)
        ),
        yaxis=styled_axis(
            title={"text": "preference accuracy"}, range=[0.4, 1.05], tickvals=[0.4, 0.6, 0.8, 1.0]
        ),
        xaxis2=styled_axis(
            title={"text": "DPO step"}, range=[0, total_steps * 1.02], domain=list(right_domain)
        ),
        yaxis2=styled_axis(
            title={"text": "held-out accuracy"},
            range=[0.60, 0.70],
            tickvals=[0.60, 0.62, 0.64, 0.66, 0.68, 0.70],
            tickformat=".2f",
        ),
        shapes=[
            {
                "type": "path",
                "path": rounded_panel_border_path(
                    *left_domain,
                    width=DPO_CARD_WIDTH,
                    height=DPO_CARD_HEIGHT,
                ),
                "xref": "paper",
                "yref": "paper",
                "line": {"color": BORDER_COLOR, "width": 1},
                "layer": "above",
            },
            {
                "type": "path",
                "path": rounded_panel_border_path(
                    *right_domain,
                    width=DPO_CARD_WIDTH,
                    height=DPO_CARD_HEIGHT,
                ),
                "xref": "paper",
                "yref": "paper",
                "line": {"color": BORDER_COLOR, "width": 1},
                "layer": "above",
            },
        ],
    )

    for xref in ("x", "x2"):
        figure.add_shape(
            type="line",
            xref=xref,
            yref="paper",
            x0=accepted,
            x1=accepted,
            y0=0,
            y1=1,
            line={"color": REFERENCE_GREY, "width": 1, "dash": "dash"},
        )

    _dpo_panel_annotation(
        figure,
        xref="x",
        yref="y",
        color=BLUE,
        total_steps=total_steps,
        start_x=telemetry.train_steps[0],
        start_v=train_start,
        acc_v=train_at_accepted,
        delta_pp=delta_train_pp,
        accepted_step=accepted,
        delta_y=0.505,
        caption_y=0.455,
        acc_x=accepted - 55,
        acc_yshift=24,
        acc_anchor="right",
        bottom_x=0.14 * total_steps,
        approx=True,
    )
    _dpo_panel_annotation(
        figure,
        xref="x2",
        yref="y2",
        color=GREEN,
        total_steps=total_steps,
        start_x=start_step,
        start_v=start_acc,
        acc_v=telemetry.accepted_eval_accuracy,
        delta_pp=delta_eval_pp,
        accepted_step=accepted,
        delta_y=0.6155,
        caption_y=0.6075,
        acc_x=accepted,
        acc_yshift=16,
        acc_anchor="center",
        bottom_x=0.03 * total_steps,
        approx=False,
    )

    # Card title / subtitle / conclusion (card_layout is single-plot; place them by hand here).
    figure.add_annotation(
        text="<b>Esme-214M-Chat: DPO preference training</b>",
        xref="paper",
        yref="paper",
        x=-0.055,
        y=1.30,
        xanchor="left",
        showarrow=False,
        font={"family": FONT_FAMILY, "size": 22, "color": TITLE_COLOR},
    )
    figure.add_annotation(
        text=(
            f"Train and held-out preference accuracy over {total_steps} steps ·"
            " UltraFeedback-binarized pairs, beta 0.5"
        ),
        xref="paper",
        yref="paper",
        x=-0.055,
        y=1.19,
        xanchor="left",
        showarrow=False,
        font={"family": FONT_FAMILY, "size": 14, "color": SUBTITLE_COLOR},
    )
    figure.add_annotation(
        text=(
            "Conclusion: train preference accuracy overfits toward 0.95, so the accepted"
            f" checkpoint is chosen at the held-out peak<br>(step {accepted},"
            f" {telemetry.accepted_eval_accuracy * 100:.1f}%) — early stopping against DPO"
            f" overfitting. Held-out still improves, modestly: +{delta_eval_pp:.1f} pp from the"
            " first eval."
        ),
        xref="paper",
        yref="paper",
        x=-0.055,
        y=-0.155,
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        font={"family": FONT_FAMILY, "size": 12, "color": SUBTITLE_COLOR},
    )
    for note in figure.layout.annotations[:2]:  # the two subplot titles
        note.font = {"family": FONT_FAMILY, "size": 13.5, "color": "#374151"}
    return figure


def build_sft_training_figure(telemetry: SftTelemetry) -> go.Figure:
    total_steps = telemetry.train_steps[-1]
    figure = go.Figure(
        layout=card_layout(
            title="Esme-214M-Instruct: multi-turn SFT",
            subtitle=(
                f"Train loss and held-out response loss over {total_steps} steps -"
                " smol-smoltalk + tulu-3-personas, from Esme-214M-Base"
            ),
            conclusion=(
                f"Conclusion: held-out response loss bottoms at"
                f" {telemetry.accepted_response_loss:.2f} on step {telemetry.accepted_step}"
                " and rises afterwards;<br>early stopping restores that checkpoint, not the"
                " final step."
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=telemetry.train_steps,
            y=telemetry.train_loss,
            mode="lines",
            name="train loss",
            line={"color": BLUE, "width": 1.6},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=telemetry.eval_steps,
            y=telemetry.eval_response_loss,
            mode="lines+markers",
            name="held-out response loss",
            line={"color": GREEN, "width": 2.4},
            marker={"size": 5},
        )
    )
    figure.update_layout(
        xaxis=styled_axis(title={"text": "SFT step"}, range=[0, total_steps * 1.02]),
        yaxis=styled_axis(
            title={"text": "loss"},
            range=[0.4, 2.3],
            tickvals=[0.5, 1.0, 1.5, 2.0],
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
                f" (response loss {telemetry.accepted_response_loss:.2f})"
            ),
            "x": telemetry.accepted_step,
            "yref": "paper",
            "y": 0.97,
            "showarrow": False,
            "xanchor": "right",
            "xshift": -6,
            "font": {"family": FONT_FAMILY, "size": 12, "color": REFERENCE_GREY},
        }
    )
    figure.add_annotation(trace_label("train loss", total_steps * 0.3, 2.05, BLUE))
    figure.add_annotation(trace_label("held-out response loss", total_steps * 0.35, 0.62, GREEN))
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
    parser.add_argument(
        "--sft-run-dir",
        type=Path,
        default=Path("runs/esme-214m-sft-multiturn-full"),
        help="Multi-turn SFT run directory with metrics.jsonl and best-checkpoint.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    try:
        grpo = load_grpo_telemetry(args.grpo_run_dir)
        dpo = load_dpo_telemetry(args.dpo_run_dir)
        sft = load_sft_telemetry(args.sft_run_dir)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "sft_training_dynamics": args.output_dir / "fig-sft-training-dynamics.svg",
        "dpo_training_dynamics": args.output_dir / "fig-dpo-training-dynamics.svg",
        "grpo_training_dynamics": args.output_dir / "fig-grpo-training-dynamics.svg",
        "grpo_countdown_evidence": args.output_dir / "fig-grpo-countdown-evidence.svg",
    }
    build_sft_training_figure(sft).write_image(
        outputs["sft_training_dynamics"], format="svg", width=CARD_WIDTH, height=CARD_HEIGHT
    )
    build_dpo_training_figure(dpo).write_image(
        outputs["dpo_training_dynamics"],
        format="svg",
        width=DPO_CARD_WIDTH,
        height=DPO_CARD_HEIGHT,
    )
    build_grpo_dynamics_figure(grpo).write_image(
        outputs["grpo_training_dynamics"], format="svg", width=CARD_WIDTH, height=CARD_HEIGHT
    )
    build_countdown_evidence_figure().write_image(
        outputs["grpo_countdown_evidence"],
        format="svg",
        width=EVIDENCE_CARD_WIDTH,
        height=EVIDENCE_CARD_HEIGHT,
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
        "sft_run_dir": str(args.sft_run_dir),
        "sft_train_rows": sft.train_rows,
        "sft_eval_rows": sft.eval_rows,
        "sft_accepted_step": sft.accepted_step,
        "sft_accepted_response_loss": sft.accepted_response_loss,
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
