"""MT-Bench-style multi-turn chat judge score, reported with spread.

This is a *reporting-only* signal: it never feeds the best-checkpoint selector or
early stopping (those use matched response loss). It runs an LLM judge over fixed
multi-turn prompts K>=5 times and reports the mean score with its spread, so a
single noisy judge pass cannot be mistaken for a result.

The judge callable is injected so the aggregation is testable without spend; the
full run does not yet wire in a real chat-judge model (it passes judge=None). If
no judge is configured the score is recorded as unavailable rather than silently
faked.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# A judge takes (prompt, generation) and returns a 1-10 score for one pass.
JudgeFn = Callable[[str, str], float]


class JudgeError(ValueError):
    pass


@dataclass(frozen=True)
class FixedMultiTurnPrompt:
    name: str
    # Rendered conversation context the model continues (multi-turn).
    prompt: str


# Fixed multi-turn prompts used for the chat-quality report. Kept small and
# deterministic; they are not training or selector data.
FIXED_MULTI_TURN_PROMPTS: tuple[FixedMultiTurnPrompt, ...] = (
    FixedMultiTurnPrompt(
        name="followup_clarification",
        prompt=(
            "user\nName three primary colors.\nassistant\nRed, yellow, and blue.\n"
            "user\nNow combine the first two. What color do you get?\nassistant\n"
        ),
    ),
    FixedMultiTurnPrompt(
        name="context_carryover",
        prompt=(
            "user\nMy dog is named Pixel.\nassistant\nNice to meet Pixel!\n"
            "user\nWhat is my dog's name?\nassistant\n"
        ),
    ),
    FixedMultiTurnPrompt(
        name="instruction_then_revise",
        prompt=(
            "user\nWrite a one-sentence summary of photosynthesis.\nassistant\n"
            "Plants turn sunlight, water, and carbon dioxide into energy and oxygen.\n"
            "user\nMake it shorter, under ten words.\nassistant\n"
        ),
    ),
)


@dataclass(frozen=True)
class JudgePassResult:
    pass_index: int
    per_prompt_scores: dict[str, float]

    @property
    def mean_score(self) -> float:
        if not self.per_prompt_scores:
            raise JudgeError("a judge pass produced no scores")
        return statistics.fmean(self.per_prompt_scores.values())


@dataclass(frozen=True)
class MultiTurnJudgeReport:
    available: bool
    passes: int
    mean_score: float | None
    score_stdev: float | None
    score_min: float | None
    score_max: float | None
    per_pass_mean_scores: tuple[float, ...]
    prompt_names: tuple[str, ...]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "is_selector": False,
            "passes": self.passes,
            "mean_score": self.mean_score,
            "score_stdev": self.score_stdev,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "per_pass_mean_scores": list(self.per_pass_mean_scores),
            "prompt_names": list(self.prompt_names),
            "note": self.note,
        }


def run_multi_turn_judge(
    generate: Callable[[str], str],
    judge: JudgeFn | None,
    *,
    passes: int = 5,
    prompts: tuple[FixedMultiTurnPrompt, ...] = FIXED_MULTI_TURN_PROMPTS,
) -> MultiTurnJudgeReport:
    if passes < 5:
        raise JudgeError("multi-turn judge requires K>=5 re-judge passes")
    if not prompts:
        raise JudgeError("multi-turn judge requires at least one fixed prompt")
    prompt_names = tuple(prompt.name for prompt in prompts)
    if judge is None:
        return MultiTurnJudgeReport(
            available=False,
            passes=passes,
            mean_score=None,
            score_stdev=None,
            score_min=None,
            score_max=None,
            per_pass_mean_scores=(),
            prompt_names=prompt_names,
            note="no chat judge configured; multi-turn judge score not computed",
        )
    # Generate once per prompt; re-judge the same generations K times so the spread
    # reflects judge variance, not sampling variance.
    generations = {prompt.name: generate(prompt.prompt) for prompt in prompts}
    pass_results: list[JudgePassResult] = []
    for pass_index in range(passes):
        scores = {
            prompt.name: float(judge(prompt.prompt, generations[prompt.name])) for prompt in prompts
        }
        pass_results.append(JudgePassResult(pass_index=pass_index, per_prompt_scores=scores))
    per_pass_means = [result.mean_score for result in pass_results]
    return MultiTurnJudgeReport(
        available=True,
        passes=passes,
        mean_score=statistics.fmean(per_pass_means),
        score_stdev=statistics.pstdev(per_pass_means) if len(per_pass_means) > 1 else 0.0,
        score_min=min(per_pass_means),
        score_max=max(per_pass_means),
        per_pass_mean_scores=tuple(per_pass_means),
        prompt_names=prompt_names,
        note="reported only; never the best-checkpoint selector or early-stop metric",
    )
