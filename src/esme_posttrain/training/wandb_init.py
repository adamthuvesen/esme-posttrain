from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: str = "esme-posttrain"
    run_name: str | None = None
    mode: str | None = None
    tags: tuple[str, ...] = ()
    group: str | None = None
    job_type: str | None = None
    notes: str | None = None
    extra_config: dict[str, Any] = field(default_factory=dict)


def start_wandb(config: Any, base_bundle_manifest: dict[str, Any] | None) -> Any | None:
    if not config.wandb.enabled:
        return None
    import wandb

    settings: dict[str, Any] = {}
    if config.wandb.mode:
        settings["mode"] = config.wandb.mode
    return wandb.init(
        project=config.wandb.project,
        name=config.wandb.run_name,
        tags=list(config.wandb.tags),
        group=config.wandb.group,
        job_type=config.wandb.job_type,
        notes=config.wandb.notes,
        config={
            "artifact_name": config.artifact_name,
            "max_steps": config.max_steps,
            "micro_batch_size": config.micro_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": config.effective_batch_size,
            "learning_rate": config.learning_rate,
            "scheduler": config.scheduler,
            "warmup_steps": config.warmup_steps,
            "weight_decay": config.weight_decay,
            "precision": config.precision,
            "tuning_mode": config.tuning_mode,
            "assistant_only_loss": config.assistant_only_loss,
            "completion_only_loss": config.completion_only_loss,
            "seed": config.seed,
            "base_bundle": base_bundle_manifest,
            **config.wandb.extra_config,
        },
        **settings,
    )
