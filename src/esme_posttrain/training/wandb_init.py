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


def start_wandb(
    config: WandbConfig,
    *,
    run_config: dict[str, Any],
    base_bundle_manifest: dict[str, Any] | None = None,
) -> Any | None:
    """Start a W&B run from explicit fields; each stage supplies its own run_config.

    The logged config is ``run_config`` plus the base bundle manifest, with
    ``config.extra_config`` layered on top. No trainer-config attributes are
    read here, so no stage needs an adapter shim.
    """
    if not config.enabled:
        return None
    import wandb

    settings: dict[str, Any] = {}
    if config.mode:
        settings["mode"] = config.mode
    return wandb.init(
        project=config.project,
        name=config.run_name,
        tags=list(config.tags),
        group=config.group,
        job_type=config.job_type,
        notes=config.notes,
        config={
            **run_config,
            "base_bundle": base_bundle_manifest,
            **config.extra_config,
        },
        **settings,
    )
