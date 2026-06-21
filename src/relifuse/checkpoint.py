"""Portable ReliFuse checkpoint I/O."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import ReliFuseConfig
from .model import ReliFuse

CHECKPOINT_FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: ReliFuse,
    *,
    expert_names: list[str] | tuple[str, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save architecture, fixed expert order, priors, and learned parameters."""

    if expert_names is not None and len(expert_names) != model.num_experts:
        raise ValueError("expert_names length must match model.num_experts")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "relifuse",
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "num_experts": model.num_experts,
        "expert_priors": model.expert_priors.detach().cpu(),
        "expert_names": list(expert_names) if expert_names is not None else None,
        "config": asdict(model.config),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": metadata or {},
    }
    torch.save(payload, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[ReliFuse, dict[str, Any]]:
    """Load a checkpoint and return ``(model, manifest)``."""

    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch 2.1 compatibility
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or payload.get("format") != "relifuse":
        raise ValueError("Not a ReliFuse checkpoint")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format version {payload.get('format_version')}")
    config = ReliFuseConfig(**payload["config"])
    model = ReliFuse(
        num_experts=int(payload["num_experts"]),
        expert_scores=payload["expert_priors"],
        config=config,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(map_location)
    manifest = {
        "format_version": payload["format_version"],
        "expert_names": payload.get("expert_names"),
        "metadata": payload.get("metadata", {}),
        "config": payload["config"],
    }
    return model, manifest
