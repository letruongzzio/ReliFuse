"""Input conversion and validation for posterior masks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

ArrayLike = Tensor | np.ndarray
PredictionInput = ArrayLike | Sequence[ArrayLike]


def _as_float_tensor(value: ArrayLike) -> Tensor:
    if isinstance(value, Tensor):
        return value.float()
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def _single_expert_batch(value: ArrayLike) -> Tensor:
    tensor = _as_float_tensor(value)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 3:
        return tensor.unsqueeze(1)
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        return tensor
    raise ValueError(
        "Each expert prediction must have shape [H,W], [B,H,W], or [B,1,H,W]; "
        f"received {tuple(tensor.shape)}"
    )


def stack_predictions(predictions: PredictionInput, num_experts: int | None = None) -> Tensor:
    """Return posterior predictions in canonical ``[B,K,H,W]`` form.

    A sequence is interpreted as one item per expert. A single tensor may be
    ``[K,H,W]`` (one sample) or ``[B,K,H,W]`` (a batch). Probabilities are
    required; values outside ``[0,1]`` are rejected instead of silently treating
    logits as posteriors.
    """

    if isinstance(predictions, Sequence) and not isinstance(predictions, (Tensor, np.ndarray)):
        if not predictions:
            raise ValueError("At least one expert prediction is required")
        batches = [_single_expert_batch(item) for item in predictions]
        reference = batches[0].shape
        if any(batch.shape != reference for batch in batches[1:]):
            shapes = [tuple(batch.shape) for batch in batches]
            raise ValueError(f"All expert predictions must share a shape; received {shapes}")
        tensor = torch.cat(batches, dim=1)
    else:
        tensor = _as_float_tensor(predictions)  # type: ignore[arg-type]
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            if num_experts == 1 and tensor.shape[0] != 1:
                tensor = tensor.unsqueeze(1)
            else:
                tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 4:
            raise ValueError(
                "Predictions must have shape [H,W], [K,H,W], [B,K,H,W], or be a list "
                f"of expert masks; received {tuple(tensor.shape)}"
            )

    if num_experts is not None and tensor.shape[1] != num_experts:
        raise ValueError(f"Expected {num_experts} expert channels, received {tensor.shape[1]}")
    if not torch.isfinite(tensor).all():
        raise ValueError("Predictions contain NaN or infinite values")
    if tensor.numel() and (tensor.min() < 0 or tensor.max() > 1):
        raise ValueError("ReliFuse expects posterior probabilities in [0, 1], not logits")
    return tensor


def as_target_tensor(targets: ArrayLike) -> Tensor:
    """Return binary/soft targets in canonical ``[B,1,H,W]`` form."""

    tensor = _as_float_tensor(targets)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim != 4 or tensor.shape[1] != 1:
        raise ValueError(
            f"Targets must have shape [H,W], [B,H,W], or [B,1,H,W]; received {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError("Targets contain NaN or infinite values")
    if tensor.numel() and (tensor.min() < 0 or tensor.max() > 1):
        raise ValueError("Targets must be in [0, 1]")
    return tensor
