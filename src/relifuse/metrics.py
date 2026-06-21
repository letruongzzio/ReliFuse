"""Small metric helpers used to derive validation quality priors."""

from __future__ import annotations

import torch
from torch import Tensor

from .inputs import ArrayLike, PredictionInput, as_target_tensor, stack_predictions


def dice_score(
    probabilities: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> Tensor:
    predictions = (probabilities >= threshold).float()
    targets = (targets >= 0.5).float()
    intersection = (predictions * targets).sum(dim=(-2, -1))
    denominator = predictions.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
    return ((2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


def legacy_batch_dice_score(
    probabilities: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> Tensor:
    """Compute one Dice value per channel after pooling pixels over a batch.

    This is the historical batch-level protocol used by the paper. ReliFuse
    outputs one channel, while an expert stack returns one score per expert.
    """

    if probabilities.ndim != 4 or targets.ndim != 4 or targets.shape[1] != 1:
        raise ValueError("Expected probabilities [B,C,H,W] and targets [B,1,H,W]")
    if probabilities.shape[0] != targets.shape[0] or probabilities.shape[-2:] != targets.shape[-2:]:
        raise ValueError("Prediction and target sample/spatial shapes must match")
    predictions = (probabilities >= threshold).float()
    binary_targets = (targets >= 0.5).float()
    dimensions = (0, 2, 3)
    intersection = (predictions * binary_targets).sum(dim=dimensions)
    denominator = predictions.sum(dim=dimensions) + binary_targets.sum(dim=dimensions)
    return (2.0 * intersection + epsilon) / (denominator + epsilon)


def expert_dice_scores(
    predictions: PredictionInput,
    targets: ArrayLike,
    threshold: float = 0.5,
    epsilon: float = 1e-6,
    batch_size: int | None = None,
) -> Tensor:
    """Compute one validation quality score per expert.

    With ``batch_size=None``, the whole validation set is one micro-averaged
    unit. Set ``batch_size`` to reproduce the paper protocol: compute Dice per
    sequential batch, then average the batch scores with equal batch weight.
    """

    posteriors = stack_predictions(predictions)
    target_tensor = as_target_tensor(targets)
    if (
        posteriors.shape[0] != target_tensor.shape[0]
        or posteriors.shape[-2:] != target_tensor.shape[-2:]
    ):
        raise ValueError("Prediction and target sample/spatial shapes must match")
    if batch_size is None:
        return legacy_batch_dice_score(posteriors, target_tensor, threshold, epsilon)
    if batch_size < 1:
        raise ValueError("batch_size must be positive when provided")
    batch_scores = [
        legacy_batch_dice_score(
            posteriors[start : start + batch_size],
            target_tensor[start : start + batch_size],
            threshold,
            epsilon,
        )
        for start in range(0, posteriors.shape[0], batch_size)
    ]
    if not batch_scores:
        raise ValueError("At least one validation sample is required")
    return torch.stack(batch_scores).mean(dim=0)


def expert_recall_scores(
    predictions: PredictionInput,
    targets: ArrayLike,
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> Tensor:
    posteriors = stack_predictions(predictions)
    target_tensor = as_target_tensor(targets)
    if (
        posteriors.shape[0] != target_tensor.shape[0]
        or posteriors.shape[-2:] != target_tensor.shape[-2:]
    ):
        raise ValueError("Prediction and target sample/spatial shapes must match")
    hard_predictions = (posteriors >= threshold).float()
    hard_targets = (target_tensor >= 0.5).float()
    true_positive = (hard_predictions * hard_targets).sum(dim=(0, 2, 3))
    positive = hard_targets.sum(dim=(0, 2, 3))
    return (true_positive + epsilon) / (positive + epsilon)
