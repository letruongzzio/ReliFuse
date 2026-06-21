"""Optional two-stage diversity-aware expert selection from the paper appendix."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .inputs import PredictionInput, stack_predictions
from .metrics import expert_dice_scores, expert_recall_scores


@dataclass(frozen=True)
class SelectionConfig:
    max_experts: int = 7
    recall_floor: float = 0.80
    family_cap: int = 3
    quality_weight: float = 1.0
    diversity_weight: float = 0.5
    minimum_gain: float = 0.01
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.max_experts < 1 or self.family_cap < 1:
            raise ValueError("max_experts and family_cap must be positive")
        if not 0 <= self.recall_floor <= 1 or not 0 <= self.threshold <= 1:
            raise ValueError("recall_floor and threshold must be in [0, 1]")


@dataclass(frozen=True)
class SelectionResult:
    selected_indices: tuple[int, ...]
    selected_names: tuple[str, ...]
    admissible_indices: tuple[int, ...]
    quality: Tensor
    minimum_recall: Tensor
    disagreement: Tensor


def pairwise_disagreement(
    predictions: PredictionInput,
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> Tensor:
    posteriors = stack_predictions(predictions)
    hard = (posteriors >= threshold).float()
    experts = hard.shape[1]
    output = torch.zeros((experts, experts), dtype=torch.float32)
    for first in range(experts):
        for second in range(first + 1, experts):
            a = hard[:, first]
            b = hard[:, second]
            intersection = (a * b).sum()
            dice = (2 * intersection + epsilon) / (a.sum() + b.sum() + epsilon)
            output[first, second] = output[second, first] = 1.0 - dice
    return output


def select_experts(
    predictions: PredictionInput,
    fold_dice: Tensor | np.ndarray,
    fold_recall: Tensor | np.ndarray,
    families: Sequence[str],
    names: Sequence[str] | None = None,
    config: SelectionConfig | None = None,
) -> SelectionResult:
    """Run Stage A screening followed by Stage B greedy selection.

    ``fold_dice`` and ``fold_recall`` must be shaped ``[K,F]``. Supplying a
    single validation split is allowed with ``F=1``; paper reproduction should
    pass the full cross-validation fold table.
    """

    config = config or SelectionConfig()
    posteriors = stack_predictions(predictions)
    experts = posteriors.shape[1]
    dice_table = torch.as_tensor(fold_dice, dtype=torch.float32)
    recall_table = torch.as_tensor(fold_recall, dtype=torch.float32)
    if dice_table.ndim == 1:
        dice_table = dice_table.unsqueeze(1)
    if recall_table.ndim == 1:
        recall_table = recall_table.unsqueeze(1)
    if dice_table.shape != recall_table.shape or dice_table.shape[0] != experts:
        raise ValueError("fold_dice and fold_recall must both have shape [K,F]")
    if len(families) != experts:
        raise ValueError("families length must match the expert count")
    resolved_names = tuple(names or [f"expert_{index}" for index in range(experts)])
    if len(resolved_names) != experts:
        raise ValueError("names length must match the expert count")

    quality = dice_table.mean(dim=1)
    minimum_recall = recall_table.amin(dim=1)
    family_thresholds: dict[str, float] = {}
    for family in sorted(set(families)):
        indices = [index for index, value in enumerate(families) if value == family]
        values = quality[indices]
        family_thresholds[family] = float(values.mean() - values.std(unbiased=False))
    admissible = [
        index
        for index in range(experts)
        if quality[index] >= family_thresholds[families[index]]
        and minimum_recall[index] >= config.recall_floor
    ]
    admissible.sort(
        key=lambda index: (
            -float(quality[index]),
            -float(minimum_recall[index]),
            resolved_names[index],
        )
    )
    if not admissible:
        raise ValueError("No admissible experts after quality and recall screening")

    disagreement = pairwise_disagreement(posteriors, config.threshold)
    selected = [admissible[0]]
    remaining = set(admissible[1:])
    candidate_rank = {index: rank for rank, index in enumerate(admissible)}
    while len(selected) < config.max_experts and remaining:
        eligible = [
            index
            for index in remaining
            if sum(families[chosen] == families[index] for chosen in selected) < config.family_cap
        ]
        if not eligible:
            break

        def candidate_key(index: int) -> tuple[float, float, float, int]:
            diversity = float(disagreement[index, selected].sum())
            gain = (
                config.quality_weight * float(quality[index]) + config.diversity_weight * diversity
            )
            return gain, float(quality[index]), diversity, -candidate_rank[index]

        best = max(eligible, key=candidate_key)
        if candidate_key(best)[0] < config.minimum_gain:
            break
        selected.append(best)
        remaining.remove(best)
    return SelectionResult(
        selected_indices=tuple(selected),
        selected_names=tuple(resolved_names[index] for index in selected),
        admissible_indices=tuple(admissible),
        quality=quality,
        minimum_recall=minimum_recall,
        disagreement=disagreement,
    )


def select_from_validation(
    predictions: PredictionInput,
    targets: Tensor | np.ndarray,
    families: Sequence[str],
    names: Sequence[str] | None = None,
    config: SelectionConfig | None = None,
) -> SelectionResult:
    """Convenience wrapper when only one held-out validation split is available."""

    config = config or SelectionConfig()
    dice = expert_dice_scores(predictions, targets, config.threshold)
    recall = expert_recall_scores(predictions, targets, config.threshold)
    return select_experts(predictions, dice, recall, families, names, config)
