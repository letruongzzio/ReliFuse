"""Structure-aware ReliFuse objective from Equations (25)--(30)."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from .config import ReliFuseConfig
from .features import clamp_probabilities, gradient_magnitude
from .inputs import ArrayLike, as_target_tensor
from .model import ReliFuseOutput


def soft_dice_loss(probabilities: Tensor, targets: Tensor, epsilon: float = 1e-6) -> Tensor:
    intersection = (probabilities * targets).sum(dim=(-2, -1))
    denominator = probabilities.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
    return 1.0 - ((2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


def relifuse_loss(
    output: ReliFuseOutput,
    targets: ArrayLike,
    config: ReliFuseConfig | None = None,
) -> dict[str, Tensor]:
    """Compute named loss components and their weighted total."""

    config = config or ReliFuseConfig()
    target_tensor = as_target_tensor(targets).to(output.logits.device)
    if target_tensor.shape != output.probabilities.shape:
        raise ValueError(
            f"Target shape {tuple(target_tensor.shape)} does not match output "
            f"{tuple(output.probabilities.shape)}"
        )
    target_tensor = target_tensor.float().clamp(0, 1)
    probabilities = clamp_probabilities(output.probabilities, config.epsilon)

    bce = F.binary_cross_entropy_with_logits(output.logits.float(), target_tensor)
    dice = soft_dice_loss(probabilities, target_tensor, config.epsilon)
    segmentation = config.bce_weight * bce + config.dice_weight * dice
    boundary = F.l1_loss(
        gradient_magnitude(probabilities, config.epsilon),
        gradient_magnitude(target_tensor, config.epsilon),
    )
    consensus = (
        (1.0 - output.ambiguity.detach())
        * (probabilities - output.prior_probabilities.detach()).abs()
    ).mean()
    sparse = (output.ambiguity * output.correction).abs().mean()
    calibration = 0.5 * F.mse_loss(probabilities, target_tensor) + 0.5 * F.mse_loss(
        output.prior_probabilities.float(), target_tensor
    )
    total = (
        segmentation
        + config.boundary_weight * boundary
        + config.consensus_weight * consensus
        + config.sparse_weight * sparse
        + config.calibration_weight * calibration
    )
    return {
        "total": total,
        "segmentation": segmentation,
        "bce": bce,
        "dice": dice,
        "boundary": boundary,
        "consensus": consensus,
        "sparse": sparse,
        "calibration": calibration,
    }
