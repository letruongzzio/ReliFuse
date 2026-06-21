"""Interpretable diagnostic state from Equations (8)--(15)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

DIAGNOSTIC_NAMES = (
    "mean",
    "validation_weighted_mean",
    "variance",
    "range",
    "top2_mean",
    "minimum",
    "maximum",
    "entropy",
    "boundary",
)


def clamp_probabilities(probabilities: Tensor, epsilon: float = 1e-6) -> Tensor:
    return torch.nan_to_num(probabilities.float(), nan=0.5, posinf=1.0, neginf=0.0).clamp(
        epsilon, 1.0 - epsilon
    )


def probability_logit(probabilities: Tensor, epsilon: float = 1e-6) -> Tensor:
    probabilities = clamp_probabilities(probabilities, epsilon)
    return torch.log(probabilities) - torch.log1p(-probabilities)


def normalize_spatial_map(values: Tensor, epsilon: float = 1e-6) -> Tensor:
    if values.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], received {tuple(values.shape)}")
    minimum = values.amin(dim=(-2, -1), keepdim=True)
    maximum = values.amax(dim=(-2, -1), keepdim=True)
    return ((values - minimum) / (maximum - minimum + epsilon)).clamp(0, 1)


def gradient_magnitude(values: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Channel-wise Sobel gradient magnitude."""

    if values.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], received {tuple(values.shape)}")
    kernel_x = values.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
    kernel_y = values.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3) / 8.0
    channels = values.shape[1]
    grad_x = F.conv2d(values, kernel_x.repeat(channels, 1, 1, 1), padding=1, groups=channels)
    grad_y = F.conv2d(values, kernel_y.repeat(channels, 1, 1, 1), padding=1, groups=channels)
    return torch.sqrt(grad_x.square() + grad_y.square() + epsilon)


def build_diagnostic_state(
    probabilities: Tensor,
    expert_priors: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    """Construct the nine-channel paper diagnostic state ``S(x)``."""

    probabilities = clamp_probabilities(probabilities, epsilon)
    if probabilities.ndim != 4:
        raise ValueError(f"Expected [B,K,H,W], received {tuple(probabilities.shape)}")
    if expert_priors.numel() != probabilities.shape[1]:
        raise ValueError("expert_priors length does not match the posterior stack")

    priors = expert_priors.reshape(1, -1, 1, 1).to(
        device=probabilities.device, dtype=probabilities.dtype
    )
    priors = priors / priors.sum().clamp_min(epsilon)
    mean = probabilities.mean(dim=1, keepdim=True)
    weighted_mean = (probabilities * priors).sum(dim=1, keepdim=True)
    variance = probabilities.var(dim=1, keepdim=True, unbiased=False)
    maximum = probabilities.amax(dim=1, keepdim=True)
    minimum = probabilities.amin(dim=1, keepdim=True)
    probability_range = maximum - minimum
    top_count = min(2, probabilities.shape[1])
    top2_mean = probabilities.topk(top_count, dim=1).values.mean(dim=1, keepdim=True)
    entropy = -(
        probabilities * torch.log(probabilities)
        + (1.0 - probabilities) * torch.log1p(-probabilities)
    ).mean(dim=1, keepdim=True) / math.log(2.0)
    boundary = normalize_spatial_map(gradient_magnitude(weighted_mean, epsilon), epsilon)
    return torch.cat(
        [
            mean,
            weighted_mean,
            variance,
            probability_range,
            top2_mean,
            minimum,
            maximum,
            entropy,
            boundary,
        ],
        dim=1,
    )
