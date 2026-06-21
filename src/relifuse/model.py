"""ReliFuse: reliability-calibrated posterior fusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .config import ReliFuseConfig
from .features import (
    build_diagnostic_state,
    clamp_probabilities,
    normalize_spatial_map,
    probability_logit,
)
from .inputs import PredictionInput, stack_predictions


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormActivation(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1
    ):
        padding = (kernel_size // 2) * dilation
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.first = ConvNormActivation(channels, channels, dilation=dilation)
        self.second = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs + self.second(self.first(inputs)))


@dataclass(frozen=True)
class ReliFuseOutput:
    """Final posterior plus interpretable intermediate fields."""

    logits: Tensor
    probabilities: Tensor
    prior_probabilities: Tensor
    ambiguity: Tensor
    correction: Tensor
    reliability: Tensor
    calibration_bias: Tensor
    diagnostic_state: Tensor


class ReliFuse(nn.Module):
    """Fuse a stack of frozen segmentation posteriors.

    Parameters
    ----------
    num_experts:
        Number of posterior maps and their fixed channel order.
    expert_scores:
        Validation Dice scores ``q_i``. They are normalized into the quality
        priors from Equation (7). Uniform priors are used when omitted.
    config:
        Paper-aligned architecture and loss configuration.
    """

    diagnostic_channels = 9

    def __init__(
        self,
        num_experts: int,
        expert_scores: Tensor | list[float] | tuple[float, ...] | None = None,
        config: ReliFuseConfig | None = None,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be at least 1")
        self.num_experts = int(num_experts)
        self.config = config or ReliFuseConfig()
        scores = (
            torch.ones(self.num_experts, dtype=torch.float32)
            if expert_scores is None
            else torch.as_tensor(expert_scores, dtype=torch.float32).flatten()
        )
        if scores.numel() != self.num_experts:
            raise ValueError(
                f"Expected {self.num_experts} expert scores, received {scores.numel()}"
            )
        if not torch.isfinite(scores).all() or torch.any(scores < 0):
            raise ValueError("expert_scores must be finite and non-negative")
        if scores.sum() <= 0:
            scores = torch.ones_like(scores)
        self.register_buffer("expert_priors", scores / scores.sum())

        hidden = self.config.hidden_channels
        feature_channels = self.num_experts + self.diagnostic_channels
        self.reliability_branch = nn.Sequential(
            ConvNormActivation(feature_channels, hidden),
            ResidualBlock(hidden, dilation=1),
            ResidualBlock(hidden, dilation=2),
            nn.Conv2d(hidden, self.num_experts * 2, kernel_size=1),
        )
        ambiguity_hidden = max(8, hidden // 2)
        self.ambiguity_branch = nn.Sequential(
            ConvNormActivation(self.diagnostic_channels + 1, ambiguity_hidden),
            ResidualBlock(ambiguity_hidden),
            nn.Conv2d(ambiguity_hidden, 1, kernel_size=1),
        )
        self.residual_branch = nn.Sequential(
            ConvNormActivation(self.diagnostic_channels + 2, hidden),
            ResidualBlock(hidden, dilation=1),
            ResidualBlock(hidden, dilation=2),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        for branch in (self.reliability_branch, self.ambiguity_branch, self.residual_branch):
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def forward(self, predictions: PredictionInput) -> ReliFuseOutput:
        posteriors = stack_predictions(predictions, self.num_experts)
        device = self.expert_priors.device
        posteriors = posteriors.to(device=device)
        epsilon = self.config.epsilon
        probabilities = clamp_probabilities(posteriors, epsilon)
        diagnostics = build_diagnostic_state(probabilities, self.expert_priors, epsilon)

        raw_reliability_bias = self.reliability_branch(
            torch.cat([probabilities, diagnostics], dim=1)
        )
        reliability_logits, raw_bias = torch.split(raw_reliability_bias, self.num_experts, dim=1)
        local_reliability = torch.sigmoid(reliability_logits)
        prior_anchor = self.expert_priors.reshape(1, -1, 1, 1).to(
            device=probabilities.device, dtype=probabilities.dtype
        )
        reliability = (local_reliability + prior_anchor).clamp_min(epsilon)
        calibration_bias = self.config.bias_scale * torch.tanh(raw_bias)

        expert_logits = probability_logit(probabilities, epsilon)
        prior_logits = (reliability * (expert_logits - calibration_bias)).sum(
            dim=1, keepdim=True
        ) / reliability.sum(dim=1, keepdim=True).clamp_min(epsilon)
        prior_probabilities = torch.sigmoid(prior_logits)

        variance = diagnostics[:, 2:3]
        probability_range = diagnostics[:, 3:4]
        boundary = diagnostics[:, 8:9]
        ambiguity_scaffold = normalize_spatial_map(
            variance
            + self.config.ambiguity_range_weight * probability_range
            + self.config.ambiguity_boundary_weight * boundary,
            epsilon,
        )
        ambiguity_refinement = torch.tanh(
            self.ambiguity_branch(torch.cat([diagnostics, prior_probabilities], dim=1))
        )
        ambiguity = (
            ambiguity_scaffold + self.config.ambiguity_refinement_scale * ambiguity_refinement
        ).clamp(0, 1)

        raw_correction = self.residual_branch(
            torch.cat([diagnostics, prior_probabilities, ambiguity], dim=1)
        )
        correction = self.config.max_logit_correction * torch.tanh(raw_correction)
        output_logits = prior_logits + ambiguity * correction
        return ReliFuseOutput(
            logits=output_logits,
            probabilities=torch.sigmoid(output_logits),
            prior_probabilities=prior_probabilities,
            ambiguity=ambiguity,
            correction=correction,
            reliability=reliability,
            calibration_bias=calibration_bias,
            diagnostic_state=diagnostics,
        )

    def fuse(self, predictions: PredictionInput, threshold: float | None = None) -> Tensor:
        """Run inference and return a probability map or thresholded mask."""

        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                probabilities = self(predictions).probabilities
        finally:
            self.train(was_training)
        if threshold is None:
            return probabilities
        return (probabilities >= threshold).to(probabilities.dtype)
