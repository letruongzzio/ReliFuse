"""Configuration objects for the paper-aligned ReliFuse implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReliFuseConfig:
    """Architecture and loss constants from the reported ReliFuse configuration.

    The defaults follow Equations (18), (20), (22), (23), and (30) in the
    manuscript. Changing them creates a new experimental variant.
    """

    hidden_channels: int = 24
    bias_scale: float = 1.25
    ambiguity_range_weight: float = 0.35
    ambiguity_boundary_weight: float = 0.25
    ambiguity_refinement_scale: float = 0.20
    max_logit_correction: float = 2.0
    bce_weight: float = 0.50
    dice_weight: float = 0.50
    boundary_weight: float = 0.080
    consensus_weight: float = 0.080
    sparse_weight: float = 2e-4
    calibration_weight: float = 0.020
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if self.max_logit_correction <= 0:
            raise ValueError("max_logit_correction must be positive")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        bounded = {
            "ambiguity_refinement_scale": self.ambiguity_refinement_scale,
            "bce_weight": self.bce_weight,
            "dice_weight": self.dice_weight,
        }
        for name, value in bounded.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        non_negative = {
            "bias_scale": self.bias_scale,
            "ambiguity_range_weight": self.ambiguity_range_weight,
            "ambiguity_boundary_weight": self.ambiguity_boundary_weight,
            "boundary_weight": self.boundary_weight,
            "consensus_weight": self.consensus_weight,
            "sparse_weight": self.sparse_weight,
            "calibration_weight": self.calibration_weight,
        }
        for name, value in non_negative.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class TrainingConfig:
    """Settings for training only the ReliFuse posterior-fusion head."""

    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    min_delta: float = 1e-5
    gradient_clip_norm: float = 5.0
    num_workers: int = 0
    seed: int = 42
    amp: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
