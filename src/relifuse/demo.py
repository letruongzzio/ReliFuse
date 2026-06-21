"""Self-contained synthetic data and toy experts for the public notebook."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import TrainingConfig
from .metrics import expert_dice_scores
from .model import ReliFuse
from .training import TrainingHistory, fit, seed_everything


class ContrastExpert(nn.Module):
    """Toy high-precision segmenter based on local image contrast."""

    def forward(self, images: Tensor) -> Tensor:
        local_background = F.avg_pool2d(images, kernel_size=9, stride=1, padding=4)
        logits = 18.0 * (images - local_background - 0.10)
        return torch.sigmoid(logits)


class SmoothIntensityExpert(nn.Module):
    """Toy high-recall segmenter that favors smooth bright structures."""

    def forward(self, images: Tensor) -> Tensor:
        smoothed = F.avg_pool2d(images, kernel_size=5, stride=1, padding=2)
        logits = 11.0 * (smoothed - 0.40)
        return torch.sigmoid(logits)


@dataclass(frozen=True)
class DemoResult:
    images: Tensor
    targets: Tensor
    expert_predictions: Tensor
    fused_predictions: Tensor
    ambiguity: Tensor
    validation_scores: Tensor
    history: TrainingHistory
    model: ReliFuse


def make_synthetic_vessels(
    count: int = 36,
    size: int = 48,
    seed: int = 7,
) -> tuple[Tensor, Tensor]:
    """Create small noisy images with curved and branching vessel-like masks."""

    generator = torch.Generator().manual_seed(seed)
    axis = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    images: list[Tensor] = []
    masks: list[Tensor] = []
    for _ in range(count):
        offset = float(torch.empty(1).uniform_(-0.25, 0.25, generator=generator))
        slope = float(torch.empty(1).uniform_(-0.35, 0.35, generator=generator))
        phase = float(torch.empty(1).uniform_(0, 6.28, generator=generator))
        width = float(torch.empty(1).uniform_(0.045, 0.085, generator=generator))
        centerline = offset + slope * xx + 0.14 * torch.sin(2.8 * xx + phase)
        vessel = (torch.abs(yy - centerline) < width).float()
        if bool(torch.rand(1, generator=generator) > 0.35):
            branch_center = centerline + 0.42 * (xx + 0.2)
            branch = (xx > -0.2) & (torch.abs(yy - branch_center) < width * 0.65)
            vessel = torch.maximum(vessel, branch.float())
        illumination = 0.20 + 0.08 * xx - 0.04 * yy
        texture = 0.07 * torch.randn((size, size), generator=generator)
        faintness = float(torch.empty(1).uniform_(0.42, 0.70, generator=generator))
        image = (illumination + faintness * vessel + texture).clamp(0, 1)
        images.append(image.unsqueeze(0))
        masks.append(vessel.unsqueeze(0))
    return torch.stack(images), torch.stack(masks)


@torch.inference_mode()
def predict_with_toy_experts(images: Tensor) -> Tensor:
    experts = (ContrastExpert().eval(), SmoothIntensityExpert().eval())
    return torch.cat([expert(images) for expert in experts], dim=1)


def run_demo(epochs: int = 6, device: str | None = None) -> DemoResult:
    images, targets = make_synthetic_vessels()
    predictions = predict_with_toy_experts(images)
    train_slice = slice(0, 24)
    validation_slice = slice(24, 32)
    test_slice = slice(32, 36)

    validation_scores = expert_dice_scores(
        predictions[validation_slice], targets[validation_slice], batch_size=8
    )
    seed_everything(42)
    model = ReliFuse(num_experts=2, expert_scores=validation_scores)
    history = fit(
        model,
        predictions[train_slice],
        targets[train_slice],
        predictions[validation_slice],
        targets[validation_slice],
        config=TrainingConfig(
            epochs=epochs,
            batch_size=4,
            patience=max(2, epochs),
            seed=42,
            amp=False,
        ),
        device=device,
    )
    model.eval()
    with torch.inference_mode():
        output = model(predictions[test_slice].to(model.expert_priors.device))
    return DemoResult(
        images=images[test_slice],
        targets=targets[test_slice],
        expert_predictions=predictions[test_slice],
        fused_predictions=output.probabilities.cpu(),
        ambiguity=output.ambiguity.cpu(),
        validation_scores=validation_scores,
        history=history,
        model=model,
    )
