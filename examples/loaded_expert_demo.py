"""Small runnable helpers for notebooks that load real PyTorch expert weights."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ExpertCheckpoint:
    name: str
    family: str
    path: Path


class TinyVesselExpert(nn.Module):
    """Tiny segmentation expert used only by notebooks."""

    def __init__(self, width: int = 8, dilation: int = 1) -> None:
        super().__init__()
        self.width = width
        self.dilation = dilation
        self.net = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation),
            nn.SiLU(),
            nn.Conv2d(width, 1, 1),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.net(images)


def make_synthetic_vessel_split(
    samples: int,
    *,
    seed: int,
    image_size: int = 48,
) -> tuple[Tensor, Tensor]:
    """Return ``(images, masks)`` with shapes ``[B,1,H,W]``."""

    generator = torch.Generator().manual_seed(seed)
    axis = torch.linspace(-1, 1, image_size)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    images: list[Tensor] = []
    masks: list[Tensor] = []
    for _ in range(samples):
        slope = torch.empty(()).uniform_(-0.55, 0.55, generator=generator)
        offset = torch.empty(()).uniform_(-0.28, 0.28, generator=generator)
        width = torch.empty(()).uniform_(0.055, 0.105, generator=generator)
        vessel = torch.exp(-(((x - slope * y - offset) / width) ** 2))
        branch = torch.exp(-(((x + 0.65 * y + offset * 0.55) / (width * 1.2)) ** 2))
        branch *= torch.exp(-(((y - 0.08) / 0.45) ** 2))
        mask = torch.maximum(vessel, branch * 0.88).gt(0.44).float()
        texture = 0.18 * torch.randn((image_size, image_size), generator=generator)
        image = (0.35 + 0.42 * mask + texture).clamp(0, 1)
        images.append(image.unsqueeze(0))
        masks.append(mask.unsqueeze(0))
    return torch.stack(images), torch.stack(masks)


def train_demo_expert(
    model: TinyVesselExpert,
    images: Tensor,
    masks: Tensor,
    *,
    epochs: int = 50,
    learning_rate: float = 5e-3,
) -> TinyVesselExpert:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    positive = masks.sum().clamp_min(1.0)
    negative = (1.0 - masks).sum().clamp_min(1.0)
    pos_weight = (negative / positive).clamp(max=12.0)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        probabilities = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, masks, pos_weight=pos_weight)
        intersection = (probabilities * masks).sum()
        dice = (2.0 * intersection + 1.0) / (probabilities.sum() + masks.sum() + 1.0)
        loss = bce + (1.0 - dice)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def create_demo_checkpoints(
    directory: str | Path,
    images: Tensor,
    masks: Tensor,
    specs: Sequence[tuple[str, str, int, int]],
) -> list[ExpertCheckpoint]:
    """Train tiny experts and save state dicts under ``directory``.

    ``specs`` entries are ``(name, family, width, dilation)``.
    """

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    checkpoints: list[ExpertCheckpoint] = []
    for name, family, width, dilation in specs:
        model = train_demo_expert(TinyVesselExpert(width, dilation), images, masks)
        path = root / f"{name}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_kwargs": {"width": width, "dilation": dilation},
            },
            path,
        )
        checkpoints.append(ExpertCheckpoint(name, family, path))
    return checkpoints


def load_expert_models(
    checkpoints: Sequence[ExpertCheckpoint],
    *,
    device: str | torch.device = "cpu",
) -> list[TinyVesselExpert]:
    """Load expert weights from disk and return eval-mode models."""

    models: list[TinyVesselExpert] = []
    resolved_device = torch.device(device)
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint.path, map_location=resolved_device, weights_only=True)
        model = TinyVesselExpert(**payload["model_kwargs"])
        model.load_state_dict(payload["state_dict"])
        model.to(resolved_device).eval()
        models.append(model)
    return models


@torch.inference_mode()
def predict_stack(
    models: Sequence[nn.Module],
    images: Tensor,
    *,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Run loaded experts and return ``[B,K,H,W]`` posterior stack."""

    resolved_device = torch.device(device)
    batch = images.to(resolved_device)
    posteriors = [torch.sigmoid(model(batch)).cpu() for model in models]
    return torch.cat(posteriors, dim=1)


def plot_fusion_case(
    image: Tensor,
    target: Tensor,
    expert_probabilities: Tensor,
    relifuse_probability: Tensor,
    *,
    expert_names: Sequence[str] | None = None,
    threshold: float = 0.5,
) -> plt.Figure:
    """Visualize input, ground truth, each expert, and ReliFuse output."""

    names = list(
        expert_names or [f"expert_{index}" for index in range(expert_probabilities.shape[0])]
    )
    if len(names) != expert_probabilities.shape[0]:
        raise ValueError("expert_names length must match expert_probabilities")
    target_mask = (target.squeeze().detach().cpu() >= 0.5).float()

    def dice_label(mask: Tensor) -> float:
        mask = mask.squeeze().detach().cpu().float()
        intersection = (mask * target_mask).sum()
        return float((2.0 * intersection + 1.0) / (mask.sum() + target_mask.sum() + 1.0))

    panels: list[tuple[str, Tensor, str]] = [
        ("Input image", image.squeeze().detach().cpu(), "gray"),
        ("Ground truth", target.squeeze().detach().cpu(), "gray"),
    ]
    panels.extend(
        (f"{name} prob", expert_probabilities[index].detach().cpu(), "viridis")
        for index, name in enumerate(names)
    )
    panels.extend(
        (
            f"{name} mask Dice {dice_label(expert_probabilities[index] >= threshold):.2f}",
            (expert_probabilities[index] >= threshold).float().detach().cpu(),
            "gray",
        )
        for index, name in enumerate(names)
    )
    relifuse_mask = (relifuse_probability.squeeze() >= threshold).float().detach().cpu()
    panels.extend(
        [
            ("ReliFuse prob", relifuse_probability.squeeze().detach().cpu(), "viridis"),
            (f"ReliFuse mask Dice {dice_label(relifuse_mask):.2f}", relifuse_mask, "gray"),
        ]
    )

    columns = min(4, len(panels))
    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.2 * rows), squeeze=False)
    axes = [axis for row in axes for axis in row]
    for axis, (title, values, cmap) in zip(axes, panels, strict=False):
        axis.imshow(values.numpy(), cmap=cmap, vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    for axis in axes[len(panels) :]:
        axis.axis("off")
    fig.tight_layout()
    return fig
