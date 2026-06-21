"""Reusable training loop for cached posterior stacks."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainingConfig
from .inputs import ArrayLike, PredictionInput, as_target_tensor, stack_predictions
from .losses import relifuse_loss
from .metrics import legacy_batch_dice_score
from .model import ReliFuse


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    train_dice: float
    validation_loss: float
    validation_dice: float
    learning_rate: float


@dataclass(frozen=True)
class TrainingHistory:
    records: tuple[EpochRecord, ...]
    best_epoch: int
    best_validation_loss: float
    best_validation_dice: float

    def to_dict(self) -> dict[str, object]:
        return {
            "records": [asdict(record) for record in self.records],
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "best_validation_dice": self.best_validation_dice,
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_loader(
    predictions: PredictionInput,
    targets: ArrayLike,
    config: TrainingConfig,
    shuffle: bool,
) -> DataLoader:
    posteriors = stack_predictions(predictions)
    target_tensor = as_target_tensor(targets)
    if (
        posteriors.shape[0] != target_tensor.shape[0]
        or posteriors.shape[-2:] != target_tensor.shape[-2:]
    ):
        raise ValueError("Prediction and target sample/spatial shapes must match")
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        TensorDataset(posteriors, target_tensor),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        generator=generator,
    )


def _run_epoch(
    model: ReliFuse,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
    use_amp: bool,
    scaler: object | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    dice_sum = 0.0
    batch_count = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for predictions, targets in loader:
            predictions = predictions.to(device)
            targets = targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = model(predictions)
                losses = relifuse_loss(output, targets, model.config)
            if optimizer is not None:
                if scaler is not None:
                    scaler.scale(losses["total"]).backward()  # type: ignore[union-attr]
                    scaler.unscale_(optimizer)  # type: ignore[union-attr]
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    scaler.step(optimizer)  # type: ignore[union-attr]
                    scaler.update()  # type: ignore[union-attr]
                else:
                    losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    optimizer.step()
            loss_sum += float(losses["total"].detach())
            dice_sum += float(
                legacy_batch_dice_score(output.probabilities.detach(), targets).mean()
            )
            batch_count += 1
    if batch_count == 0:
        raise ValueError("DataLoader produced no samples")
    return loss_sum / batch_count, dice_sum / batch_count


def fit_loaders(
    model: ReliFuse,
    train_loader: DataLoader,
    validation_loader: DataLoader | None = None,
    config: TrainingConfig | None = None,
    device: str | torch.device | None = None,
) -> TrainingHistory:
    """Fit a fusion head from DataLoaders yielding ``(posteriors, target)``."""

    config = config or TrainingConfig()
    seed_everything(config.seed)
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.epochs),
        eta_min=config.learning_rate * 0.05,
    )
    use_amp = config.amp and resolved_device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_amp else None
    validation_loader = validation_loader or train_loader

    records: list[EpochRecord] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_loss = float("inf")
    best_dice = float("nan")
    early_stopping_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, config.epochs + 1):
        train_loss, train_dice = _run_epoch(
            model,
            train_loader,
            resolved_device,
            optimizer,
            config.gradient_clip_norm,
            use_amp,
            scaler,
        )
        validation_loss, validation_dice = _run_epoch(
            model,
            validation_loader,
            resolved_device,
            None,
            config.gradient_clip_norm,
            use_amp,
            None,
        )
        records.append(
            EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                train_dice=train_dice,
                validation_loss=validation_loss,
                validation_dice=validation_dice,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
        )
        # Appendix H: select checkpoints only by the lowest validation objective.
        # Validation Dice is recorded as a convergence diagnostic, not a selector.
        if validation_loss < best_loss:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_dice = validation_dice
            best_loss = validation_loss

        if validation_loss < early_stopping_loss - config.min_delta:
            early_stopping_loss = validation_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= config.patience:
            break
    model.load_state_dict(best_state)
    return TrainingHistory(tuple(records), best_epoch, best_loss, best_dice)


def fit(
    model: ReliFuse,
    train_predictions: PredictionInput,
    train_targets: ArrayLike,
    validation_predictions: PredictionInput | None = None,
    validation_targets: ArrayLike | None = None,
    config: TrainingConfig | None = None,
    device: str | torch.device | None = None,
) -> TrainingHistory:
    """Convenience in-memory API for fitting from arrays or tensors."""

    config = config or TrainingConfig()
    train_loader = _make_loader(train_predictions, train_targets, config, shuffle=True)
    if (validation_predictions is None) != (validation_targets is None):
        raise ValueError("Provide both validation_predictions and validation_targets, or neither")
    validation_loader = None
    if validation_predictions is not None and validation_targets is not None:
        validation_loader = _make_loader(
            validation_predictions, validation_targets, config, shuffle=False
        )
    return fit_loaders(model, train_loader, validation_loader, config, device)
