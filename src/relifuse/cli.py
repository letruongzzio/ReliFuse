"""Command-line interface for array-based training and fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .checkpoint import load_checkpoint, save_checkpoint
from .config import TrainingConfig
from .metrics import expert_dice_scores
from .model import ReliFuse
from .training import fit, seed_everything


def _load_array(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def _train(args: argparse.Namespace) -> None:
    train_predictions = _load_array(args.train_predictions)
    train_targets = _load_array(args.train_targets)
    validation_predictions = _load_array(args.validation_predictions)
    validation_targets = _load_array(args.validation_targets)
    scores = expert_dice_scores(
        validation_predictions,
        validation_targets,
        batch_size=args.prior_batch_size,
    )
    seed_everything(args.seed)
    model = ReliFuse(train_predictions.shape[1], expert_scores=scores)
    training = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
    )
    history = fit(
        model,
        train_predictions,
        train_targets,
        validation_predictions,
        validation_targets,
        config=training,
        device=args.device,
    )
    save_checkpoint(
        args.output,
        model,
        expert_names=args.expert_name,
        metadata={"training_history": history.to_dict()},
    )
    print(json.dumps(history.to_dict(), indent=2))


def _fuse(args: argparse.Namespace) -> None:
    model, manifest = load_checkpoint(args.checkpoint, map_location=args.device)
    if len(args.mask) != model.num_experts:
        raise ValueError(
            f"Checkpoint expects {model.num_experts} expert masks in this order: "
            f"{manifest.get('expert_names')}"
        )
    predictions = [_load_array(path) for path in args.mask]
    fused = model.fuse(predictions, threshold=args.threshold).squeeze().cpu().numpy()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, fused)
    print(destination)


def _inspect(args: argparse.Namespace) -> None:
    model, manifest = load_checkpoint(args.checkpoint, map_location="cpu")
    payload = {
        "num_experts": model.num_experts,
        "expert_priors": model.expert_priors.tolist(),
        **manifest,
    }
    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relifuse", description="ReliFuse posterior fusion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train from cached .npy arrays")
    train_parser.add_argument("--train-predictions", required=True, help="[N,K,H,W] .npy")
    train_parser.add_argument("--train-targets", required=True, help="[N,H,W] .npy")
    train_parser.add_argument("--validation-predictions", required=True, help="[N,K,H,W] .npy")
    train_parser.add_argument("--validation-targets", required=True, help="[N,H,W] .npy")
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--expert-name", action="append")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--patience", type=int, default=10)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument(
        "--prior-batch-size",
        type=int,
        default=8,
        help="validation batch size used for paper-style expert Dice priors",
    )
    train_parser.add_argument("--device", default=None)
    train_parser.set_defaults(handler=_train)

    fuse_parser = subparsers.add_parser("fuse", help="fuse one .npy mask per expert")
    fuse_parser.add_argument("--checkpoint", required=True)
    fuse_parser.add_argument("--mask", action="append", required=True)
    fuse_parser.add_argument("--output", required=True)
    fuse_parser.add_argument("--threshold", type=float, default=None)
    fuse_parser.add_argument("--device", default="cpu")
    fuse_parser.set_defaults(handler=_fuse)

    inspect_parser = subparsers.add_parser("inspect", help="show checkpoint metadata")
    inspect_parser.add_argument("--checkpoint", required=True)
    inspect_parser.set_defaults(handler=_inspect)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
