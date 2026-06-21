import torch

from relifuse import ReliFuse, ReliFuseConfig, TrainingConfig, fit


def test_fit_smoke_restores_a_best_epoch() -> None:
    torch.manual_seed(4)
    targets = (torch.rand(8, 1, 12, 12) > 0.75).float()
    predictions = torch.cat(
        [
            (targets * 0.8 + 0.1).clamp(0, 1),
            (targets * 0.6 + 0.2 + 0.05 * torch.rand_like(targets)).clamp(0, 1),
        ],
        dim=1,
    )
    model = ReliFuse(2, config=ReliFuseConfig(hidden_channels=8))
    history = fit(
        model,
        predictions[:6],
        targets[:6],
        predictions[6:],
        targets[6:],
        config=TrainingConfig(epochs=2, batch_size=2, patience=2, amp=False),
        device="cpu",
    )
    assert len(history.records) == 2
    assert history.best_epoch in {1, 2}
    assert history.best_validation_loss == min(record.validation_loss for record in history.records)
    assert 0 <= history.best_validation_dice <= 1
