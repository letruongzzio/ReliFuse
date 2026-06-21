import torch

from relifuse import DIAGNOSTIC_NAMES, ReliFuse, ReliFuseConfig, relifuse_loss


def test_paper_aligned_forward_shapes_and_initialization() -> None:
    torch.manual_seed(1)
    model = ReliFuse(
        num_experts=2,
        expert_scores=[0.8, 0.2],
        config=ReliFuseConfig(hidden_channels=8),
    )
    predictions = torch.rand(3, 2, 16, 12)
    output = model(predictions)
    assert len(DIAGNOSTIC_NAMES) == 9
    assert output.probabilities.shape == (3, 1, 16, 12)
    assert output.diagnostic_state.shape == (3, 9, 16, 12)
    assert output.reliability.shape == predictions.shape
    assert torch.isfinite(output.logits).all()
    assert torch.allclose(output.correction, torch.zeros_like(output.correction))
    assert torch.allclose(output.calibration_bias, torch.zeros_like(output.calibration_bias))
    assert torch.allclose(output.probabilities, output.prior_probabilities)


def test_structure_aware_loss_backpropagates() -> None:
    model = ReliFuse(2, config=ReliFuseConfig(hidden_channels=8))
    predictions = torch.rand(2, 2, 12, 12)
    targets = (torch.rand(2, 1, 12, 12) > 0.75).float()
    losses = relifuse_loss(model(predictions), targets, model.config)
    assert set(losses) == {
        "total",
        "segmentation",
        "bce",
        "dice",
        "boundary",
        "consensus",
        "sparse",
        "calibration",
    }
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_fuse_can_threshold_a_list() -> None:
    model = ReliFuse(2, config=ReliFuseConfig(hidden_channels=8))
    result = model.fuse([torch.full((10, 10), 0.7), torch.full((10, 10), 0.8)], 0.5)
    assert result.shape == (1, 1, 10, 10)
    assert torch.all(result == 1)
