import math

import torch
import torch.nn.functional as F

from relifuse import (
    ReliFuse,
    ReliFuseConfig,
    build_diagnostic_state,
    expert_dice_scores,
    legacy_batch_dice_score,
    relifuse_loss,
)


def _normalize(values: torch.Tensor, epsilon: float) -> torch.Tensor:
    minimum = values.amin(dim=(-2, -1), keepdim=True)
    maximum = values.amax(dim=(-2, -1), keepdim=True)
    return (values - minimum) / (maximum - minimum + epsilon)


def _sobel_magnitude(values: torch.Tensor, epsilon: float) -> torch.Tensor:
    kernel_x = values.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8
    kernel_y = values.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3) / 8
    grad_x = F.conv2d(values, kernel_x, padding=1)
    grad_y = F.conv2d(values, kernel_y, padding=1)
    return torch.sqrt(grad_x.square() + grad_y.square() + epsilon)


def test_equations_8_to_15_diagnostic_state() -> None:
    epsilon = 1e-6
    probabilities = torch.tensor([[[[0.10, 0.80], [0.30, 0.60]], [[0.70, 0.40], [0.50, 0.20]]]])
    priors = torch.tensor([0.75, 0.25])
    mean = probabilities.mean(dim=1, keepdim=True)
    weighted = (probabilities * priors.view(1, 2, 1, 1)).sum(dim=1, keepdim=True)
    variance = ((probabilities - mean) ** 2).mean(dim=1, keepdim=True)
    maximum = probabilities.max(dim=1, keepdim=True).values
    minimum = probabilities.min(dim=1, keepdim=True).values
    entropy = -(
        probabilities * torch.log(probabilities)
        + (1 - probabilities) * torch.log(1 - probabilities)
    ).mean(dim=1, keepdim=True) / math.log(2)
    boundary = _normalize(_sobel_magnitude(weighted, epsilon), epsilon)
    expected = torch.cat(
        [
            mean,
            weighted,
            variance,
            maximum - minimum,
            probabilities.topk(2, dim=1).values.mean(dim=1, keepdim=True),
            minimum,
            maximum,
            entropy,
            boundary,
        ],
        dim=1,
    )
    actual = build_diagnostic_state(probabilities, priors, epsilon)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_equations_17_to_24_at_zero_initialized_branches() -> None:
    config = ReliFuseConfig(hidden_channels=8)
    model = ReliFuse(2, expert_scores=[0.75, 0.25], config=config)
    probabilities = torch.tensor([[[[0.10, 0.80], [0.30, 0.60]], [[0.70, 0.40], [0.50, 0.20]]]])
    output = model(probabilities)
    priors = torch.tensor([0.75, 0.25]).view(1, 2, 1, 1)
    reliability = 0.5 + priors
    expert_logits = torch.log(probabilities) - torch.log1p(-probabilities)
    expected_logits = (reliability * expert_logits).sum(dim=1, keepdim=True) / reliability.sum(
        dim=1, keepdim=True
    )
    scaffold = _normalize(
        output.diagnostic_state[:, 2:3]
        + 0.35 * output.diagnostic_state[:, 3:4]
        + 0.25 * output.diagnostic_state[:, 8:9],
        config.epsilon,
    )
    assert torch.allclose(output.reliability, reliability)
    assert torch.allclose(output.calibration_bias, torch.zeros_like(output.calibration_bias))
    assert torch.allclose(output.ambiguity, scaffold)
    assert torch.allclose(output.correction, torch.zeros_like(output.correction))
    assert torch.allclose(output.logits, expected_logits, atol=1e-6)
    assert torch.allclose(output.probabilities, torch.sigmoid(expected_logits), atol=1e-6)


def test_equation_30_total_is_the_reported_weighted_sum() -> None:
    config = ReliFuseConfig(hidden_channels=8)
    model = ReliFuse(2, config=config)
    predictions = torch.rand(2, 2, 10, 10)
    targets = (torch.rand(2, 1, 10, 10) > 0.7).float()
    losses = relifuse_loss(model(predictions), targets, config)
    expected = (
        losses["segmentation"]
        + 0.080 * losses["boundary"]
        + 0.080 * losses["consensus"]
        + 2e-4 * losses["sparse"]
        + 0.020 * losses["calibration"]
    )
    assert torch.allclose(losses["total"], expected)


def test_paper_validation_prior_uses_mean_of_batch_level_dice() -> None:
    targets = torch.zeros(3, 1, 4, 4)
    targets[0, 0, :2, :2] = 1
    targets[1:, 0, :, :] = 1
    predictions = torch.zeros(3, 2, 4, 4)
    predictions[0, 0, :2, :2] = 1
    predictions[1:, 0, :2, :] = 1
    predictions[:, 1] = targets[:, 0]

    expected = torch.stack(
        [
            legacy_batch_dice_score(predictions[:2], targets[:2]),
            legacy_batch_dice_score(predictions[2:], targets[2:]),
        ]
    ).mean(dim=0)
    actual = expert_dice_scores(predictions, targets, batch_size=2)
    micro = expert_dice_scores(predictions, targets)
    assert torch.allclose(actual, expected)
    assert not torch.allclose(actual, micro)
