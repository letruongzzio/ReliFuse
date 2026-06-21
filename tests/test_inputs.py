import numpy as np
import pytest
import torch

from relifuse import stack_predictions


def test_stack_predictions_accepts_list_of_masks() -> None:
    first = np.full((8, 9), 0.2, dtype=np.float32)
    second = torch.full((8, 9), 0.8)
    stacked = stack_predictions([first, second], num_experts=2)
    assert stacked.shape == (1, 2, 8, 9)
    assert torch.allclose(stacked[:, 0], torch.tensor(first).unsqueeze(0))


def test_stack_predictions_rejects_logits_and_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        stack_predictions(torch.tensor([[[-2.0, 2.0]]]))
    with pytest.raises(ValueError, match="share a shape"):
        stack_predictions([torch.zeros(4, 4), torch.zeros(5, 4)])
