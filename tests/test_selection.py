import torch

from relifuse import SelectionConfig, select_experts


def test_two_stage_selection_respects_family_cap() -> None:
    predictions = torch.zeros(4, 4, 8, 8)
    predictions[:, 0, 2:6, 2:6] = 0.9
    predictions[:, 1, 2:6, 2:7] = 0.9
    predictions[:, 2, 1:6, 2:6] = 0.9
    predictions[:, 3, 4:7, 4:7] = 0.9
    fold_dice = torch.tensor([[0.92, 0.91], [0.90, 0.90], [0.89, 0.88], [0.87, 0.86]])
    fold_recall = torch.full_like(fold_dice, 0.9)
    result = select_experts(
        predictions,
        fold_dice,
        fold_recall,
        families=["cnn", "cnn", "transformer", "mamba"],
        names=["a", "b", "c", "d"],
        config=SelectionConfig(max_experts=3, family_cap=1, recall_floor=0.8),
    )
    selected_families = [["cnn", "cnn", "transformer", "mamba"][i] for i in result.selected_indices]
    assert result.selected_names[0] == "a"
    assert len(selected_families) == len(set(selected_families))
    assert result.disagreement.shape == (4, 4)
