import torch

from relifuse import ReliFuse, ReliFuseConfig, load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path) -> None:
    model = ReliFuse(2, [0.75, 0.25], ReliFuseConfig(hidden_channels=8))
    predictions = torch.rand(1, 2, 10, 10)
    expected = model.fuse(predictions)
    path = save_checkpoint(
        tmp_path / "relifuse.pt",
        model,
        expert_names=["sharp", "sensitive"],
        metadata={"purpose": "test"},
    )
    loaded, manifest = load_checkpoint(path)
    actual = loaded.fuse(predictions)
    assert torch.allclose(actual, expected)
    assert manifest["expert_names"] == ["sharp", "sensitive"]
    assert manifest["metadata"] == {"purpose": "test"}
