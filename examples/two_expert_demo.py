"""Run the same two-expert workflow used by the example notebook."""

from relifuse.demo import run_demo

if __name__ == "__main__":
    result = run_demo()
    print("Validation expert Dice:", result.validation_scores.tolist())
    print("Selected epoch:", result.history.best_epoch)
    print("Selected validation loss:", f"{result.history.best_validation_loss:.6f}")
    print("Validation Dice at selected epoch:", f"{result.history.best_validation_dice:.4f}")
    print("Fused test tensor:", tuple(result.fused_predictions.shape))
