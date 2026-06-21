# Reproducibility protocol

ReliFuse is a posterior-level method. Reproducing it means reproducing both the provenance of the expert posteriors and the fusion-head split discipline; it does not require using the original pulmonary base architectures unless the goal is to reproduce the paper's exact numbers.

## Required tensors

- Posterior stack: `[N,K,H,W]`, float probabilities in `[0,1]`.
- Ground truth: `[N,H,W]` or `[N,1,H,W]`.
- Stable expert order: channel `i` must refer to the same expert in every split and checkpoint.
- Validation quality: one Dice score per expert, estimated without test labels.

Do not pass logits, RGB images, or test-derived statistics into the fusion head.

## Recommended split discipline

1. Reserve the test set before choosing experts or fusion variants.
2. Generate development posteriors out-of-fold or from a split on which the corresponding base model was not optimized.
3. Use only development/validation data for expert quality priors, diversity selection, hyperparameters, early stopping, and threshold selection.
4. Train only the ReliFuse head; keep base experts frozen.
5. Select the checkpoint with the lowest fusion-validation loss; validation Dice is diagnostic only.
6. Save expert names and order in the ReliFuse checkpoint.
7. Evaluate the frozen system on the test set once.

If base predictions are generated in-sample, the fusion head can learn an unrealistically clean posterior distribution. The library cannot detect this leakage from arrays alone, so provenance remains the caller's responsibility.

## Paper-aligned configuration

`ReliFuseConfig()` uses the selected configuration described by Equations (18), (20), (22), (23), and (30):

| Component | Default |
|---|---:|
| Diagnostic channels | 9 |
| Hidden channels | 24 |
| Calibration bias scale | 1.25 |
| Ambiguity scaffold | `Norm(Var + 0.35 Range + 0.25 Boundary)` |
| Learned ambiguity refinement | 0.20 |
| Maximum residual logit correction | 2.0 |
| Segmentation objective | `0.5 BCE + 0.5 Dice` |
| Boundary loss | 0.080 |
| Consensus preservation | 0.080 |
| Sparse correction | `2e-4` |
| Calibration loss | 0.020 |

The historical notebooks explored a tenth gap/topology cue and directional topology branch. They are not present in the public default because the final manuscript reports the lightweight boundary-aware, no-topology ReliFuse with the nine-channel state.

## Diversity-aware selection

`select_experts(...)` implements the two appendix stages:

- Stage A: retain experts whose mean fold Dice is at least their family mean minus one family standard deviation and whose minimum fold recall clears the recall floor.
- Stage B: anchor on the best admissible expert, then greedily add quality-plus-disagreement candidates while enforcing a family cap.

For an exact selection study, pass `[K,F]` Dice and recall tables from all cross-validation folds. `select_from_validation(...)` is a practical one-split convenience API, not a substitute for the paper's cross-validation protocol.

## Metrics

`expert_dice_scores(..., batch_size=None)` computes a full-tensor micro score. To reproduce the paper priors, pass `batch_size=8`: TP/FP/FN are accumulated inside each sequential validation batch, Dice is computed per batch, and those batch rows are averaged equally. The trainer uses the same batch-level aggregation for its loss and Dice history. Use an identical batch protocol for every compared method when reproducing numerical tables.
