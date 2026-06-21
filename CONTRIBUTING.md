# Contributing

Contributions are welcome when they preserve the distinction between image-to-mask experts and posterior fusion.

1. Create a focused branch.
2. Add tests for behavioral changes.
3. Run `make test`, `make lint`, and `make format-check`.
4. State whether a change is paper-aligned or an experimental variant.

Please do not commit private datasets, `results_v*`, cached predictions, or model checkpoints. New fusion variants should use explicit configuration names and should not silently change `ReliFuseConfig()` defaults.
