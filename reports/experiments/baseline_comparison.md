# Existing Baseline Comparison

| Rank | Model | Group | Accuracy | F1 | ROC-AUC | MCC | Log-loss | Best Val | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | LinearSVC | baseline | 0.9150 | 0.9152 | 0.9720 | 0.8300 |  |  |  |
| 2 | SGDClassifier | baseline | 0.9111 | 0.9117 | 0.9714 | 0.8223 |  |  |  |
| 3 | Hard voting | ensemble | 0.9107 | 0.9112 |  | 0.8215 |  |  |  |
| 4 | Stacking prefit | ensemble | 0.9099 | 0.9101 | 0.9686 | 0.8198 | 0.4079 |  |  |
| 5 | LogisticRegression | baseline | 0.9095 | 0.9101 | 0.9710 | 0.8191 | 0.2707 |  |  |
| 6 | Soft voting | ensemble | 0.9095 | 0.9102 | 0.9707 | 0.8191 | 0.2654 |  |  |
| 7 | PassiveAggressive | baseline | 0.9062 | 0.9062 | 0.9656 | 0.8124 |  |  |  |
| 8 | Tiny MLX d24 l4 s512 | scratch_transformer | 0.8943 | 0.8963 | 0.9603 | 0.7892 | 0.2789 | 0.8935 | 263,306 params; 1026.4s train |
| 9 | LightGBM | baseline | 0.8918 | 0.8925 | 0.9605 | 0.7837 | 0.2617 |  |  |
| 10 | MultinomialNB tuned | baseline | 0.8854 | 0.8856 | 0.9554 | 0.7708 | 0.2934 |  |  |
| 11 | MultinomialNB | baseline | 0.8841 | 0.8846 | 0.9531 | 0.7682 | 0.3135 |  |  |
| 12 | ComplementNB | baseline | 0.8841 | 0.8846 | 0.9531 | 0.7682 | 0.3135 |  |  |
| 13 | Tiny Torch CUDA d24 l4 s1024 | scratch_transformer | 0.8840 | 0.8821 | 0.9529 | 0.7684 | 0.2920 | 0.8742 | 263,306 params; 220.1s train |
| 14 | ExtraTrees | baseline | 0.8749 | 0.8737 | 0.9466 | 0.7499 | 0.4342 |  |  |
| 15 | Tiny Torch CUDA d24 l4 s512 | scratch_transformer | 0.8733 | 0.8714 | 0.9482 | 0.7469 | 0.3012 | 0.8730 | 263,306 params; 119.3s train |
| 16 | RandomForest | baseline | 0.8633 | 0.8625 | 0.9376 | 0.7266 | 0.4554 |  |  |
| 17 | XGBoost | baseline | 0.8509 | 0.8553 | 0.9355 | 0.7031 | 0.3565 |  |  |
