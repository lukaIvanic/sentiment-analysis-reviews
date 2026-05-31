# Cached Prefit Stacking

This ensemble implements the same idea as scikit-learn's `StackingClassifier` with prefit base models:

```python
StackingClassifier(cv="prefit", stack_method="auto")
```

The runner is implemented manually so the base-model stack features can be cached. That avoids repeating the expensive part:

```text
raw reviews -> saved base pipelines -> stack feature matrix
```

The saved baseline pipelines are loaded from:

```text
outputs/baselines/*/model.joblib
```

They are not retrained.

## What Gets Trained

Only the final meta-classifier is trained:

```text
saved base models -> base predictions/scores -> LogisticRegression meta-classifier
```

The saved base pipelines already contain their fitted TF-IDF vocabularies, IDF weights, and trained classifiers.

The stack features are cached here:

```text
outputs/ensemble/stacking_classifier_prefit/meta_features_cache/
```

After the meta-classifier is fitted, it is saved to:

```text
outputs/ensemble/stacking_classifier_prefit/model.joblib
```

Later runs load both the cached stack features and the saved meta-classifier unless `--force-rebuild-cache` or `--force-refit-stack` is passed.

## Auto Stack Method

With `stack_method="auto"`, the runner uses sklearn's usual stacking preference order:

- `predict_proba` when available
- otherwise `decision_function` when available
- otherwise `predict`

This lets the stacker use probability models, margin models such as `LinearSVC`, and hard-prediction-only models in one ensemble.

In the baseline run, the selected methods were:

- `predict_proba`: `MultinomialNB`, `ComplementNB`, `LogisticRegression`, `RandomForestClassifier`, `ExtraTreesClassifier`, `XGBoostClassifier`, `LightGBMClassifier`, tuned `MultinomialNB`
- `decision_function`: `LinearSVC`, `SGDClassifier`, `PassiveAggressiveClassifier`

## Caveat

This is intentionally fast and practical, but it is not the statistically cleanest stacking setup. Because the base models were trained on the same training split used to fit the meta-classifier, the meta-classifier sees in-sample base predictions.

A stricter version would use `cv=3` or `cv=5` without `prefit`, so the meta-classifier is trained on out-of-fold predictions. That would retrain base models many times and is much slower.

## Run

Run from the project root:

```bash
python -m classifiers.stacking_classifier_prefit.run
```

Expected outputs:

- `run_config.json`
- `metrics.json`
- `confusion_matrix.json`
- `classification_report.txt`
- `model.joblib`

## Baseline Result

Plain baseline run:

```bash
python -m classifiers.stacking_classifier_prefit.run
```

First full run, when the stack-feature cache had to be built:

```text
real 110.35s
user 116.46s
sys  3.61s
```

Immediate rerun using cached stack features and cached meta-classifier:

```text
real 1.98s
user 2.29s
sys  1.08s
```

Cache files:

- `x_train_meta.joblib`: `3.4M`
- `x_test_meta.joblib`: `860K`
- `model.joblib`: `4K`

Ensemble status:

- members: `11`
- meta-features: `11`
- base models retrained: `False`
- final estimator: `LogisticRegression`
- stack method requested: `auto`

Held-out test split:

- train rows: `40,000`
- test rows: `10,000`
- test negatives: `5,000`
- test positives: `5,000`

Metrics:

- accuracy: `0.9099`
- balanced accuracy: `0.9099`
- precision: `0.9077`
- recall: `0.9126`
- F1: `0.9101`
- ROC-AUC: `0.9686`
- PR-AUC: `0.9669`
- MCC: `0.8198`
- log-loss: `0.4079`

Confusion matrix:

| actual \\ predicted | negative | positive |
| --- | ---: | ---: |
| negative | 4,536 | 464 |
| positive | 437 | 4,563 |

This prefit stacking run is close to `LogisticRegression` and the soft/hard voting ensembles on hard-label metrics, but it does not beat `LinearSVC`. The log-loss is also worse than the soft-voting ensemble and `LightGBMClassifier`, which suggests the prefit/in-sample stacking setup is not producing well-calibrated probabilities here.
