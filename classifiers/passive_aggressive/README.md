# PassiveAggressiveClassifier

`PassiveAggressiveClassifier` is an online linear margin classifier.

It still learns a linear score over the TF-IDF vector:

```text
z = w . x + b
```

So it is not a radically different prediction form from `LinearSVC` or `SGDClassifier(loss="hinge")`. The main difference is the training/update rule.

## Algorithm

For labels represented as `y ∈ {-1, +1}`, the margin is:

```text
margin = y * z
```

The hinge-style loss is:

```text
loss = max(0, 1 - margin)
```

The algorithm is:

- passive if `loss = 0`
- aggressive if `loss > 0`

It does not only update on wrong predictions. It also updates on correct predictions that are too close to the decision boundary:

```text
wrong prediction:        margin <= 0       -> update
correct but too close:   0 < margin < 1    -> update
correct with margin:     margin >= 1       -> no update
```

The basic update has the form:

```text
w_new = w_old + tau * y * x
```

where `tau` controls update size. In the PA-I variant, roughly:

```text
tau = min(C, loss / ||x||^2)
```

So `C` controls aggressiveness:

- larger `C`: allows larger updates
- smaller `C`: caps updates more strongly

This is best understood as a different online training algorithm for a linear margin classifier, not a completely different representation of the text problem.

## Probability And Log-Loss

Plain `PassiveAggressiveClassifier` does not expose `predict_proba`.

This baseline therefore:

- uses `decision_function` scores for ROC-AUC and PR-AUC
- does not calculate log-loss
- records log-loss as unavailable/null in `metrics.json`

## TF-IDF Setup

This run uses the same TF-IDF setup as the earlier baselines:

```python
TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    token_pattern=r"(?u)\b\w\w+\b",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=50_000,
    sublinear_tf=True,
)
```

Run from the project root:

```bash
python -m classifiers.passive_aggressive.run
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
python -m classifiers.passive_aggressive.run
```

Runtime on Luka's machine:

```text
real 19.17s
user 19.96s
sys  2.08s
```

Optimizer status:

- `C`: `1.0`
- configured `max_iter`: `1000`
- actual `n_iter_`: `16`
- convergence warning: none

Held-out test split:

- train rows: `40,000`
- test rows: `10,000`
- test negatives: `5,000`
- test positives: `5,000`

Metrics:

- accuracy: `0.9062`
- balanced accuracy: `0.9062`
- precision: `0.9064`
- recall: `0.9060`
- F1: `0.9062`
- ROC-AUC: `0.9656`
- PR-AUC: `0.9638`
- MCC: `0.8124`
- log-loss: unavailable for `PassiveAggressiveClassifier`

Confusion matrix:

| actual \\ predicted | negative | positive |
| --- | ---: | ---: |
| negative | 4,532 | 468 |
| positive | 470 | 4,530 |

This is a solid linear baseline, but below `LogisticRegression`, `SGDClassifier(loss="hinge")`, and `LinearSVC` in the current setup:

- `PassiveAggressiveClassifier`: accuracy `0.9062`, F1 `0.9062`
- `LogisticRegression`: accuracy `0.9095`, F1 `0.9101`
- `SGDClassifier(loss="hinge")`: accuracy `0.9111`, F1 `0.9117`
- `LinearSVC`: accuracy `0.9150`, F1 `0.9152`
