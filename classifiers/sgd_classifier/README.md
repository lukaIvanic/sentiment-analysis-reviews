# SGDClassifier

`SGDClassifier` is a scalable linear classifier trained with stochastic gradient descent.

For this baseline, it is configured as:

```python
SGDClassifier(loss="hinge")
```

That makes it an SVM-style linear classifier, similar in objective to `LinearSVC`, but trained with a stochastic optimizer.

## Algorithm

The model still learns a linear score over the TF-IDF vector:

```text
z = w . x + b
```

With `loss="hinge"`, it uses the same basic margin idea as a linear SVM:

```text
loss = max(0, 1 - y z)
```

The difference is optimization. Instead of using a specialized full-batch linear SVM solver, `SGDClassifier` updates the weights incrementally:

```text
for epoch in training_epochs:
    shuffle training examples
    for one review or mini-step:
        compute current margin
        compute gradient
        update w and b
```

This is useful for large sparse datasets and online/incremental learning. On this 50k-review dataset, `LinearSVC` is still expected to be the stronger or more stable baseline.

## Probability And Log-Loss

With `loss="hinge"`, plain `SGDClassifier` does not expose `predict_proba`.

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
python -m classifiers.sgd_classifier.run
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
python -m classifiers.sgd_classifier.run
```

Runtime on Luka's machine:

```text
real 20.83s
user 21.86s
sys  1.36s
```

Optimizer status:

- loss: `hinge`
- penalty: `l2`
- alpha: `0.0001`
- configured `max_iter`: `1000`
- actual `n_iter_`: `13`
- convergence warning: none

Held-out test split:

- train rows: `40,000`
- test rows: `10,000`
- test negatives: `5,000`
- test positives: `5,000`

Metrics:

- accuracy: `0.9111`
- balanced accuracy: `0.9111`
- precision: `0.9052`
- recall: `0.9184`
- F1: `0.9117`
- ROC-AUC: `0.9714`
- PR-AUC: `0.9707`
- MCC: `0.8223`
- log-loss: unavailable for `SGDClassifier(loss="hinge")`

Confusion matrix:

| actual \\ predicted | negative | positive |
| --- | ---: | ---: |
| negative | 4,519 | 481 |
| positive | 408 | 4,592 |

This lands between `LogisticRegression` and `LinearSVC` on the current baseline setup:

- `LogisticRegression`: accuracy `0.9095`, F1 `0.9101`
- `SGDClassifier(loss="hinge")`: accuracy `0.9111`, F1 `0.9117`
- `LinearSVC`: accuracy `0.9150`, F1 `0.9152`
