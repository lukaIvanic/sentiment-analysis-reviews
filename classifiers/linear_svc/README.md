# LinearSVC

`LinearSVC` is the linear Support Vector Machine baseline for the IMDb sentiment project.

It learns a linear score over the TF-IDF vector:

```text
z = w . x + b
```

The prediction is based on which side of the learned hyperplane the review falls on.

## Feature Dimensions

This baseline uses the same TF-IDF feature space as the earlier models:

```text
max_features = 50,000
```

So each review is represented as a sparse vector with up to `50,000` unigram/bigram dimensions. Most entries are zero for any individual review.

This is not the same as the raw text. The raw input is text; the model input is the TF-IDF vector produced from that text.

## Linear SVM vs Kernel SVM

The “SVM maps data into a higher-dimensional space” idea refers to kernel SVMs.

There are two broad cases:

- `LinearSVC`: learns a linear hyperplane directly in the existing feature space
- kernel SVM: uses a kernel function to behave as if data were mapped into a richer, often much higher-dimensional feature space

Common kernels include:

- linear kernel
- polynomial kernel
- RBF/Gaussian kernel

The RBF kernel can be interpreted as an implicit infinite-dimensional feature mapping. That is the classic “kernel trick”: the model gets nonlinear decision boundaries without explicitly constructing all those dimensions.

For TF-IDF text classification, the feature space is already very high-dimensional and sparse, so linear SVMs are usually the practical choice. Full kernel SVMs are often too expensive for datasets like 50,000 reviews with tens of thousands of text features.

## Hinge Loss

For labels represented as `y ∈ {-1, +1}`, the signed margin is:

```text
margin = y * z
```

The hinge loss is:

```text
loss = max(0, 1 - margin)
```

That means:

- if `margin >= 1`, the example is correctly classified with enough margin, so loss is `0`
- if `0 < margin < 1`, the example is correct but too close to the boundary, so it is penalized
- if `margin <= 0`, the example is on the wrong side of the boundary, so it is penalized more

This margin-based objective is the key difference from `LogisticRegression`, which optimizes probability/log-loss instead.

## Probability And Log-Loss

Plain `LinearSVC` does not expose `predict_proba`.

This baseline therefore:

- uses `decision_function` scores for ROC-AUC and PR-AUC
- does not calculate log-loss
- records log-loss as unavailable/null in `metrics.json`

To calculate meaningful log-loss for an SVM, we would need probability calibration, for example `CalibratedClassifierCV(LinearSVC(...))`. We are not doing that for this plain baseline.

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
python -m classifiers.linear_svc.run
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
python -m classifiers.linear_svc.run
```

Runtime on Luka's machine:

```text
real 19.38s
user 19.55s
sys  1.15s
```

Optimizer status:

- configured `max_iter`: `5000`
- actual `n_iter_`: `38`
- convergence warning: none

The model converged far before the iteration cap, so no increase to `max_iter` is needed for this baseline.

Held-out test split:

- train rows: `40,000`
- test rows: `10,000`
- test negatives: `5,000`
- test positives: `5,000`

Metrics:

- accuracy: `0.9150`
- balanced accuracy: `0.9150`
- precision: `0.9135`
- recall: `0.9168`
- F1: `0.9152`
- ROC-AUC: `0.9720`
- PR-AUC: `0.9708`
- MCC: `0.8300`
- log-loss: unavailable for uncalibrated `LinearSVC`

Confusion matrix:

| actual \\ predicted | negative | positive |
| --- | ---: | ---: |
| negative | 4,566 | 434 |
| positive | 416 | 4,584 |

This is the strongest baseline so far. It slightly improves over `LogisticRegression`, which reached `0.9095` accuracy and `0.9101` F1 with the same TF-IDF defaults.
