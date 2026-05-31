# LogisticRegression

`LogisticRegression` is the first discriminative linear baseline for the IMDb sentiment project.

Unlike `MultinomialNB` and `ComplementNB`, it does not estimate word probabilities per class. Instead, it directly learns a linear decision boundary over the TF-IDF feature vector.

## Algorithm

After TF-IDF vectorization, every review is represented as a sparse vector:

```text
x = [tfidf_1, tfidf_2, ..., tfidf_d]
```

For binary sentiment classification, Logistic Regression learns:

```text
z = w . x + b
```

where:

- `w` is one learned weight per TF-IDF feature
- `b` is a learned bias/intercept
- `z` is the logit score

The logit is converted into a probability with the sigmoid function:

```text
P(positive | x) = sigmoid(z) = 1 / (1 + exp(-z))
P(negative | x) = 1 - P(positive | x)
```

Prediction:

```text
if P(positive | x) >= 0.5:
    predict positive
else:
    predict negative
```

Training minimizes binary cross-entropy / negative log-likelihood:

```text
loss = -sum(y log(p) + (1 - y) log(1 - p))
```

with L2 regularization:

```text
regularized_loss = loss + lambda * ||w||^2
```

In sklearn, the regularization strength is controlled by `C`, where smaller `C` means stronger regularization.

## Is This A Neural Network?

Not really in the way people usually mean it. Logistic Regression is mathematically similar to a single sigmoid/logit layer with no hidden layers:

```text
TF-IDF features -> linear weights -> sigmoid probability
```

There are no embeddings, no hidden layers, no attention, and no deep backpropagation pipeline. The solver optimizes a convex objective, so for this binary linear model there is one global optimum up to numerical tolerance.

## Why It Is A Useful Next Baseline

- it usually performs strongly on sparse TF-IDF text features
- it learns feature weights directly from the labels
- coefficients are interpretable as positive/negative sentiment evidence
- it exposes `predict_proba`, so ROC-AUC, PR-AUC, and log-loss work cleanly
- it is a natural next step after Naive Bayes

## TF-IDF Setup

This run uses the same TF-IDF setup as the Naive Bayes baselines:

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
python -m classifiers.logistic_regression.run
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
python -m classifiers.logistic_regression.run
```

Runtime on Luka's machine:

```text
real 22.34s
user 28.68s
sys  4.10s
```

Optimizer status:

- configured `max_iter`: `1000`
- actual `n_iter_`: `6`
- convergence warning: none

The model converged far before the iteration cap, so no increase to `max_iter` is needed for this baseline.

Held-out test split:

- train rows: `40,000`
- test rows: `10,000`
- test negatives: `5,000`
- test positives: `5,000`

Metrics:

- accuracy: `0.9095`
- balanced accuracy: `0.9095`
- precision: `0.9044`
- recall: `0.9158`
- F1: `0.9101`
- ROC-AUC: `0.9710`
- PR-AUC: `0.9702`
- MCC: `0.8191`
- log-loss: `0.2707`

Confusion matrix:

| actual \\ predicted | negative | positive |
| --- | ---: | ---: |
| negative | 4,516 | 484 |
| positive | 421 | 4,579 |

This is a clear improvement over the Naive Bayes baselines, which reached `0.8841` accuracy and `0.8846` F1 with the same TF-IDF defaults.
