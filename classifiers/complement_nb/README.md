# ComplementNB

`ComplementNB` is the second Naive Bayes baseline for the IMDb sentiment project.

It is closely related to `MultinomialNB`, but estimates each class using the complement of that class. For binary sentiment classification:

- the `positive` class is estimated from the `negative` reviews
- the `negative` class is estimated from the `positive` reviews

The classifier then penalizes a class when the review looks too much like that class's complement. This often improves Naive Bayes behavior for text classification, especially when classes are imbalanced. The Kaggle IMDb split is balanced, so this is mostly a clean sibling comparison against `MultinomialNB`.

This run uses the same TF-IDF setup as the MultinomialNB baseline:

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
python -m classifiers.complement_nb.run
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
python -m classifiers.complement_nb.run
```

Runtime on Luka's machine:

```text
real 20.40s
user 20.60s
sys  1.13s
```

Held-out test split:

- train rows: `40,000`
- test rows: `10,000`
- test negatives: `5,000`
- test positives: `5,000`

Metrics:

- accuracy: `0.8841`
- balanced accuracy: `0.8841`
- precision: `0.8808`
- recall: `0.8884`
- F1: `0.8846`
- ROC-AUC: `0.9531`
- PR-AUC: `0.9513`
- MCC: `0.7682`
- log-loss: `0.3135`

Confusion matrix:

| actual \\ predicted | negative | positive |
| --- | ---: | ---: |
| negative | 4,399 | 601 |
| positive | 558 | 4,442 |

With the current balanced Kaggle IMDb split and the same TF-IDF defaults, this result is identical to the plain `MultinomialNB` baseline.

This is plausible because the task is binary and perfectly balanced in the train/test split. In a two-class problem, the complement of `positive` is simply `negative`, and the complement of `negative` is simply `positive`. Since both classes have the same number of examples and the same TF-IDF representation is used, the complement estimates end up producing the same decision boundary as the direct class estimates in this baseline configuration.

In more imbalanced or multi-class text datasets, `ComplementNB` can behave differently from `MultinomialNB` because complement statistics reduce the tendency for the largest class to dominate the estimated feature probabilities.
