# RandomForestClassifier

`RandomForestClassifier` is an ensemble of decision trees.

Each tree learns a sequence of feature-threshold questions over the TF-IDF matrix. In this project those features are words or n-grams, so a split can be understood as checking whether a review has enough weight for a particular token or phrase.

## Algorithm

A random forest reduces the instability of one decision tree by training many trees and combining their votes:

- each tree is trained on a bootstrap sample of the training rows
- each split considers only a random subset of features
- final class prediction is the majority vote across trees
- class probabilities are averaged across trees

For classification, the default split criterion is Gini impurity:

```text
Gini = 1 - sum(p_class^2)
```

Pure nodes have low impurity. The tree greedily chooses splits that reduce impurity in the child nodes.

This is not gradient descent or backpropagation. It is a greedy tree-building procedure repeated many times.

## Why It May Be Slow Here

Our TF-IDF vectors are sparse and high-dimensional. With `max_features=50_000`, the forest has many possible n-gram features to choose from.

The baseline uses:

```python
RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)
```

With `max_features="sqrt"`, each split considers roughly:

```text
sqrt(50,000) ~= 224 candidate features
```

That keeps the run feasible, but tree ensembles are still expected to be slower than the linear classifiers.

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
python -m classifiers.random_forest.run
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
python -m classifiers.random_forest.run
```

Runtime on Luka's machine:

```text
real 37.15s
user 101.39s
sys  2.76s
```

Forest status:

- trees: `100`
- criterion: `gini`
- tree `max_depth`: unlimited / `None`
- split `max_features`: `sqrt`
- actual TF-IDF features: `50,000`

Held-out test split:

- train rows: `40,000`
- test rows: `10,000`
- test negatives: `5,000`
- test positives: `5,000`

Metrics:

- accuracy: `0.8633`
- balanced accuracy: `0.8633`
- precision: `0.8673`
- recall: `0.8578`
- F1: `0.8625`
- ROC-AUC: `0.9376`
- PR-AUC: `0.9299`
- MCC: `0.7266`
- log-loss: `0.4554`

Confusion matrix:

| actual \\ predicted | negative | positive |
| --- | ---: | ---: |
| negative | 4,344 | 656 |
| positive | 711 | 4,289 |

This is below the current linear baselines:

- `RandomForestClassifier`: accuracy `0.8633`, F1 `0.8625`
- `MultinomialNB`: accuracy `0.8841`, F1 `0.8846`
- `LogisticRegression`: accuracy `0.9095`, F1 `0.9101`
- `SGDClassifier(loss="hinge")`: accuracy `0.9111`, F1 `0.9117`
- `LinearSVC`: accuracy `0.9150`, F1 `0.9152`

The result is a useful contrast: tree ensembles are powerful for many tabular problems, but sparse n-gram TF-IDF sentiment classification is usually a better fit for additive linear models.
