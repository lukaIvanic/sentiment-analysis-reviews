# ExtraTreesClassifier

`ExtraTreesClassifier` is an ensemble of randomized decision trees.

It is closely related to `RandomForestClassifier`: both train many trees and combine their predictions by voting. The main difference is that Extra Trees adds more randomness when building the trees.

## Algorithm

The ensemble works like this:

- each tree receives the training data
- each split considers only a random subset of features
- for those features, split thresholds are randomized more aggressively than in a normal random forest
- final class prediction is the majority vote across trees
- class probabilities are averaged across trees

The default Extra Trees setup in sklearn does not use bootstrap sampling. Random Forest usually varies trees through bootstrap samples plus random feature subsets. Extra Trees varies trees mostly through random split choices plus random feature subsets.

For classification, the default split criterion is Gini impurity:

```text
Gini = 1 - sum(p_class^2)
```

This is still not gradient descent or backpropagation. It is a tree ensemble built by randomized greedy splitting.

## Why Compare It After Random Forest

Extra Trees is useful here because it answers a narrow question: if a Random Forest is somewhat awkward on sparse TF-IDF text, does a more randomized tree ensemble help?

The expectation is that it may train quickly and may be competitive with Random Forest, but it is still unlikely to beat linear classifiers on this representation. Sparse n-gram sentiment classification usually benefits from additive linear evidence across many words and phrases.

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

The baseline uses:

```python
ExtraTreesClassifier(
    n_estimators=100,
    criterion="gini",
    max_features="sqrt",
    bootstrap=False,
    n_jobs=-1,
    random_state=42,
)
```

Run from the project root:

```bash
python -m classifiers.extra_trees.run
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
python -m classifiers.extra_trees.run
```

Runtime on Luka's machine:

```text
real 37.75s
user 129.70s
sys  2.15s
```

Ensemble status:

- trees: `100`
- criterion: `gini`
- tree `max_depth`: unlimited / `None`
- split `max_features`: `sqrt`
- bootstrap: `False`
- actual TF-IDF features: `50,000`

Held-out test split:

- train rows: `40,000`
- test rows: `10,000`
- test negatives: `5,000`
- test positives: `5,000`

Metrics:

- accuracy: `0.8749`
- balanced accuracy: `0.8749`
- precision: `0.8820`
- recall: `0.8656`
- F1: `0.8737`
- ROC-AUC: `0.9466`
- PR-AUC: `0.9398`
- MCC: `0.7499`
- log-loss: `0.4342`

Confusion matrix:

| actual \\ predicted | negative | positive |
| --- | ---: | ---: |
| negative | 4,421 | 579 |
| positive | 672 | 4,328 |

This beats the `RandomForestClassifier` baseline but remains below the linear text classifiers:

- `RandomForestClassifier`: accuracy `0.8633`, F1 `0.8625`
- `ExtraTreesClassifier`: accuracy `0.8749`, F1 `0.8737`
- `MultinomialNB`: accuracy `0.8841`, F1 `0.8846`
- `LogisticRegression`: accuracy `0.9095`, F1 `0.9101`
- `SGDClassifier(loss="hinge")`: accuracy `0.9111`, F1 `0.9117`
- `LinearSVC`: accuracy `0.9150`, F1 `0.9152`

The result supports the same pattern as Random Forest: tree ensembles can learn useful text signals, but sparse TF-IDF sentiment classification is still better served by models that add evidence linearly across many tokens.
