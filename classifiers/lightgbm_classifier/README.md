# LightGBMClassifier

`LightGBMClassifier` is a gradient-boosted tree classifier.

Like `XGBoostClassifier`, it builds an additive sequence of trees. Each new tree improves the current ensemble rather than acting as an independent voter.

## Algorithm

For binary classification, LightGBM learns an additive tree score and converts it to a probability:

```text
P(positive | x) = sigmoid(score)
```

The score is a sum of tree outputs:

```text
score(x) = tree_1(x) + tree_2(x) + ... + tree_M(x)
```

The objective is binary log-loss plus regularization:

```text
objective = binary_log_loss + regularization
```

The key implementation difference is that LightGBM commonly grows trees leaf-wise. Instead of expanding all leaves level by level, it repeatedly expands the leaf that gives the largest loss reduction. This can make it fast and accurate, but it also means `num_leaves`, `min_child_samples`, and depth constraints matter for overfitting.

## Why Compare It After XGBoost

Both XGBoost and LightGBM are boosted tree models:

```text
tree 1 -> tree 2 corrects errors -> tree 3 corrects remaining errors -> ...
```

LightGBM is usually optimized for speed and memory usage. On sparse TF-IDF text, it may train faster than XGBoost, but it is still not guaranteed to beat linear classifiers because sentiment evidence is spread across many small word and n-gram signals.

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
LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    metric="binary_logloss",
    n_estimators=200,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    force_col_wise=True,
    n_jobs=-1,
    random_state=42,
)
```

Run from the project root:

```bash
python -m classifiers.lightgbm_classifier.run
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
python -m classifiers.lightgbm_classifier.run
```

Runtime on Luka's machine:

```text
real 83.78s
user 193.16s
sys  38.80s
```

Boosting status:

- boosted trees: `200`
- objective: `binary`
- boosting type: `gbdt`
- metric: `binary_logloss`
- learning rate: `0.1`
- num leaves: `31`
- max depth: unlimited / `-1`
- min child samples: `20`
- subsample: `0.8`
- colsample by tree: `0.8`
- force column-wise histogram building: `True`
- actual TF-IDF features: `50,000`

Held-out test split:

- train rows: `40,000`
- test rows: `10,000`
- test negatives: `5,000`
- test positives: `5,000`

Metrics:

- accuracy: `0.8918`
- balanced accuracy: `0.8918`
- precision: `0.8865`
- recall: `0.8986`
- F1: `0.8925`
- ROC-AUC: `0.9605`
- PR-AUC: `0.9599`
- MCC: `0.7837`
- log-loss: `0.2617`

Confusion matrix:

| actual \\ predicted | negative | positive |
| --- | ---: | ---: |
| negative | 4,425 | 575 |
| positive | 507 | 4,493 |

This is the best tree-based baseline so far:

- `XGBoostClassifier`: accuracy `0.8509`, F1 `0.8553`, log-loss `0.3565`
- `RandomForestClassifier`: accuracy `0.8633`, F1 `0.8625`, log-loss `0.4554`
- `ExtraTreesClassifier`: accuracy `0.8749`, F1 `0.8737`, log-loss `0.4342`
- `LightGBMClassifier`: accuracy `0.8918`, F1 `0.8925`, log-loss `0.2617`

It still does not catch the strongest linear classifier on accuracy:

- `LogisticRegression`: accuracy `0.9095`, F1 `0.9101`, log-loss `0.2707`
- `SGDClassifier(loss="hinge")`: accuracy `0.9111`, F1 `0.9117`
- `LinearSVC`: accuracy `0.9150`, F1 `0.9152`

However, its probability quality is strong in this baseline: `LightGBMClassifier` has slightly better log-loss than `LogisticRegression` in the current runs. That makes it the most interesting tree model so far, especially if we later compare probability-based metrics separately from hard-label accuracy.
