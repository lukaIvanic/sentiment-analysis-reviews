# XGBoostClassifier

`XGBoostClassifier` is a gradient-boosted tree classifier.

Unlike `RandomForestClassifier` and `ExtraTreesClassifier`, the trees are not trained independently and then simply voted together. XGBoost builds trees sequentially: each new tree is trained to improve the current ensemble.

## Algorithm

For binary classification, XGBoost produces a logit score and converts it to a probability:

```text
P(positive | x) = sigmoid(score)
```

The score is an additive sum of tree outputs:

```text
score(x) = tree_1(x) + tree_2(x) + ... + tree_M(x)
```

At each boosting step, XGBoost adds a new tree that improves the objective. For logistic binary classification, that objective is based on log-loss plus regularization:

```text
objective = log_loss + tree_complexity_penalty
```

So this is still not neural-network backpropagation over dense layers. It is gradient boosting over decision trees: gradients guide which tree should be added next, but the model itself remains an ensemble of trees.

## Why Compare It After Random Forest And Extra Trees

Random Forest and Extra Trees are bagging-style ensembles:

```text
many independent trees -> vote/average
```

XGBoost is a boosting ensemble:

```text
tree 1 -> tree 2 corrects errors -> tree 3 corrects remaining errors -> ...
```

This makes it a genuinely different tree-based classifier. It is often very strong on tabular data, but sparse TF-IDF text can still favor linear classifiers because sentiment evidence is spread additively across many words and phrases.

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
XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)
```

Run from the project root:

```bash
python -m classifiers.xgboost_classifier.run
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
python -m classifiers.xgboost_classifier.run
```

Runtime on Luka's machine:

```text
real 99.99s
user 256.35s
sys  62.17s
```

Boosting status:

- boosted trees: `200`
- objective: `binary:logistic`
- eval metric: `logloss`
- learning rate: `0.1`
- max depth: `4`
- subsample: `0.8`
- colsample by tree: `0.8`
- tree method: `hist`
- actual TF-IDF features: `50,000`

Held-out test split:

- train rows: `40,000`
- test rows: `10,000`
- test negatives: `5,000`
- test positives: `5,000`

Metrics:

- accuracy: `0.8509`
- balanced accuracy: `0.8509`
- precision: `0.8307`
- recall: `0.8814`
- F1: `0.8553`
- ROC-AUC: `0.9355`
- PR-AUC: `0.9337`
- MCC: `0.7031`
- log-loss: `0.3565`

Confusion matrix:

| actual \\ predicted | negative | positive |
| --- | ---: | ---: |
| negative | 4,102 | 898 |
| positive | 593 | 4,407 |

This baseline is weaker than the other tree ensembles on accuracy, but its predicted probabilities are less poorly calibrated than the plain forest probabilities:

- `XGBoostClassifier`: accuracy `0.8509`, F1 `0.8553`, log-loss `0.3565`
- `RandomForestClassifier`: accuracy `0.8633`, F1 `0.8625`, log-loss `0.4554`
- `ExtraTreesClassifier`: accuracy `0.8749`, F1 `0.8737`, log-loss `0.4342`

It remains below the best linear text classifiers:

- `LogisticRegression`: accuracy `0.9095`, F1 `0.9101`, log-loss `0.2707`
- `SGDClassifier(loss="hinge")`: accuracy `0.9111`, F1 `0.9117`
- `LinearSVC`: accuracy `0.9150`, F1 `0.9152`

This run is a useful baseline, but it is not a tuned XGBoost result. Boosted trees have more important hyperparameters than the previous linear baselines, especially tree depth, learning rate, number of estimators, and feature sampling.
