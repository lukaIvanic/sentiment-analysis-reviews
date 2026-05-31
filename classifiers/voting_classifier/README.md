# VotingClassifier

`VotingClassifier` is a ready-made scikit-learn ensemble class:

```python
from sklearn.ensemble import VotingClassifier
```

It combines several classifiers into one final classifier.

## VotingClassifier vs StackingClassifier

`VotingClassifier` combines base model predictions directly:

- hard voting: each model predicts a class, and the majority class wins
- soft voting: each model predicts probabilities, and the averaged probability wins

For binary sentiment, soft voting looks like this:

```text
P_final(positive) =
    average(
        P_logistic(positive),
        P_multinomial_nb(positive),
        P_lightgbm(positive),
    )
```

`StackingClassifier` is also from scikit-learn:

```python
from sklearn.ensemble import StackingClassifier
```

Stacking trains a second-level model on the outputs of the base models. For example:

```text
review -> base models -> model probabilities -> meta-classifier -> final prediction
```

Stacking can be stronger, but it is easier to overfit if it is not done with cross-validated out-of-fold predictions. Scikit-learn handles that when using `StackingClassifier(cv=...)`, but it is more complex and slower than voting.

## Saved Models

The baseline model artifacts are saved under:

```text
outputs/baselines/*/model.joblib
```

For this ensemble, we do not directly load those fitted model files. Instead, the ensemble is fitted as a fresh shared pipeline:

```text
raw review text -> one shared TF-IDF vectorizer -> VotingClassifier
```

This is cleaner than averaging already-fitted full pipelines because every ensemble member receives the exact same TF-IDF matrix from the same vocabulary.

## Ensemble Members

This first ensemble uses probability-capable models:

- `LogisticRegression`
- `MultinomialNB`
- `LightGBMClassifier`

It uses soft voting, so all three models contribute predicted probabilities.

## TF-IDF Setup

The ensemble uses the same baseline TF-IDF setup:

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
python -m classifiers.voting_classifier.run
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
python -m classifiers.voting_classifier.run
```

Runtime on Luka's machine:

```text
real 117.81s
user 198.05s
sys  77.19s
```

Ensemble status:

- ensemble type: soft voting
- members: `LogisticRegression`, `MultinomialNB`, `LightGBMClassifier`
- weights: equal / `None`
- actual TF-IDF features: `50,000`

Held-out test split:

- train rows: `40,000`
- test rows: `10,000`
- test negatives: `5,000`
- test positives: `5,000`

Metrics:

- accuracy: `0.9095`
- balanced accuracy: `0.9095`
- precision: `0.9036`
- recall: `0.9168`
- F1: `0.9102`
- ROC-AUC: `0.9707`
- PR-AUC: `0.9702`
- MCC: `0.8191`
- log-loss: `0.2654`

Confusion matrix:

| actual \\ predicted | negative | positive |
| --- | ---: | ---: |
| negative | 4,511 | 489 |
| positive | 416 | 4,584 |

Compared with the main individual members:

- `MultinomialNB`: accuracy `0.8841`, F1 `0.8846`, log-loss `0.3135`
- `LightGBMClassifier`: accuracy `0.8918`, F1 `0.8925`, log-loss `0.2617`
- `LogisticRegression`: accuracy `0.9095`, F1 `0.9101`, log-loss `0.2707`
- `VotingClassifier`: accuracy `0.9095`, F1 `0.9102`, log-loss `0.2654`

The ensemble essentially matches `LogisticRegression` on hard-label accuracy and F1, while improving its log-loss. It does not beat `LinearSVC` on accuracy/F1, but unlike `LinearSVC` it exposes probabilities without calibration.

This satisfies the required final-model family (`VotingClassifier` or `StackingClassifier`) with an interpretable first ensemble. A later tuned version could choose weights or ensemble members using cross-validation instead of the held-out test set.
