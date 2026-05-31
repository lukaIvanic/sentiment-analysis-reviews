# Hard VotingClassifier

This ensemble uses scikit-learn's ready-made `VotingClassifier`:

```python
from sklearn.ensemble import VotingClassifier
```

Unlike the first soft-voting ensemble, this one gives every member its own complete text pipeline:

```text
raw review text -> member-specific TF-IDF -> member classifier -> class vote
```

That means the members do not need to share one vocabulary. This is useful for hard voting because it can include classifiers that do not expose probabilities, such as `LinearSVC`, `SGDClassifier(loss="hinge")`, and `PassiveAggressiveClassifier`.

## Members

This hard-voting ensemble includes the required classifier families trained so far, plus the tuned MultinomialNB variant:

- `MultinomialNB`
- `MultinomialNB` tuned variant
- `ComplementNB`
- `LogisticRegression`
- `LinearSVC`
- `SGDClassifier`
- `PassiveAggressiveClassifier`
- `RandomForestClassifier`
- `ExtraTreesClassifier`
- `XGBoostClassifier`
- `LightGBMClassifier`

That gives `11` voters, so exact hard-vote ties are impossible.

## Hard Voting

Each member predicts a class:

```text
model_1 -> positive
model_2 -> positive
model_3 -> negative
...
```

The final prediction is the majority class:

```text
positive votes > negative votes -> positive
negative votes > positive votes -> negative
```

Hard voting does not expose calibrated probabilities. The run still records extra vote-fraction metrics:

```text
positive_vote_fraction = positive_votes / number_of_members
```

Those vote fractions are useful for ROC-AUC and PR-AUC as ordinal scores, but they should not be treated as calibrated probabilities.

## Run

Run from the project root:

```bash
python -m classifiers.hard_voting_classifier.run
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
python -m classifiers.hard_voting_classifier.run
```

Runtime on Luka's machine:

```text
real 421.79s
user 803.23s
sys  173.12s
```

Ensemble status:

- ensemble type: hard voting
- members: `11`
- vote ties: `0`
- independent TF-IDF pipeline per member: `True`
- includes tuned MultinomialNB variant: `True`
- tree ensemble size: `100` trees for Random Forest and Extra Trees
- boosted tree ensemble size: `200` trees for XGBoost and LightGBM

Held-out test split:

- train rows: `40,000`
- test rows: `10,000`
- test negatives: `5,000`
- test positives: `5,000`

Metrics:

- accuracy: `0.9107`
- balanced accuracy: `0.9107`
- precision: `0.9059`
- recall: `0.9166`
- F1: `0.9112`
- MCC: `0.8215`
- standard ROC-AUC: unavailable for hard voting
- standard PR-AUC: unavailable for hard voting
- standard log-loss: unavailable for hard voting

Vote-fraction score metrics:

- vote-fraction ROC-AUC: `0.9574`
- vote-fraction PR-AUC: `0.9416`
- vote-fraction clipped log-loss: `0.9574`

The vote-fraction metrics use:

```text
positive_vote_fraction = positive_votes / 11
```

These are useful as rough ordinal scores, but they are not calibrated probabilities. The high clipped log-loss is a reminder that hard vote fractions are much coarser than probability outputs from models such as `LogisticRegression` or `LightGBMClassifier`.

Confusion matrix:

| actual \\ predicted | negative | positive |
| --- | ---: | ---: |
| negative | 4,524 | 476 |
| positive | 417 | 4,583 |

Comparison with the other ensemble and strongest individual models:

- `VotingClassifier` soft: accuracy `0.9095`, F1 `0.9102`, log-loss `0.2654`
- `VotingClassifier` hard: accuracy `0.9107`, F1 `0.9112`, log-loss unavailable
- `LogisticRegression`: accuracy `0.9095`, F1 `0.9101`, log-loss `0.2707`
- `SGDClassifier(loss="hinge")`: accuracy `0.9111`, F1 `0.9117`
- `LinearSVC`: accuracy `0.9150`, F1 `0.9152`

The hard voter improves slightly over the soft voter on hard-label metrics, likely because it can include `LinearSVC`, `SGDClassifier`, and `PassiveAggressiveClassifier`. It still does not beat the best individual model, `LinearSVC`, and it is much slower to train because it fits a separate TF-IDF vectorizer for every member.

The runtime mostly comes from fitting the ensemble from scratch. In this script, `VotingClassifier.fit(...)` retrains every member pipeline:

```text
fit TF-IDF -> fit classifier
fit TF-IDF -> fit classifier
...
```

for all `11` members. The majority vote itself is cheap. A separate script that loads the already-saved `outputs/baselines/*/model.joblib` pipelines and only calls `predict(...)` on the test set would be much faster, because those saved pipelines already contain their fitted TF-IDF vocabularies, IDF weights, and trained classifiers.
