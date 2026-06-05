# MultinomialNB

`MultinomialNB` is the first baseline for the IMDb sentiment project.

It is a Naive Bayes classifier designed for non-negative count-like features. TF-IDF values are non-negative, so the model is a natural fit for classic text classification. The model estimates which words are more likely under each class, then combines those word contributions to decide whether a review is more likely positive or negative.

Why it is a good first classifier:

- it is fast on the 50k-row Kaggle dataset
- it works well with sparse TF-IDF matrices
- it exposes `predict_proba`, which lets us calculate ROC-AUC, PR-AUC, and log-loss cleanly
- it gives a simple baseline before stronger linear models

## Vocabulary And TF-IDF

This pipeline does not use byte-pair encoding, WordPiece, embeddings, or transformer-style tokenization.

Vocabulary generation is handled by sklearn's `TfidfVectorizer` during `pipeline.fit(...)`, using the training reviews only.

Current baseline settings:

```python
TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    token_pattern=r"(?u)\b\w\w+\b",  # sklearn default
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=50_000,
    sublinear_tf=True,
)
```

The default token pattern keeps word-like tokens with at least two characters. With `ngram_range=(1, 2)`, the vocabulary contains both unigrams and bigrams, for example:

```text
"excellent"
"bad"
"not good"
"waste time"
```

`min_df=2` removes tokens/ngrams that appear in fewer than two training documents. `max_df=0.95` removes tokens/ngrams that appear in more than 95% of training documents. `max_features=50_000` keeps the most frequent remaining vocabulary items.

After fitting, the vocabulary is a fixed mapping from token/ngram to sparse-vector column:

```python
{
    "excellent": 123,
    "not good": 456,
    "waste time": 789,
}
```

The held-out test reviews are transformed using this same fixed vocabulary. Tokens that only appear in the test set and were never learned from the training set are ignored.

TF-IDF means:

```text
TF-IDF(term, document) = TF(term in document) * IDF(term)
```

`TF` measures how much a term appears in one review. `IDF` downweights terms that appear in many reviews, because very common terms usually carry less class-specific information.

Run from the project root:

```bash
python -m classifiers.multinomial_nb.run
```

NB-specific tuned run with `RandomizedSearchCV`:

```bash
python -m classifiers.multinomial_nb.run --tune
```

`RandomizedSearchCV` samples model/vectorizer hyperparameter combinations and scores them with cross-validation on the training set. For example, `--n-iter 10 --cv 3` means:

- sample `10` random parameter combinations from the search space
- score each combination using `3` cross-validation folds inside the training split
- run about `10 * 3 = 30` train/validation fits
- refit the best combination on the full training split
- evaluate once on the held-out test split

The held-out test split is not used to choose hyperparameters.

Optional quick smoke-test run:

```bash
python -m classifiers.multinomial_nb.run --limit 1000 --output-dir outputs/baselines/multinomial_nb_smoke
```

Optional quick tuned smoke-test run:

```bash
python -m classifiers.multinomial_nb.run --limit 1000 --tune --n-iter 5 --cv 3 --output-dir outputs/baselines/multinomial_nb_tuned_smoke
```

The final report also includes a broader `RandomizedSearchCV` coverage run over
all ten required classifier families:

```bash
python -m classifiers.randomized_search_cv.run --models all --n-iter 5 --cv 3 --scoring f1 --output-dir outputs/searches/randomized_search_cv_required_n5_cv3
```

Expected outputs:

- `run_config.json`
- `metrics.json`
- `confusion_matrix.json`
- `classification_report.txt`
- `search_results.csv` when `--tune` is used
- `model.joblib`

## Baseline Result

Plain baseline run:

```bash
python -m classifiers.multinomial_nb.run
```

Runtime on Luka's machine:

```text
real 22.46s
user 21.42s
sys  1.57s
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

## Tuned Result

Randomized search run:

```bash
python -m classifiers.multinomial_nb.run --tune --n-iter 10 --cv 3 --output-dir outputs/baselines/multinomial_nb_tuned_n10_cv3
```

Runtime on Luka's machine:

```text
real 105.77s
user 431.33s
sys  62.46s
```

Search setup:

- candidates sampled: `10`
- CV folds: `3`
- scoring metric: `f1`
- total CV fits: `30`
- best mean CV F1: `0.8850`

Best parameters:

- `classifier__alpha`: `0.5`
- `tfidf__ngram_range`: `(1, 2)`
- `tfidf__min_df`: `3`
- `tfidf__max_df`: `0.9`
- `tfidf__max_features`: `80,000`
- `tfidf__sublinear_tf`: `true`

Held-out test metrics:

- accuracy: `0.8854`
- balanced accuracy: `0.8854`
- precision: `0.8842`
- recall: `0.8870`
- F1: `0.8856`
- ROC-AUC: `0.9554`
- PR-AUC: `0.9537`
- MCC: `0.7708`
- log-loss: `0.2934`

Confusion matrix:

| actual \\ predicted | negative | positive |
| --- | ---: | ---: |
| negative | 4,419 | 581 |
| positive | 565 | 4,435 |
