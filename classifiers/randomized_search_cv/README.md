# RandomizedSearchCV coverage

This folder contains the shared hyperparameter-search runner used to satisfy the
course requirement for `RandomizedSearchCV` with cross-validation.

The runner applies the same stratified train/test split as the baseline scripts,
then performs a modest but real `RandomizedSearchCV` for each required
TF-IDF-based classifier family:

- MultinomialNB
- ComplementNB
- LogisticRegression
- LinearSVC
- SGDClassifier
- PassiveAggressiveClassifier
- RandomForestClassifier
- ExtraTreesClassifier
- XGBoostClassifier
- LightGBMClassifier

Recommended broad coverage run:

```bash
python -m classifiers.randomized_search_cv.run \
  --models all \
  --n-iter 5 \
  --cv 3 \
  --scoring f1 \
  --output-dir outputs/searches/randomized_search_cv_required_n5_cv3
```

The output directory contains a global `summary.csv` / `summary.md` and one
subdirectory per classifier with `search_results.csv`, metrics, confusion matrix,
classification report, and run configuration.
