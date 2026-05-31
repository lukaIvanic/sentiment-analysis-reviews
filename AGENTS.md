# AGENTS.md

This workspace is for the `Primjena umjetne inteligencije` course project.

## Source Of Truth

Use the introductory course PDF as the source of truth for the selected project task:

`materials/files/12488848/mod_resource/content/0/PUI - Osnovne Informacije o kolegiju.pdf`

The selected topic is:

`7. Analiza sentimenta recenzija (tekst klasifikacija)`

The `Sandi_Baressi_Segota - Teme za projekt.pdf` file contains a separate `SBS` topic list. Its topic number 7 is `Concrete Strength`; do not use that as the project task.

## Project Task

Build a Python sentiment-analysis project for IMDb movie reviews.

This is a supervised binary text classification task:

- input: movie review text
- output label: positive or negative sentiment

Required feature representation:

- `TF-IDF`

Required model comparison:

- `MultinomialNB`
- `ComplementNB`
- `LogisticRegression`
- `LinearSVC`
- `SGDClassifier`
- `PassiveAggressiveClassifier`
- `RandomForestClassifier`
- `ExtraTreesClassifier`
- `XGBoostClassifier`
- `LightGBMClassifier`

Required validation/search:

- use at least 10 classifiers
- use `RandomizedSearchCV`
- use cross-validation

Required metrics:

- accuracy (`ACC`)
- balanced accuracy (`BACC`)
- precision
- recall
- F1
- ROC-AUC
- PR-AUC
- Matthews correlation coefficient (`MCC`)
- log-loss
- confusion matrix

Required final model:

- `VotingClassifier` or `StackingClassifier`

## Datasets

Professor-provided sources:

- Kaggle: IMDb 50K Movie Reviews
- Stanford: IMDb Large Movie Review Dataset

Downloaded local data:

- `sentiment_analysis_reviews/data/raw/kaggle_imdb_50k/IMDB Dataset.csv`
- `sentiment_analysis_reviews/data/raw/stanford_imdb/aclImdb`

Do not manually label data. The selected topic already provides labelled sentiment datasets.

## Workspace Layout

Use `sentiment_analysis_reviews` for implementation:

- `sentiment_analysis_reviews/src`: Python modules and scripts
- `sentiment_analysis_reviews/notebooks`: exploratory notebooks
- `sentiment_analysis_reviews/data/raw`: original downloaded datasets
- `sentiment_analysis_reviews/data/processed`: generated processed datasets
- `sentiment_analysis_reviews/reports`: seminar/report artifacts
- `sentiment_analysis_reviews/presentation`: presentation artifacts

Treat `materials` as read-only reference material unless the user explicitly asks to update the local course archive.

## Deliverables

The course requires:

- seminar/report in `.pdf`, `.docx`, or `.tex`
- Python source code or notebook plus short run instructions
- presentation in `.tex` or `.pptx`

Deadline:

`2026-06-05 23:59`

Keep future work focused on making the project runnable, reproducible, and easy to explain in the final report and presentation.
