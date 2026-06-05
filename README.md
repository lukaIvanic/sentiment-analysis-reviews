# Sentiment Analysis Project

Actual implementation folder for topic 7:

`Analiza sentimenta recenzija (tekst klasifikacija)`

Use the `AGENTS.md` in this folder for the full course task definition.

## Submission Artifacts

Final professor-facing artifacts:

- report: `reports/final_report/analiza_sentimenta_recenzija.docx`
- report source: `reports/final_report/report.md`
- presentation: `presentation/analiza_sentimenta_recenzija.pptx`
- presentation notes: `presentation/README.md`
- generated explanatory figures: `figures/generated/*.png`
- final result tables: `reports/final_report/results_table.csv` and
  `reports/final_report/results_tables.md`

## Data

Raw datasets are in `data/raw`:

- `kaggle_imdb_50k/IMDB Dataset.csv`
- `stanford_imdb/aclImdb`

Processed files should go in `data/processed`.

The main experiments use the Kaggle CSV. The Stanford ACL IMDB dataset is also
downloaded because it is listed as a professor-provided source for the topic.
Neither dataset requires manual labelling.

## Setup

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional extras:

```bash
# Apple MLX experiments
python -m pip install -e ".[mlx]"

# PyTorch / pretrained transformer experiments
python -m pip install -e ".[pretrained]"
```

All commands below assume they are run from the repository root with the
environment activated. If the package is not installed in editable mode, prefix
commands with `PYTHONPATH=.`.

## Expected Implementation

The final implementation should:

- load IMDb review data
- vectorize text with `TF-IDF`
- train and tune the required classifiers
- evaluate using the required metrics
- build a final `VotingClassifier` or `StackingClassifier`
- save tables/figures needed for the report and presentation

## Reproducing Main Runs

Run the best classical baseline:

```bash
PYTHONPATH=. python -m classifiers.linear_svc.run
```

Run the broad RandomizedSearchCV coverage search used in the final report:

```bash
PYTHONPATH=. python -m classifiers.randomized_search_cv.run \
  --models all \
  --n-iter 5 \
  --cv 3 \
  --scoring f1 \
  --output-dir outputs/searches/randomized_search_cv_required_n5_cv3
```

Run the final soft-voting ensemble:

```bash
PYTHONPATH=. python -m classifiers.voting_classifier.run
```

Run the hard-voting ensemble:

```bash
PYTHONPATH=. python -m classifiers.hard_voting_classifier.run
```

Run the prefit stacking ensemble:

```bash
PYTHONPATH=. python -m classifiers.stacking_classifier_prefit.run
```

Generated model artifacts are written under `outputs/`. Each run writes at
least:

- `metrics.json`
- `confusion_matrix.json`
- `classification_report.txt`
- `run_config.json`

The full pretrained transformer runs are documented in
`classifiers/pretrained_transformer_torch/README.md`. They were run on a remote
RTX 3090 instance, not on the local Mac.

## Current Best Results

The strongest classical baseline so far is `LinearSVC` with TF-IDF features:

- test accuracy: `0.9150`
- test F1: `0.9152`
- test ROC-AUC: `0.9720`

The strongest transformer result so far is pretrained `microsoft/deberta-v3-small`
fine-tuned on the Kaggle IMDb split:

- validation accuracy: `0.9565`
- test accuracy: `0.9564`
- test F1: `0.9566`
- test ROC-AUC: `0.9895`

The transformer experiments were run on a remote RTX 3090 Vast.ai instance, not
on the local Mac. Generated artifacts are under `outputs/transformer/`; the
runner code is in `classifiers/pretrained_transformer_torch`.

## Generated Figures

The final report uses labeled generated infographics stored under
`figures/generated/`. They explain the TF-IDF pipeline, model families,
RandomizedSearchCV, metrics/log-loss, ensembles, result summaries, and the
optional pretrained-transformer extension.

## Notes For Review

- Required TF-IDF/classical work and optional transformer work are separate.
- `LinearSVC`, hinge-loss `SGDClassifier`, `PassiveAggressiveClassifier`, and
  hard voting do not expose calibrated class probabilities in the chosen
  configuration, so standard log-loss is not reported for those rows.
- The final report and presentation copy the important results into tracked
  files because `outputs/` is intentionally ignored.
