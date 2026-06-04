# Sentiment Analysis Project

Actual implementation folder for topic 7:

`Analiza sentimenta recenzija (tekst klasifikacija)`

Use the `AGENTS.md` in this folder for the full course task definition.

## Data

Raw datasets are in `data/raw`:

- `kaggle_imdb_50k/IMDB Dataset.csv`
- `stanford_imdb/aclImdb`

Processed files should go in `data/processed`.

## Expected Implementation

The final implementation should:

- load IMDb review data
- vectorize text with `TF-IDF`
- train and tune the required classifiers
- evaluate using the required metrics
- build a final `VotingClassifier` or `StackingClassifier`
- save tables/figures needed for the report and presentation

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
