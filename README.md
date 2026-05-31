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
