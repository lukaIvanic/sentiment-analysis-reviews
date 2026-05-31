# Raw Datasets

This directory contains the original downloaded datasets for the selected sentiment-analysis project.

## Kaggle IMDb 50K Movie Reviews

Path:

`kaggle_imdb_50k/IMDB Dataset.csv`

Archive:

`kaggle_imdb_50k/imdb-dataset-of-50k-movie-reviews.zip`

Rows:

- total: `50,000`
- positive: `25,000`
- negative: `25,000`

Columns:

- `review`
- `sentiment`

Archive SHA-256:

`73a235bc5fc4df57bb5d517afa480fe6bfd4e2afc25dc5e5867fc87f2d25614d`

## Stanford IMDb Large Movie Review Dataset

Path:

`stanford_imdb/aclImdb`

Archive:

`stanford_imdb/aclImdb_v1.tar.gz`

Labelled split:

- `train/pos`: `12,500`
- `train/neg`: `12,500`
- `test/pos`: `12,500`
- `test/neg`: `12,500`

Extra unlabeled split:

- `train/unsup`: `50,000`

Archive SHA-256:

`c40f74a18d3b61f90feba1e17730e0d38e8b97c05fde7008942e91923d1658fe`

Do not edit raw dataset files directly. Put generated outputs under `sentiment_analysis_reviews/data/processed`.
