# Pretrained Transformer Fine-Tuning

This folder contains the PyTorch/Hugging Face runner for fine-tuning pretrained
transformers on the Kaggle IMDb sentiment dataset.

## Purpose

The classical TF-IDF baselines are strong, especially `LinearSVC`, but they are
not close to modern sentiment-analysis practice. This runner tests whether
compact pretrained transformer models improve the held-out IMDb result while
staying within a small GPU budget.

## Data Split

The runner uses the same project split convention:

- Kaggle IMDb 50k reviews
- 80% train+validation, 20% held-out test
- 10% of the train+validation pool used as validation
- `random_state=42`

That gives:

- train rows: `36,000`
- validation rows: `4,000`
- test rows: `10,000`

## Runs

All full runs below were executed remotely on an RTX 3090 Vast.ai instance.

| Run | Model | Max length | Batch | Epochs | Best val acc | Test acc | Test F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Scratch best | 263k-param tiny transformer, MLX | 512 | 16 effective | 5 | 0.8935 | 0.8943 | 0.8963 | 0.9603 |
| Scratch CUDA context check | 263k-param tiny transformer | 1024 | 16 | 5 | 0.8742 | 0.8840 | 0.8821 | 0.9529 |
| Pre-FT diagnostic | `distilbert-base-uncased` + random classifier head | 512 | 64 eval | 0 | n/a | 0.4036 | 0.4531 | 0.3712 |
| Fine-tune sanity | `distilbert-base-uncased` | 512 | 32 | 3 | 0.9340 | 0.9369 | 0.9374 | 0.9840 |
| Pre-FT diagnostic | `microsoft/deberta-v3-small` + random classifier head | 512 | 32 eval | 0 | n/a | 0.5000 | 0.0000 | 0.4869 |
| Fine-tune main | `microsoft/deberta-v3-small` | 512 | 16 | 3 | 0.9565 | 0.9564 | 0.9566 | 0.9895 |

The important result is that pretrained fine-tuning clearly beats the best
classical baseline:

- best classical baseline: `LinearSVC`, test accuracy `0.9150`
- best pretrained transformer: `DeBERTa-v3-small`, test accuracy `0.9564`

The pre-FT diagnostic rows were run with `--epochs 0`. They are not fair
zero-shot sentiment results: `AutoModelForSequenceClassification` loads the
pretrained encoder but initializes a new binary classification head. DeBERTa
therefore reached `0.5000` accuracy by predicting every test example as
negative, while DistilBERT landed at `0.4036` with a negative MCC.

## Commands

DistilBERT:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -m classifiers.pretrained_transformer_torch.run \
  --model-name distilbert-base-uncased \
  --output-dir outputs/transformer/distilbert_s512_lr2e5_b32_e3 \
  --max-length 512 \
  --epochs 3 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.06 \
  --amp-dtype float16 \
  --num-workers 2 \
  --log-every 100 \
  --save-predictions
```

DeBERTa-v3-small:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -m classifiers.pretrained_transformer_torch.run \
  --model-name microsoft/deberta-v3-small \
  --output-dir outputs/transformer/deberta_v3_small_s512_lr2e5_b16_e3 \
  --max-length 512 \
  --epochs 3 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.06 \
  --amp-dtype float16 \
  --num-workers 2 \
  --log-every 100 \
  --save-predictions
```

Pre-fine-tune diagnostic:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -m classifiers.pretrained_transformer_torch.run \
  --model-name distilbert-base-uncased \
  --output-dir outputs/transformer/distilbert_s512_pre_finetune_eval \
  --max-length 512 \
  --epochs 0 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.06 \
  --amp-dtype float16 \
  --num-workers 2

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -m classifiers.pretrained_transformer_torch.run \
  --model-name microsoft/deberta-v3-small \
  --output-dir outputs/transformer/deberta_v3_small_s512_pre_finetune_eval \
  --max-length 512 \
  --epochs 0 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.06 \
  --amp-dtype float16 \
  --num-workers 2
```

## Notes

- `transformers>=4.44,<5` is used because the earlier accidental
  `transformers 5.x` install was incompatible with the remote Torch build.
- `protobuf` is required for the DeBERTa-v3 tokenizer conversion path.
- The generated output directories contain the metric JSON, training history,
  confusion matrix, classification report, and optional predictions.
