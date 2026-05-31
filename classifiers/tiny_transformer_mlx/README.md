# TinyTransformerClassifier (MLX)

This is an extra from-scratch neural experiment, separate from the professor's required
scikit-learn classifier list.

## Idea

The model is a very small bidirectional transformer encoder trained directly on the IMDb
sentiment classification objective. It does not use pretrained embeddings or a pretrained
language model.

Pipeline:

1. Train a compact BPE tokenizer on the training split only.
2. Encode reviews as `[CLS] review [SEP]` token ids with padding masks.
3. Feed the sequence through 4 bidirectional transformer encoder blocks.
4. Classify from either the final `[CLS]` representation or a masked mean over tokens.

Default architecture:

- BPE vocabulary: 256 tokens
- Maximum sequence length: 128 tokens
- Transformer layers: 4
- Model width: 16
- Attention heads: 2
- Feed-forward width: 24
- Non-trainable sinusoidal positional encoding
- RMSNorm + SwiGLU feed-forward layers
- Optimizer: AdamW

The default model is intentionally tiny: about 13k trainable parameters, depending on the
exact tokenizer vocabulary size.

With the medium config (`d_model=64`, 4 heads, `ff_dim=128`) and a full 10k BPE
vocabulary, the classifier has 804,546 trainable parameters. The MLM pretraining model
has 1,444,546 trainable parameters because of the vocabulary projection head.

## Commands

Use the project venv:

```bash
source .venv/bin/activate
```

Smoke test:

```bash
python3 -m classifiers.tiny_transformer_mlx.run --limit 1000 --epochs 1 --output-dir outputs/transformer/tiny_transformer_mlx_smoke
```

Full first pass:

```bash
python3 -m classifiers.tiny_transformer_mlx.run
```

Compiled train step:

```bash
python3 -m classifiers.tiny_transformer_mlx.run --compile-step
```

Large-tokenizer classifier experiment:

```bash
python3 -m classifiers.tiny_transformer_mlx.run \
  --vocab-size 10000 \
  --d-model 64 \
  --num-heads 4 \
  --ff-dim 128 \
  --pooling cls \
  --lr-schedule warmup-cosine \
  --compile-step \
  --output-dir outputs/transformer/tiny_transformer_mlx_vocab10k_cls_sched
```

Mean-pooling variant:

```bash
python3 -m classifiers.tiny_transformer_mlx.run \
  --vocab-size 10000 \
  --d-model 64 \
  --num-heads 4 \
  --ff-dim 128 \
  --pooling mean \
  --lr-schedule warmup-cosine \
  --compile-step \
  --output-dir outputs/transformer/tiny_transformer_mlx_vocab10k_mean_sched
```

MLM pretraining:

```bash
python3 -m classifiers.tiny_transformer_mlx.pretrain_mlm \
  --vocab-size 10000 \
  --d-model 64 \
  --num-heads 4 \
  --ff-dim 128 \
  --lr-schedule warmup-cosine \
  --compile-step \
  --output-dir outputs/transformer/tiny_transformer_mlx_mlm_vocab10k
```

Fine-tune from MLM-pretrained encoder weights:

```bash
python3 -m classifiers.tiny_transformer_mlx.run \
  --vocab-size 10000 \
  --d-model 64 \
  --num-heads 4 \
  --ff-dim 128 \
  --pooling mean \
  --lr-schedule warmup-cosine \
  --compile-step \
  --pretrained-weights outputs/transformer/tiny_transformer_mlx_mlm_vocab10k/mlm_model_weights.npz \
  --output-dir outputs/transformer/tiny_transformer_mlx_vocab10k_mean_mlm_finetune
```

Causal decoder label generation:

```bash
python3 -m classifiers.tiny_transformer_mlx.run_decoder \
  --tokenizer-path outputs/transformer/tiny_transformer_mlx_vocab10k_mean_warmup_cosine_full/tokenizer.json \
  --vocab-size 10000 \
  --d-model 64 \
  --num-heads 4 \
  --ff-dim 128 \
  --lr-schedule warmup-cosine \
  --lm-loss-weight 0.25 \
  --label-loss-weight 1.0 \
  --compile-step \
  --output-dir outputs/transformer/tiny_transformer_decoder_mlx_vocab10k_lm025_label1_full
```

Long-sequence encoder with gradient accumulation:

```bash
python3 -m classifiers.tiny_transformer_mlx.run \
  --tokenizer-path outputs/transformer/tiny_transformer_mlx_vocab10k_mean_warmup_cosine_full/tokenizer.json \
  --vocab-size 10000 \
  --max-length 1024 \
  --d-model 24 \
  --num-heads 4 \
  --ff-dim 48 \
  --num-layers 8 \
  --pooling mean \
  --batch-size 16 \
  --micro-batch-size 4 \
  --eval-batch-size 4 \
  --parameter-dtype bfloat16 \
  --lr-schedule warmup-cosine \
  --output-dir outputs/transformer/tiny_transformer_mlx_vocab10k_mean_warmup_cosine_d24_l8_s1024_bf16_accum_full
```

Profile transformer shape cost:

```bash
python3 -m classifiers.tiny_transformer_mlx.profile
```

Optionally capture a Metal trace for Xcode Instruments:

```bash
MTL_CAPTURE_ENABLED=1 python3 -m classifiers.tiny_transformer_mlx.profile --capture-shape 16x128 --sequence-lengths 128 --batch-sizes 16
```

Tokenized arrays are cached under each run's `encoded_cache` directory, so repeated
training runs with the same tokenizer/split/sequence length avoid re-tokenizing the 50k
reviews. The saved BPE tokenizer lives in `tokenizer.json` next to each run's artifacts.

The first attempted full run used sequence length 256, but a full epoch was too slow for
iteration. The default was changed to 128 tokens because attention cost scales
quadratically with sequence length, while the parameter count stays the same.

## First full run

Command:

```bash
python3 -m classifiers.tiny_transformer_mlx.run
```

Setup:

- Dataset: Kaggle IMDb 50K Movie Reviews
- Train/test split: 80/20 stratified, `random_state=42`
- Validation split: 10% carved out of the training split
- Train rows used after validation split: 36,000
- Validation rows: 4,000
- Test rows: 10,000
- Trainable parameters: 12,978
- Epochs: 8
- Batch size: 512
- Learning rate: 0.001
- Weight decay: 0.01
- Total training time after tokenization/cache: 70.2 seconds
- Best validation epoch: 8

Metrics on the test split:

- Accuracy: 0.6385
- Balanced accuracy: 0.6385
- Precision: 0.6421
- Recall: 0.6260
- F1: 0.6339
- ROC-AUC: 0.6893
- PR-AUC: 0.6672
- MCC: 0.2771
- Log-loss: 0.6379

Confusion matrix:

| Actual \\ Predicted | negative | positive |
| --- | ---: | ---: |
| negative | 3255 | 1745 |
| positive | 1870 | 3130 |

This confirms that the tiny transformer learns real signal from scratch, but its capacity
and 128-token truncation leave it far below the TF-IDF linear baselines. The experiment is
useful as a modern architecture comparison, not as the strongest project model.

## Full 10k BPE experiments

These runs used the full Kaggle IMDb dataset with the same 80/20 train/test split and 10%
validation split from the training set. All used:

- BPE vocabulary: requested 10,000, actual 10,000
- Maximum sequence length: 128
- Transformer layers: 4
- Model width: 64
- Attention heads: 4
- Feed-forward width: 128
- Batch size: 16
- Optimizer: AdamW, weight decay 0.01
- Compiled MLX train step: enabled
- Trainable parameters: 804,546

The first run trained and saved the tokenizer/cache under
`outputs/transformer/tiny_transformer_mlx_vocab10k_cls_constant_full`. The later runs
reused the exact same saved `tokenizer.json` and encoded arrays so the comparison only
changes pooling/schedule behavior.

| Run | Pooling | LR schedule | Best val epoch | Best val accuracy | Test accuracy | F1 | ROC-AUC | PR-AUC | Log-loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tiny_transformer_mlx_vocab10k_cls_constant_full` | `[CLS]` | constant `1e-3` | 3 | 0.8303 | 0.8354 | 0.8320 | 0.9159 | 0.9128 | 0.3921 |
| `tiny_transformer_mlx_vocab10k_mean_constant_full` | masked mean | constant `1e-3` | 3 | 0.8330 | 0.8298 | 0.8233 | 0.9123 | 0.9095 | 0.4024 |
| `tiny_transformer_mlx_vocab10k_mean_warmup_cosine_full` | masked mean | 10% warmup + cosine | 2 | 0.8335 | 0.8389 | 0.8451 | 0.9188 | 0.9163 | 0.3709 |

Validation curves:

| Run | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `[CLS]`, constant LR | 0.8027 | 0.8273 | 0.8303 | 0.8260 | 0.8150 |
| mean, constant LR | 0.8137 | 0.8315 | 0.8330 | 0.8247 | 0.8207 |
| mean, warmup-cosine | 0.8060 | 0.8335 | 0.8287 | 0.8313 | 0.8280 |

The 10k tokenizer fixed the earlier degenerate behavior from the 256-token setup. The
model now learns real lexical sentiment features instead of collapsing to a near-constant
classifier. It still overfits quickly: train loss keeps improving after epoch 2-3 while
validation loss rises. Mean pooling alone did not improve the final test score, but
mean pooling plus warmup-cosine gave the best result in this group.

These results are still far below the TF-IDF linear baselines, especially LinearSVC, but
they are much more credible for a from-scratch transformer comparison than the 256-token
experiment.

### Batch-size ablation

The previous best encoder setup was rerun with `batch_size=32` instead of 16:

- Output directory: `outputs/transformer/tiny_transformer_mlx_vocab10k_mean_warmup_cosine_b32_full`
- Same saved 10k BPE tokenizer and encoded cache as the batch-16 run
- Mean pooling
- Warmup-cosine LR schedule
- Learning rate: 0.001
- Epochs: 5

| Batch size | Steps/epoch | Warmup steps | Best val epoch | Best val accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 2250 | 1125 | 2 | 0.8335 |
| 32 | 1125 | 562 | 2 | 0.8303 |

Batch 32 started slightly better in epoch 1, but peaked lower and then overfit:

| Epoch | Val accuracy |
| ---: | ---: |
| 1 | 0.8173 |
| 2 | 0.8303 |
| 3 | 0.8250 |
| 4 | 0.8250 |
| 5 | 0.8235 |

For this setup, batch 16 remains the best validation result. Batch 32 halves the number
of optimizer updates per epoch, so this is not only a memory/speed change; it also changes
the optimization path and scheduler step count.

### Narrower/deeper ablation

Because the 10k-token embedding matrix dominates the parameter count, a narrower model
cuts parameters aggressively while still allowing more depth. Two 8-layer variants were
run with the same tokenizer/cache, mean pooling, batch size 16, and warmup-cosine schedule:

| Model | Parameters | Best val epoch | Best val accuracy | Notes |
| --- | ---: | ---: | ---: | --- |
| `d_model=64`, 4 layers, `ff_dim=128` | 804,546 | 2 | 0.8335 | Previous best encoder |
| `d_model=24`, 8 layers, `ff_dim=48` | 286,538 | 4 | 0.8335 | Ties best with far fewer parameters |
| `d_model=24`, 8 layers, `ff_dim=96` | 314,186 | 3 | 0.8325 | Slightly more FFN capacity, slightly worse |

Validation accuracy by epoch:

| Model | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `d24_l8_ff48` | 0.8033 | 0.8300 | 0.8315 | 0.8335 | 0.8290 |
| `d24_l8_ff96` | 0.8103 | 0.8277 | 0.8325 | 0.8325 | 0.8317 |

The smaller/deeper shape did not improve the validation ceiling, but it matched the best
result with about 36% of the parameters. That suggests the 804k model is not obviously
necessary for this dataset, but simply shrinking the network does not solve the
generalization limit either.

### Longer-context ablation

The strongest local diagnostic was that `max_length=128` truncates about 89% of reviews.
A longer-context run was trained with `max_length=512`, which fully covers about 85.5% of
the Kaggle IMDb reviews while remaining much cheaper than 1024-token dense attention.

Setup:

- Output directory: `outputs/transformer/tiny_transformer_mlx_vocab10k_mean_warmup_cosine_d24_l4_s512_accum_fp32_full`
- BPE vocabulary: reused saved full 10k tokenizer
- Maximum sequence length: 512
- Transformer layers: 4
- Model width: 24
- Attention heads: 4
- Feed-forward width: 48
- Pooling: masked mean
- Trainable parameters: 263,306
- Effective batch size: 16
- Micro-batch size: 4
- Eval batch size: 4
- Parameter dtype: float32
- LR schedule: 10% warmup + cosine decay

Validation results:

| Epoch | Val accuracy | Val loss |
| ---: | ---: | ---: |
| 1 | 0.8612 | 0.3238 |
| 2 | 0.8868 | 0.2744 |
| 3 | 0.8900 | 0.2746 |
| 4 | 0.8935 | 0.2783 |
| 5 | 0.8930 | 0.2796 |

Best validation accuracy was 0.8935 at epoch 4. Test accuracy from the best-validation
checkpoint was 0.8943, with ROC-AUC 0.9603 and PR-AUC 0.9592.

This result strongly supports the truncation hypothesis. The model is smaller and
shallower than the previous 128-token best, yet validation accuracy jumped from 0.8335 to
0.8935 simply by allowing the model to read much more of each review.

## Causal decoder experiment

This experiment changes the architecture and objective from bidirectional encoder
classification to GPT-style causal decoding. The sequence format is:

```text
review tokens [CLS] label-token
```

At validation/test time the model sees only:

```text
review tokens [CLS]
```

The next-token logits at `[CLS]` are restricted to the two label tokens and softmaxed as
negative/positive probabilities. With the saved 10k BPE tokenizer, the plain words
`positive` and `negative` split into multiple tokens, but the natural leading-space
versions are single tokens:

- ` negative` -> token id 3771
- ` positive` -> token id 2847

The full run used both objectives:

- next-token language-modeling loss over the review/label sequence, weight 0.25
- final label-token loss at `[CLS]`, weight 1.0

Setup:

- BPE vocabulary: reused saved full 10k tokenizer
- Maximum sequence length: 128
- Transformer layers: 4
- Model width: 64
- Attention heads: 4
- Feed-forward width: 128
- Batch size: 16
- Optimizer: AdamW, weight decay 0.01
- LR schedule: 10% warmup + cosine decay
- Compiled MLX train step: enabled
- Trainable parameters: 1,444,416
- Output directory: `outputs/transformer/tiny_transformer_decoder_mlx_vocab10k_lm025_label1_full`

Validation results:

| Epoch | Val accuracy | Val label loss | Val LM loss |
| ---: | ---: | ---: | ---: |
| 1 | 0.8097 | 0.4136 | 5.7853 |
| 2 | 0.8320 | 0.3755 | 5.5184 |
| 3 | 0.8325 | 0.4098 | 5.3751 |
| 4 | 0.8327 | 0.4311 | 5.3073 |
| 5 | 0.8305 | 0.4615 | 5.2981 |

Best validation accuracy was 0.8327 at epoch 4. This is competitive with the encoder
experiments but slightly below the best 10k BPE encoder run, which reached 0.8335 with
mean pooling and warmup-cosine.

The language-modeling loss kept improving while label loss worsened after epoch 2. That
suggests the auxiliary LM objective was still learning text continuation, but after the
early epochs it was no longer aligned with the classification objective.

## Memory notes

The default batch size is now 16 to keep the Mac responsive. A synthetic
`batch_size=16`, `sequence_length=128` profile reached about 48.8 MB peak Metal memory.
For context, `batch_size=128`, `sequence_length=128` reached about 384.5 MB peak Metal
memory. The first full run above used `--batch-size 512`, which was fast but caused about
1.17 GB of peak Metal memory at sequence length 128. A synthetic `batch_size=256`,
`sequence_length=256` profile reached about 1.63 GB peak Metal memory.

Do not run multiple MLX training/profile commands in parallel on this 8 GB unified-memory
Mac. The model weights are tiny, but transformer training creates large temporary tensors,
especially attention score tensors shaped roughly like:

```text
batch_size * num_heads * sequence_length * sequence_length
```

Those tensors are quadratic in sequence length, and gradients/intermediate buffers multiply
the memory beyond the raw attention matrix itself.

For long-sequence experiments, the runner supports gradient accumulation:

- `--batch-size` is the effective optimizer batch size.
- `--micro-batch-size` is the forward/backward size used before accumulating gradients.
- `--eval-batch-size` is the validation/test forward batch size. When it is omitted, it
  defaults to the micro-batch size if gradient accumulation is active.
- Example: `--batch-size 16 --micro-batch-size 4` performs four micro-batches, then one
  AdamW update.

The validation/test batch size matters for memory too. An earlier smoke test appeared to
show that float32 gradient accumulation did not reduce peak memory, but the actual cause
was validation/test evaluation still using a forward batch of 16. After defaulting eval to
the micro-batch size, a full train/validation/test smoke at `sequence_length=1024` dropped
from roughly 5.5 GB to roughly 3.1 GB in float32. Isolating only the training step showed
the expected behavior: direct batch 16 peaked around 3.36 GB, direct batch 4 around 1.06
GB, and accumulated batch 16 via micro-batches of 4 around 1.28 GB.

The runner also supports `--parameter-dtype float32|float16|bfloat16`. In smoke tests at
`sequence_length=1024`, naive `float16` training produced NaNs at `learning_rate=1e-3`.
`bfloat16` completed the same fixed-eval smoke test and reduced peak memory footprint to
roughly 1.6 GB, so `bfloat16` is the safer reduced-precision path on this setup.

## Batch 16 profiling

A real-data run on a 5,000-row subset with `batch_size=16` trained 3,600 rows for one epoch
in 1.4 seconds, excluding tokenizer/cache setup. The same cached run under `cProfile`
finished in about 3.1 seconds total, including Python startup/imports, CSV loading,
training, validation, test evaluation, and artifact writes.

Manual timing of 200 real training steps at `batch_size=16`, `sequence_length=128`:

| Stage | ms/step | Share |
| --- | ---: | ---: |
| Batch slicing + MLX array conversion | 0.020 | 0.4% |
| Loss + gradient construction | 0.423 | 7.6% |
| AdamW optimizer update | 0.366 | 6.6% |
| `mx.eval(...)` GPU synchronization | 4.744 | 85.4% |
| `loss.item()` | 0.001 | 0.0% |

Total: 5.56 ms/step, about 48.8 MB peak Metal memory. The bottleneck at this batch size
is not Python data loading; it is queued MLX/Metal work being synchronized in `mx.eval`.

Deeper profiling options:

- `mlx.profiler` is not available in the installed MLX package.
- MLX exposes Metal capture through `mx.metal.start_capture(...)`; this creates a
  `.gputrace` package for Xcode Instruments.
- The local command-line developer tools do not currently provide `xctrace`, so the trace
  cannot be exported textually from the shell on this machine.
- `profile.py --layer-breakdown` adds a readable MLX-level forward-pass breakdown by
  synchronizing after each model stage. This changes timing behavior, but it is useful for
  finding obvious stage-level bottlenecks.

Layer breakdown at `batch_size=16`, `sequence_length=128`:

| Group | Forced-sync ms | Share |
| --- | ---: | ---: |
| Residual/dropout stages | 2.327 | 32.0% |
| Norm stages | 1.848 | 25.4% |
| Attention stages | 1.639 | 22.5% |
| SwiGLU FFN stages | 1.049 | 14.4% |
| Classifier head | 0.227 | 3.1% |
| Attention mask bias | 0.178 | 2.5% |

No single stage dominates. The profile looks like many small GPU operations plus
synchronization overhead, not one broken layer.

`MLX_METAL_FAST_SYNCH=1` was also tested and did not materially change this run:
`train_step_ms_mean` stayed around 5.8 ms.

Compiling the train step with `--compile-step` helped after warmup:

| Mode | Epoch 1 | Epoch 2 | Epoch 3 |
| --- | ---: | ---: | ---: |
| Eager | 1.52s | 1.37s | 1.45s |
| Compiled | 1.21s | 1.14s | 1.06s |

The comparison used the 5,000-row subset, `batch_size=16`, 3,600 training rows per epoch.
Compilation has overhead, so it is less useful for one-off smoke tests, but it is worth
using for longer training runs.

Additional kernel-path checks:

| Variant | Step time | Notes |
| --- | ---: | --- |
| GPU, additive mask, dropout, eager | 6.30 ms | Current eager-style path |
| GPU, boolean mask, dropout, eager | 5.56 ms | Boolean mask helps eager slightly |
| GPU, additive mask, no dropout, eager | 5.14 ms | Faster, but changes regularization |
| GPU, additive mask, dropout, compiled | 4.21 ms | Best tested training path |
| CPU, additive mask, dropout, eager | 11.07 ms | Slower than GPU |

`nn.MultiHeadAttention` already calls MLX's `mx.fast.scaled_dot_product_attention`, so the
attention kernel itself is already the optimized MLX path. The remaining overhead comes
from the small transformer being split into many tiny operations. `mx.compile` is the
practical way to fuse some of that work without writing custom Metal kernels.
