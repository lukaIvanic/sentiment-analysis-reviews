"""BPE tokenizer training and fixed-length encoding helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm


PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]


def train_bpe_tokenizer(
    texts: Iterable[str],
    *,
    tokenizer_path: Path,
    vocab_size: int,
    min_frequency: int,
    lowercase: bool,
    length: int | None = None,
) -> Tokenizer:
    """Train and save a compact BPE tokenizer on the training split only."""

    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    if lowercase:
        tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
    else:
        tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=length)

    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    """Load a saved tokenizer."""

    return Tokenizer.from_file(str(tokenizer_path))


def special_token_ids(tokenizer: Tokenizer, *, require_mask: bool = False) -> dict[str, int]:
    """Return the ids for the special tokens used by the transformer."""

    ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    required_tokens = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN]
    if require_mask:
        required_tokens.append(MASK_TOKEN)
    missing = [token for token in required_tokens if ids[token] is None]
    if missing:
        raise ValueError(f"Tokenizer is missing special tokens: {missing}")
    return {token: int(token_id) for token, token_id in ids.items() if token_id is not None}


def encode_texts(
    tokenizer: Tokenizer,
    texts: Iterable[str],
    *,
    max_length: int,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode texts as padded token id and attention mask arrays."""

    ids = special_token_ids(tokenizer)
    pad_id = ids[PAD_TOKEN]
    cls_id = ids[CLS_TOKEN]
    sep_id = ids[SEP_TOKEN]

    text_list = list(texts)
    input_ids = np.full((len(text_list), max_length), pad_id, dtype=np.uint16)
    attention_mask = np.zeros((len(text_list), max_length), dtype=np.uint8)
    iterator = tqdm(text_list, desc="Encoding reviews", disable=not show_progress)

    for row_index, text in enumerate(iterator):
        token_ids = tokenizer.encode(str(text)).ids
        token_ids = [cls_id] + token_ids[: max_length - 2] + [sep_id]
        length = len(token_ids)
        input_ids[row_index, :length] = token_ids
        attention_mask[row_index, :length] = 1

    return input_ids, attention_mask
