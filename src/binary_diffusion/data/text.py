"""Text corpora: uint16 records of ``data.max_seq_len`` tokens, ``{split}.npy``
[N, L] (``scripts/prep_{text8,tinystories,owt}.py``). Packed corpora (Text8,
OpenWebText) cut a token stream into records; padded ones (``data.padded``,
TinyStories) hold one document, its EOS and PAD up to L. Nothing is anchored
(``cond_mask`` is all False); the loss skips PAD positions.
"""
from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np
from omegaconf import DictConfig
from transformers import AutoTokenizer

from .factory import SPLIT_STREAM, epoch_batches


def tokenizer(cfg: DictConfig):
    """The corpus tokenizer with the diffusion mask added after it. A trained PAD
    that sits last is taken out of the content alphabet; a tokenizer without PAD
    gets a sentinel id."""
    tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name)
    num_classes = len(tok)
    if tok.pad_token_id == num_classes - 1:
        num_classes -= 1
    tok.add_special_tokens({"mask_token": f"<|diffusion_mask_{num_classes}|>"})
    if tok.pad_token_id is None:
        # Text8 defines no PAD and its records carry no padding; the loss still
        # needs an id, so add a never-occurring sentinel.
        tok.add_special_tokens({"pad_token": f"<|pad_{num_classes + 1}|>"})
    tok.num_classes = num_classes
    return tok


def load_split(data_dir: str, split: str) -> np.ndarray:
    return np.load(os.path.join(data_dir, f"{split}.npy"), mmap_mode="r")


def _batch(rows: np.ndarray) -> dict:
    ids = np.asarray(rows, dtype=np.int32)
    return {"input_ids": ids, "cond_mask": np.zeros(ids.shape, dtype=bool)}


def batches(cfg: DictConfig, split: str, batch_size: int, start: int) -> Iterator[dict]:
    rows = load_split(cfg.data.dataset_path, split)
    stream = cfg.training.seed + SPLIT_STREAM[split]
    for idx, _ in epoch_batches(len(rows), batch_size, stream, start):
        yield _batch(rows[idx])


def instances(cfg: DictConfig, split: str, hard: bool) -> dict[str, np.ndarray]:
    if hard:
        raise ValueError("text corpora have no hard subset")
    return _batch(load_split(cfg.data.dataset_path, split))


def decode(tokens: np.ndarray, tokenizer, padded: bool) -> list[str]:
    """Decode each row without its special tokens; a padded record ends at its
    first EOS, and a row without one is decoded whole."""
    texts = []
    for row in np.asarray(tokens):
        ids = row.tolist()
        if padded:
            hits = np.flatnonzero(row == tokenizer.eos_token_id)
            if len(hits):
                ids = ids[:int(hits[0])]
        texts.append(tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    return texts


def count_chars(tokens: np.ndarray, tokenizer, scored: np.ndarray) -> int:
    """Characters the scored tokens decode to. EOS and BOS are scored but are not
    text, so they decode to none."""
    return sum(len(tokenizer.decode(row[keep].tolist(), skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False))
               for row, keep in zip(np.asarray(tokens), np.asarray(scored), strict=True))


def pad_anchors(cfg: DictConfig, tokenizer, n: int, seed: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """The PAD tails of ``n`` validation records drawn with replacement, so samples
    follow the corpus' length distribution. Returns ``(seq, cond)``: PAD at the
    anchors (``cond``) and zeros elsewhere."""
    rows = load_split(cfg.data.dataset_path, "val")
    ids = np.random.default_rng(seed).integers(len(rows), size=n)
    cond = np.asarray(rows[ids]) == tokenizer.pad_token_id
    return np.where(cond, tokenizer.pad_token_id, 0).astype(np.int32), cond


def reference_texts(cfg: DictConfig, tokenizer, split: str, n: int, seed: int) -> list[str]:
    """``n`` distinct records of ``split``, uniformly drawn and decoded like samples."""
    rows = load_split(cfg.data.dataset_path, split)
    ids = np.random.default_rng(seed).choice(len(rows), n, replace=False)
    return decode(rows[ids], tokenizer, cfg.data.padded)
