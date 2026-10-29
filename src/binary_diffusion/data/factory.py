"""Task registry and shared batch helpers. ``data.source`` names the task module,
which provides ``tokenizer(cfg)``, the endless ``batches(cfg, split, batch_size,
start)``, the evaluation ``instances(cfg, split, hard)`` and, for the logic tasks,
``solved(batch, pred)``. Batches are ``{"input_ids": int32 [B, L], "cond_mask": bool
[B, L]}`` with the clean anchors in ``cond_mask``, plus ``"adjacency": bool [B, n, n]``
for graph coloring.
"""
from __future__ import annotations

import importlib
from collections.abc import Iterator
from itertools import count
from types import ModuleType
from typing import Any

import numpy as np
from omegaconf import DictConfig

# Offset that gives each split its own sampling stream off the seed.
SPLIT_STREAM = {"train": 1, "val": 2, "test": 3}

# Pad id of the tasks without padding: it never occurs, so every unanchored
# position is scored.
NO_PAD = -1


class StaticTokenizer:
    """The tokenizer attributes the model and the losses read, for a fixed
    vocabulary: ``num_classes`` content ids first, the task's specials after."""

    def __init__(self, vocab_size: int, mask_token_id: int, num_classes: int,
                 pad_token_id: int):
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.num_classes = num_classes
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return self.vocab_size


def epoch_batches(n: int, batch_size: int, seed: int, start: int
                  ) -> Iterator[tuple[np.ndarray, np.random.Generator]]:
    """Row indices and the batch's own generator, in reshuffled epochs forever.

    Each epoch walks a fresh permutation, so every instance is drawn once per
    pass over the pool before any is drawn again; a batch repeats an instance
    only when ``batch_size > n``. Permutation and per-batch randomness are both
    seeded from the epoch and batch number, so batch ``k`` is the same batch
    whatever ran before it and ``start`` seeks to where an interrupted run stopped.
    """
    pos = start * batch_size
    epoch, perm = -1, None
    for k in count(start):
        idx = np.empty(batch_size, dtype=np.int64)
        filled = 0
        while filled < batch_size:
            e, offset = divmod(pos, n)
            if e != epoch:
                epoch, perm = e, np.random.default_rng([seed, e]).permutation(n)
            take = min(batch_size - filled, n - offset)
            idx[filled:filled + take] = perm[offset:offset + take]
            filled += take
            pos += take
        yield idx, np.random.default_rng([seed, epoch, k])


def task_module(cfg: DictConfig) -> ModuleType:
    """The data module of ``data.source``, imported on first use so that a run loads
    only its own task's dependencies."""
    return importlib.import_module(f"{__package__}.{cfg.data.source}")


def create_tokenizer(cfg: DictConfig) -> Any:
    return task_module(cfg).tokenizer(cfg)


def create_eval_batches(cfg: DictConfig, split: str, batch_size: int, n: int | None,
                        seed: int, hard: bool) -> list[dict]:
    """Fixed, unaugmented evaluation instances in batches of ``batch_size``, the
    last one possibly smaller.

    ``n=None`` takes the whole split in stored order, otherwise the first ``n`` of
    a ``seed``-ed permutation, so a smaller set is a prefix of a larger one. Graph
    coloring generates its instances and takes the first ``n`` of the split's
    stream. ``hard`` selects the task's hard subset (``HARD_*`` in its module).
    """
    task = task_module(cfg)
    if cfg.data.source == "graphcolor":
        arrays = task.instances(cfg, split, hard, n)
    else:
        arrays = task.instances(cfg, split, hard)
        if n is not None:
            total = len(arrays["input_ids"])
            if n > total:
                raise ValueError(f"asked for {n} instances, but {split} has {total}")
            idx = np.random.default_rng(seed).permutation(total)[:n]
            arrays = {key: value[idx] for key, value in arrays.items()}
    total = len(arrays["input_ids"])
    return [{key: value[i:i + batch_size] for key, value in arrays.items()}
            for i in range(0, total, batch_size)]
