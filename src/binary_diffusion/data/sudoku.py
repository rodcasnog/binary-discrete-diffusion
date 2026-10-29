"""Sudoku: the 81 cells of the solution grid over the digits 1-9 (ids 0-8), with
the puzzle's given cells as clean anchors. Only the blank cells are corrupted and
scored; for the bit kernels a blank cell's active set is its set of candidate
digits.

Arrays (``scripts/prep_sudoku.py``): ``{split}_sol.npy`` and ``{split}_clue.npy``,
int8 [N, 81], the clue mask 1 at a given cell.
"""
from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np
from omegaconf import DictConfig

from .factory import NO_PAD, SPLIT_STREAM, StaticTokenizer, epoch_batches

N_DIGITS = 9
MASK_ID = N_DIGITS
HARD_MAX_CLUES = 22    # the hard subset: puzzles with at most this many givens


def tokenizer(cfg: DictConfig) -> StaticTokenizer:
    return StaticTokenizer(N_DIGITS + 1, MASK_ID, N_DIGITS, NO_PAD)


def load_split(data_dir: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    return tuple(np.load(os.path.join(data_dir, f"{split}_{name}.npy"), mmap_mode="r")
                 for name in ("sol", "clue"))


def augment_sudoku(sol: np.ndarray, clue: np.ndarray, rng: np.random.Generator
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Validity-preserving augmentation, drawn independently per sample: digit
    relabeling, row-band and within-band row permutations, column-stack and
    within-stack column permutations, and transposition. ``sol`` and ``clue`` are
    [B, 81]; positions permute jointly, the relabeling touches values only."""
    B = sol.shape[0]
    sol9 = sol.reshape(B, 9, 9)
    clue9 = clue.reshape(B, 9, 9)
    bands = np.argsort(rng.random((B, 3)), axis=1)
    rows = (3 * bands[:, :, None] + np.argsort(rng.random((B, 3, 3)), axis=2)
            ).reshape(B, 9)
    stacks = np.argsort(rng.random((B, 3)), axis=1)
    cols = (3 * stacks[:, :, None] + np.argsort(rng.random((B, 3, 3)), axis=2)
            ).reshape(B, 9)
    b = np.arange(B)[:, None, None]
    s9 = sol9[b, rows[:, :, None], cols[:, None, :]]
    c9 = clue9[b, rows[:, :, None], cols[:, None, :]]
    t = (rng.integers(2, size=B) == 1)[:, None, None]
    s9 = np.where(t, s9.transpose(0, 2, 1), s9)
    c9 = np.where(t, c9.transpose(0, 2, 1), c9)
    relabel = np.argsort(rng.random((B, 9)), axis=1).astype(sol.dtype)
    s_out = np.take_along_axis(relabel, s9.reshape(B, 81), axis=1)
    return s_out, c9.reshape(B, 81)


def batches(cfg: DictConfig, split: str, batch_size: int, start: int) -> Iterator[dict]:
    sol, clue = load_split(cfg.data.dataset_path, split)
    stream = cfg.training.seed + SPLIT_STREAM[split]
    for idx, rng in epoch_batches(len(sol), batch_size, stream, start):
        s_b, c_b = sol[idx], clue[idx]
        if split == "train":
            s_b, c_b = augment_sudoku(s_b, c_b, rng)
        yield {"input_ids": np.asarray(s_b, dtype=np.int32),
               "cond_mask": np.asarray(c_b, dtype=bool)}


def instances(cfg: DictConfig, split: str, hard: bool) -> dict[str, np.ndarray]:
    sol, clue = load_split(cfg.data.dataset_path, split)
    keep = clue.sum(1) <= HARD_MAX_CLUES if hard else slice(None)
    return {"input_ids": np.asarray(sol[keep], dtype=np.int32),
            "cond_mask": np.asarray(clue[keep], dtype=bool)}


def solved(batch: dict, pred: np.ndarray) -> np.ndarray:
    """Whether each prediction completes the reference grid."""
    return (np.asarray(pred) == np.asarray(batch["input_ids"])).all(1)
