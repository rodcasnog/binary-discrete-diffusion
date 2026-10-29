"""Boxoban (Sokoban on 10x10 boards): a board and an A* solution as one sequence
``[board (100) | actions (128)]``, with the board cells as clean anchors. The
action region holds the moves, then STAY filler, and is corrupted and scored end
to end, so the model also learns where a solution ends; a prediction is read up
to its first STAY.

Vocabulary (13 ids)::

    0..3   moves up / right / down / left     the content alphabet
    4      STAY                               (N_CONTENT = 5)
    5..11  board symbols # .$@*+ (wall, floor, goal, box, player, box on goal,
           player on goal), anchors only
    12     MASK

Arrays (``scripts/prep_boxoban.py``): ``{split}_seq.npy`` int8 [N, 228].
"""
from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np
from omegaconf import DictConfig

from .factory import NO_PAD, SPLIT_STREAM, StaticTokenizer, epoch_batches

GRID_H = GRID_W = 10
BOARD = GRID_H * GRID_W
ACTIONS = 128          # up to 127 moves, then at least one STAY
N_MOVES = 4
STAY_ID = 4
N_CONTENT = N_MOVES + 1
BOARD_CHARS = "# .$@*+"  # order fixes ids 5..11
CHAR_TO_ID = {c: N_CONTENT + i for i, c in enumerate(BOARD_CHARS)}
VOCAB_SIZE = N_CONTENT + len(BOARD_CHARS) + 1
MASK_ID = VOCAB_SIZE - 1
HARD_MIN_MOVES = 40    # the hard subset: solutions of at least this many moves

# move id -> (d_row, d_col); row 0 is the top of the board, so up decreases row.
MOVE_DELTAS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int32)


def tokenizer(cfg: DictConfig) -> StaticTokenizer:
    return StaticTokenizer(VOCAB_SIZE, MASK_ID, N_CONTENT, NO_PAD)


# Move-id remap per D4 element e (e % 4 counter-clockwise rotations, after a left-right
# flip for e >= 4), indexed by any action id: the moves, then STAY.
_PERMS = np.array([[0, 1, 2, 3, 4], [3, 0, 1, 2, 4], [2, 3, 0, 1, 4], [1, 2, 3, 0, 4],
                   [0, 3, 2, 1, 4], [3, 2, 1, 0, 4], [2, 1, 0, 3, 4], [1, 0, 3, 2, 4]], np.int8)


def augment_d4(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply an independent random rotation or reflection to each row's board and
    remap its moves to match, so a solution stays a solution of the same length:
    the board is square and has no directional symbol."""
    seq = np.array(seq)
    elems = rng.integers(0, 8, size=seq.shape[0])
    boards = seq[:, :BOARD].reshape(-1, GRID_H, GRID_W)
    moves = seq[:, BOARD:]
    for e in range(8):
        m = elems == e
        b = boards[m][:, :, ::-1] if e >= 4 else boards[m]
        boards[m] = np.rot90(b, e % 4, axes=(1, 2))
        moves[m] = _PERMS[e][moves[m]]
    seq[:, :BOARD] = boards.reshape(-1, BOARD)
    seq[:, BOARD:] = moves
    return seq


def solves(seq_row: np.ndarray, moves: np.ndarray) -> bool:
    """Whether ``moves`` (ids 0..3) solve the board prefix of ``seq_row``: every
    move is legal (it neither leaves the board nor walks into a wall, and pushes a
    box only onto a free cell) and every box ends on a goal."""
    g = np.asarray(seq_row[:BOARD]).reshape(GRID_H, GRID_W)
    walls = g == CHAR_TO_ID["#"]
    targets = (g == CHAR_TO_ID["."]) | (g == CHAR_TO_ID["*"]) | (g == CHAR_TO_ID["+"])
    boxes = (g == CHAR_TO_ID["$"]) | (g == CHAR_TO_ID["*"])
    pr, pc = (int(v) for v in np.argwhere((g == CHAR_TO_ID["@"]) | (g == CHAR_TO_ID["+"]))[0])
    for m in moves:
        dr, dc = MOVE_DELTAS[int(m)]
        nr, nc = pr + dr, pc + dc
        if not (0 <= nr < GRID_H and 0 <= nc < GRID_W) or walls[nr, nc]:
            return False
        if boxes[nr, nc]:
            br, bc = nr + dr, nc + dc
            if not (0 <= br < GRID_H and 0 <= bc < GRID_W) or walls[br, bc] or boxes[br, bc]:
                return False
            boxes[nr, nc] = False
            boxes[br, bc] = True
        pr, pc = nr, nc
    return bool(np.all(boxes == (boxes & targets)) and boxes.sum() == targets.sum())


def load_split(data_dir: str, split: str) -> np.ndarray:
    return np.load(os.path.join(data_dir, f"{split}_seq.npy"), mmap_mode="r")


def _batch(seq: np.ndarray) -> dict:
    cond = np.zeros(seq.shape, dtype=bool)
    cond[:, :BOARD] = True
    return {"input_ids": seq, "cond_mask": cond}


def batches(cfg: DictConfig, split: str, batch_size: int, start: int) -> Iterator[dict]:
    seq = load_split(cfg.data.dataset_path, split)
    stream = cfg.training.seed + SPLIT_STREAM[split]
    for idx, rng in epoch_batches(len(seq), batch_size, stream, start):
        batch = np.asarray(seq[idx], dtype=np.int32)
        if split == "train":
            batch = augment_d4(batch, rng)
        yield _batch(batch)


def instances(cfg: DictConfig, split: str, hard: bool) -> dict[str, np.ndarray]:
    seq = np.asarray(load_split(cfg.data.dataset_path, split), dtype=np.int32)
    if hard:
        seq = seq[(seq[:, BOARD:] < N_MOVES).sum(1) >= HARD_MIN_MOVES]
    return _batch(seq)


def solved(batch: dict, pred: np.ndarray) -> np.ndarray:
    """Whether each prediction's moves, read up to the first STAY, solve its board."""
    actions = np.asarray(pred)[:, BOARD:]
    stay = actions == STAY_ID
    ends = np.where(stay.any(1), stay.argmax(1), ACTIONS)
    return np.array([solves(row, a[:end]) for row, a, end in
                     zip(np.asarray(batch["input_ids"]), actions, ends, strict=True)])
