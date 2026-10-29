"""Build the Boxoban arrays data/boxoban.py reads, ``{train,val,test}_seq.npy`` (int8
[N, 228]: a board, its A* solution's moves and STAY filler), from the unfiltered
boxoban-levels boards joined on (file, level) with the boxoban-astar-solutions moves.

A level is kept when its solution is a move string of at most 127 moves that solves
its board (upstream marks failed searches, and some solutions do not solve their
board). The official valid split is carved with seed 0 into test (25k levels), val
(25k) and the rest, which joins the official train split; its one board that also
appears in the official test split is dropped.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import tarfile
import urllib.request

import numpy as np
from huggingface_hub import hf_hub_download

from binary_diffusion.data.boxoban import ACTIONS, BOARD, CHAR_TO_ID, STAY_ID, solves

LEVELS_COMMIT = "78011d8c86ad423bfb2e1005012393b4c3c24409"
LEVELS_URL = f"https://github.com/google-deepmind/boxoban-levels/archive/{LEVELS_COMMIT}.tar.gz"
SOLUTIONS_REPO = "AlignmentResearch/boxoban-astar-solutions"
SOLUTIONS_REVISION = "4b30b9c3eda5e937dc16b9f6e58c8a27952aa6cd"
SPLITS = ("train", "valid", "test")
CARVE = 25_000


def parse_levels(text: str) -> dict[int, np.ndarray]:
    """``{level: board}`` of one levels file, where ``; <level>`` opens a board."""
    levels, rows = {}, []
    for line in text.splitlines():
        if line.startswith(";"):
            levels[int(line[1:])] = rows = []
        elif line:
            rows.append(line)
    return {level: np.array([[CHAR_TO_ID[c] for c in row] for row in rows], dtype=np.int8)
            for level, rows in levels.items()}


def read_boards() -> dict[str, dict[tuple[int, int], np.ndarray]]:
    """Every unfiltered board, keyed by split and then by (file, level)."""
    boards = {split: {} for split in SPLITS}
    with urllib.request.urlopen(LEVELS_URL) as response, \
            tarfile.open(fileobj=response, mode="r|gz") as tar:
        for member in tar:
            # boxoban-levels-<commit>/unfiltered/<split>/<file>.txt
            parts = member.name.split("/")
            if (len(parts) == 4 and parts[1] == "unfiltered" and parts[2] in boards
                    and parts[3].endswith(".txt")):
                levels = parse_levels(tar.extractfile(member).read().decode())
                file_idx = int(parts[3][:-4])
                boards[parts[2]].update(
                    {(file_idx, level): grid for level, grid in levels.items()})
    return boards


def read_split(split: str, boards: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    """The kept sequences of one upstream split, in the order of its solutions file."""
    path = hf_hub_download(SOLUTIONS_REPO, f"unfiltered_{split}.csv.gz",
                           repo_type="dataset", revision=SOLUTIONS_REVISION)
    seqs = []
    with gzip.open(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            actions = row["Actions"] or ""
            if not re.fullmatch("[0-3]+", actions):
                continue
            grid = boards.get((int(row["File"]), int(row["Level"])))
            if grid is None:
                continue
            moves = np.frombuffer(actions.encode(), dtype=np.uint8) - ord("0")
            if len(moves) >= ACTIONS:       # every sequence ends with a STAY
                continue
            seq = np.full(BOARD + ACTIONS, STAY_ID, dtype=np.int8)
            seq[:BOARD] = grid.reshape(-1)
            seq[BOARD:BOARD + len(moves)] = moves
            if solves(seq, moves):
                seqs.append(seq)
    print(f"{split}: kept {len(seqs)} levels", flush=True)
    return np.stack(seqs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory to write the .npy arrays into")
    out = ap.parse_args().out
    os.makedirs(out, exist_ok=True)

    boards = read_boards()
    train, valid, test = (read_split(split, boards[split]) for split in SPLITS)
    official = {row[:BOARD].tobytes() for row in test}
    order = np.random.default_rng(0).permutation(len(valid))
    cand = order[[valid[i, :BOARD].tobytes() not in official for i in order]]
    splits = {
        "train": np.concatenate([train, valid[np.sort(cand[2 * CARVE:])]]),
        "val": valid[np.sort(cand[CARVE:2 * CARVE])],
        "test": valid[np.sort(cand[:CARVE])],
    }
    for split, seq in splits.items():
        np.save(os.path.join(out, f"{split}_seq.npy"), seq)
        print(f"{split}: {len(seq)} sequences -> {out}", flush=True)


if __name__ == "__main__":
    main()
