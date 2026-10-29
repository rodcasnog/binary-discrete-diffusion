"""Download ``sapientinc/sudoku-extreme`` and write the arrays data/sudoku.py reads:

  ``{split}_sol.npy``   [N, 81] int8   solution grid, digits 1-9 mapped to 0-8
  ``{split}_clue.npy``  [N, 81] int8   given-cell mask (1 where the puzzle shows a clue)

for ``split`` in train, val and test. Val and test are disjoint subsets of the
upstream test pool, taken in the order of the index files of the split repository.

The source CSVs have columns ``source,question,answer,rating``, where ``question``
is the 81-character puzzle (``.`` for a blank) and ``answer`` the solution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os

import numpy as np
from huggingface_hub import hf_hub_download

REPO = "sapientinc/sudoku-extreme"
REVISION = "58942f96baeb572ca3127e2a9e9c70f330783d6b"
SPLIT_REPO = "rodcasnog/sudoku-extreme-split"
SPLIT_SHA256 = {
    "val": "06414bf101f9d34a93b8656738272579fd02839766930a4448a0f98a2033f44d",
    "test": "dbff47578f4b65ff4dcbf6f7d084922109d8732d73a425fb3562409015ebb762",
}
GRID = 81


def read_puzzles(name: str) -> tuple[np.ndarray, np.ndarray]:
    """``(sol, clue)`` of one upstream CSV."""
    path = hf_hub_download(REPO, f"{name}.csv", repo_type="dataset", revision=REVISION)
    answers, questions = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            answers.append(row["answer"])
            questions.append(row["question"])
    answer = np.frombuffer("".join(answers).encode(), np.uint8).reshape(-1, GRID)
    question = np.frombuffer("".join(questions).encode(), np.uint8).reshape(-1, GRID)
    return (answer - ord("1")).astype(np.int8), (question != ord(".")).astype(np.int8)


def split_index(split: str) -> np.ndarray:
    """Rows of the upstream test pool that form ``split``, checked against their hash."""
    path = hf_hub_download(SPLIT_REPO, f"{split}_index.npy", repo_type="dataset")
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != SPLIT_SHA256[split]:
        raise ValueError(f"{path} has sha256 {digest}, expected {SPLIT_SHA256[split]}")
    return np.load(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory to write the .npy arrays into")
    out = ap.parse_args().out
    os.makedirs(out, exist_ok=True)

    splits = {"train": read_puzzles("train")}
    pool = read_puzzles("test")
    for split in ("val", "test"):
        idx = split_index(split)
        splits[split] = (pool[0][idx], pool[1][idx])
    for split, (sol, clue) in splits.items():
        np.save(os.path.join(out, f"{split}_sol.npy"), sol)
        np.save(os.path.join(out, f"{split}_clue.npy"), clue)
        print(f"{split}: {len(sol)} puzzles -> {out}", flush=True)


if __name__ == "__main__":
    main()
