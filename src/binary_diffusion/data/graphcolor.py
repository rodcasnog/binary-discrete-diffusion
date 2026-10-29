"""Graph k-coloring: color graphs with a planted balanced k-coloring, their n
vertices split into k equal groups and each cross-group edge present w.p. ``p``.
The sequence is the n vertex colors, none of them anchored, and the adjacency
enters the model as its attention structure (``adjacency``). Many proper
colorings exist besides the planted one, so a solution is any coloring without a
monochromatic edge. Instances are generated, one stream per split off
``data.graphcolor.seed``. Vocabulary: colors ``0..k-1``, MASK ``k``.
"""
from __future__ import annotations

from collections.abc import Iterator
from itertools import count, islice

import numpy as np
from omegaconf import DictConfig

from .factory import NO_PAD, SPLIT_STREAM, StaticTokenizer

# Evaluation instances are generated in chunks of this many, so the instances do
# not depend on the evaluation batch size and a smaller set is a prefix of a
# larger one.
CHUNK = 1024
# The hard set: ten times the vertices at the same mean degree. The model keeps
# its own n; GraphDiT clamps the degrees past it.
HARD_N, HARD_P = 640, 0.02


def tokenizer(cfg: DictConfig) -> StaticTokenizer:
    k = cfg.data.graphcolor.k
    return StaticTokenizer(k + 1, k, k, NO_PAD)


def generate_instances(
    rng: np.random.Generator, batch: int, n: int, k: int, p: float
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``batch`` planted instances. Returns (colors [B,n] int, edges [B,E] bool)
    with E = n(n-1)/2 in ``np.triu_indices(n, 1)`` order. Requires k | n."""
    base = np.repeat(np.arange(k), n // k)
    perm = np.argsort(rng.random((batch, n)), axis=1)
    colors = base[perm]                                            # [B, n]
    iu, ju = np.triu_indices(n, 1)
    cross = colors[:, iu] != colors[:, ju]                         # [B, E]
    edges = cross & (rng.random((batch, iu.size)) < p)
    return colors, edges


def _stream(n: int, k: int, p: float, stream: int, batch_size: int, start: int
            ) -> Iterator[dict]:
    iu, ju = np.triu_indices(n, 1)
    for b in count(start):
        colors, edges = generate_instances(
            np.random.default_rng([stream, b]), batch_size, n, k, p)
        adj = np.zeros((batch_size, n, n), dtype=bool)
        adj[:, iu, ju] = edges
        adj[:, ju, iu] = edges
        yield {"input_ids": colors.astype(np.int32),
               "cond_mask": np.zeros(colors.shape, dtype=bool),
               "adjacency": adj}


def batches(cfg: DictConfig, split: str, batch_size: int, start: int) -> Iterator[dict]:
    g = cfg.data.graphcolor
    return _stream(g.n, g.k, g.p, g.seed + SPLIT_STREAM[split], batch_size, start)


def instances(cfg: DictConfig, split: str, hard: bool, n: int | None
              ) -> dict[str, np.ndarray]:
    """The first ``n`` instances of the split's stream."""
    if n is None:
        raise ValueError("graph coloring generates its instances: give their number n")
    g = cfg.data.graphcolor
    n_vertices, p = (HARD_N, HARD_P) if hard else (g.n, g.p)
    chunks = list(islice(_stream(n_vertices, g.k, p, g.seed + SPLIT_STREAM[split], CHUNK, 0),
                         -(-n // CHUNK)))
    return {key: np.concatenate([c[key] for c in chunks])[:n] for key in chunks[0]}


def solved(batch: dict, pred: np.ndarray) -> np.ndarray:
    """Whether each prediction colors its graph without a monochromatic edge."""
    pred = np.asarray(pred)
    same = pred[:, :, None] == pred[:, None, :]
    return ~(np.asarray(batch["adjacency"]) & same).any((1, 2))
