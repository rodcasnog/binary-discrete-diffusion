"""Solve rate of a logic checkpoint: every evaluation instance is sampled once from
its clues, with the key ``fold_in(PRNGKey(seed), i)`` for instance ``i`` whatever the
batch size, and checked by the task's verifier."""
from __future__ import annotations

import json
import os
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from omegaconf import DictConfig

from ..checkpoints import load_checkpoint
from ..data.factory import create_eval_batches, task_module
from ..sample import Sampler, sample


@hydra.main(config_path="../../../configs/eval", config_name="solve", version_base="1.3")
def main(cfg: DictConfig) -> None:
    model, run_cfg, _, kernel = load_checkpoint(cfg.checkpoint_dir, cfg.overrides)
    sampler = Sampler(**cfg.sampler)
    task = task_module(run_cfg)
    key = random.PRNGKey(cfg.seed)
    solved, start = [], 0
    for batch in create_eval_batches(run_cfg, cfg.split, cfg.batch_size, cfg.n,
                                     cfg.subset_seed, cfg.hard):
        size = len(batch["input_ids"])
        keys = jax.vmap(partial(random.fold_in, key))(jnp.arange(start, start + size))
        device = jax.device_put(batch)
        # The sampler sees the clues only.
        seq = jnp.where(device["cond_mask"], device["input_ids"], 0)
        pred = sample(model, kernel, sampler, seq, device["cond_mask"], keys,
                      adjacency=device.get("adjacency"), noisy_pad=False)
        solved.append(task.solved(batch, np.asarray(pred)))
        start += size
        print(f"[solve] {start} instances, solve rate {np.concatenate(solved).mean():.4f}",
              flush=True)
    solved = np.concatenate(solved)
    result = {"solve_rate": float(solved.mean()), "n": len(solved)}
    with open(os.path.join(cfg.out, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
