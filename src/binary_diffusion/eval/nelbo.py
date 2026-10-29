"""Continuous-time NELBO of a checkpoint on fixed evaluation instances, in nats per
scored token (and bits per character for text): the path ELBO on the training
schedule, the reconstruction term at ``t_min`` and the prior KL at ``t_max``. Each
of ``passes`` draws fresh noise over the same batches, weighted by their share of
the scored tokens; ``se`` is the Monte Carlo error of the noise given the instances.
"""
from __future__ import annotations

import json
import math
import os

import flax.nnx as nn
import hydra
import jax
import numpy as np
from jax import random
from omegaconf import DictConfig

from ..checkpoints import load_checkpoint
from ..data.factory import create_eval_batches
from ..data.text import count_chars
from ..loss import make_loss_fn, prior_kl, reconstruction


@hydra.main(config_path="../../../configs/eval", config_name="nelbo", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.passes < 2:
        raise ValueError("the standard error needs passes >= 2")
    model, run_cfg, tokenizer, kernel = load_checkpoint(cfg.checkpoint_dir, cfg.overrides)
    loss_fn = make_loss_fn(run_cfg, tokenizer)

    @nn.jit
    def path_elbo(model, batch, rng):
        _, (metrics, _) = loss_fn(model, batch, rng)
        return metrics["elbo"]

    batches = create_eval_batches(run_cfg, cfg.split, cfg.batch_size, cfg.n,
                                  cfg.subset_seed, cfg.hard)
    scored = [(b["input_ids"] != tokenizer.pad_token_id) & ~b["cond_mask"] for b in batches]
    counts = np.array([s.sum() for s in scored], np.float64)
    key = random.PRNGKey(cfg.seed)
    terms = np.zeros((cfg.passes, len(batches), 2))      # [pass, batch, (path, rec)]
    for p in range(cfg.passes):
        for b, batch in enumerate(batches):
            path_rng, rec_rng = random.split(random.fold_in(random.fold_in(key, p), b))
            device = jax.device_put(batch)
            terms[p, b] = (float(path_elbo(model, device, path_rng)),
                           float(reconstruction(model, device, rec_rng, kernel=kernel,
                                                pad_token_id=tokenizer.pad_token_id)))
        print(f"[nelbo] pass {p + 1}/{cfg.passes}", flush=True)

    w = counts / counts.sum()
    v = terms.sum(-1)
    prior = prior_kl(kernel)
    nelbo = float(v.mean(0) @ w) + prior
    se = math.sqrt(float(w ** 2 @ v.var(0, ddof=1)) / cfg.passes)
    result = {"nelbo": nelbo, "se": se, "n": sum(len(b["input_ids"]) for b in batches)}
    if run_cfg.data.source == "text":
        chars = sum(count_chars(b["input_ids"], tokenizer, s)
                    for b, s in zip(batches, scored, strict=True))
        per_char = counts.sum() / (math.log(2) * chars)
        result.update(bits_per_char=nelbo * per_char, bits_per_char_se=se * per_char)
    with open(os.path.join(cfg.out, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
