"""Training entry point: AdamW after a linear warmup, an EMA of the parameters, the
adaptive time sampler, a validation NELBO and Orbax checkpoints, data parallel over
the local GPUs."""
from __future__ import annotations

import json
import os
from functools import partial
from itertools import islice

import flax.nnx as nn
import flax.struct
import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from hydra.core.hydra_config import HydraConfig
from jax import random
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from omegaconf import DictConfig, OmegaConf

from .checkpoints import restore
from .data.factory import task_module
from .kernels import Kernel
from .loss import make_loss_fn, prior_kl, reconstruction
from .models import build_model
from .schedule import AdaptiveClockState, bin_probs, clock_init, clock_update


@flax.struct.dataclass
class TrainState:
    step: jnp.ndarray
    base_rng: jnp.ndarray
    clock: AdaptiveClockState


def decay_mask(params):
    """Weight decay on matrices only: embeddings and every 1-D parameter (biases,
    norm scales) are left out."""
    def mask(path, p):
        name = "/".join(str(getattr(k, "key", getattr(k, "name", k))) for k in path).lower()
        return getattr(p, "ndim", 0) >= 2 and "embed" not in name
    return jax.tree_util.tree_map_with_path(mask, params)


def get_ckpt_state(model: nn.Module, optimizer: nn.Optimizer, state: TrainState) -> dict:
    """Everything a resume needs."""
    return {
        "params": nn.state(model, nn.Param),
        "opt_state": nn.state(optimizer),
        "step": state.step,
        "base_rng": state.base_rng,
        "clock": {"moments": state.clock.moments, "updates": state.clock.updates},
    }


@hydra.main(config_path="../../configs", config_name=None, version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_dir = HydraConfig.get().runtime.output_dir
    t = cfg.training
    devices = jax.device_count()
    if t.global_batch_size % devices:
        raise ValueError(f"training.global_batch_size={t.global_batch_size} must be "
                         f"divisible by the {devices} devices")
    mesh = Mesh(mesh_utils.create_device_mesh((devices,)), ("data",))
    shard = partial(jax.device_put, device=NamedSharding(mesh, PartitionSpec("data")))
    # Model, optimizer and train state are replicated over the mesh: a restore
    # places each array like the one it replaces, and a step cannot mix arrays
    # committed to one device with batches sharded over all of them.
    replicate = partial(jax.device_put, device=NamedSharding(mesh, PartitionSpec()))

    task = task_module(cfg)
    tokenizer = task.tokenizer(cfg)
    kernel = Kernel.from_config(cfg, tokenizer)
    loss_fn = make_loss_fn(cfg, tokenizer)
    prior = prior_kl(kernel)
    lr = float(t.learning_rate)
    schedule = optax.join_schedules(
        [optax.linear_schedule(0.0, lr, t.warmup_steps), optax.constant_schedule(lr)],
        [t.warmup_steps])
    ema_decay = float(t.ema_decay)

    init_rng, base_rng = random.split(random.PRNGKey(t.seed))
    model = build_model(cfg, tokenizer, nn.Rngs(init_rng))
    nn.update(model, replicate(nn.state(model)))
    tx = optax.chain(optax.clip_by_global_norm(t.max_grad_norm),
                     optax.adamw(schedule, b1=t.adam_b1, b2=t.adam_b2,
                                 weight_decay=t.weight_decay, mask=decay_mask))
    optimizer = nn.Optimizer(model, tx, wrt=nn.Param)
    nn.update(optimizer, replicate(nn.state(optimizer)))
    state = replicate(TrainState(step=jnp.zeros((), dtype=jnp.int32), base_rng=base_rng,
                                 clock=clock_init()))
    ema_params = jax.tree.map(jnp.copy, nn.state(model, nn.Param))

    @partial(nn.jit, donate_argnums=(0,))
    def ema_update(ema_params, params):
        return optax.incremental_update(params, ema_params, 1.0 - ema_decay)

    @nn.jit
    def train_step(model, optimizer, state, batch):
        rng = random.fold_in(state.base_rng, state.step)
        probs = bin_probs(state.clock)
        probe = jnp.zeros((batch["input_ids"].shape[0],), jnp.float32)

        def objective(model, probe):
            return loss_fn(model, batch, rng, bin_probs=probs, probe=probe)

        (_, (metrics, time_sample)), (grads, probe_grad) = nn.value_and_grad(
            objective, argnums=(0, 1), has_aux=True)(model, probe)
        grad_norm = optax.global_norm(grads)
        optimizer.update(model, grads)
        new_state = TrainState(step=state.step + 1, base_rng=state.base_rng,
                               clock=clock_update(state.clock, time_sample, probe_grad))
        return new_state, {**metrics, "grad_norm": grad_norm}

    @nn.jit
    def eval_step(model, batch, rng):
        _, (metrics, _) = loss_fn(model, batch, rng)
        return metrics

    val_batches = [shard(b) for b in islice(task.batches(cfg, "val", t.global_batch_size, 0),
                                            t.val_batches)]
    val_key = random.PRNGKey(t.seed + 1)

    def validate(step: int) -> dict:
        """The path terms averaged over the validation batches, the reconstruction
        term on the first one, and the NELBO; the time schedule is the base one."""
        val_rng, rec_rng = random.split(random.fold_in(val_key, step))
        total = None
        for i, batch in enumerate(val_batches):
            metrics = eval_step(model, batch, random.fold_in(val_rng, i))
            total = metrics if total is None else {k: total[k] + v for k, v in metrics.items()}
        result = {k: float(v) / len(val_batches) for k, v in jax.device_get(total).items()}
        result["rec"] = float(reconstruction(model, val_batches[0], rec_rng, kernel=kernel,
                                             pad_token_id=tokenizer.pad_token_id))
        result["prior_kl"] = prior
        result["nelbo"] = result["elbo"] + result["rec"] + prior
        return result

    c = cfg.checkpoint
    ckpt = ocp.CheckpointManager(
        os.path.join(run_dir, "checkpoints"),
        options=ocp.CheckpointManagerOptions(max_to_keep=c.max_to_keep, create=True))
    ema_ckpt = ocp.CheckpointManager(
        os.path.join(run_dir, "checkpoints_ema"),
        options=ocp.CheckpointManagerOptions(max_to_keep=c.ema_max_to_keep,
                                             keep_period=c.keep_period, create=True))

    # Resume from the newest step both managers hold: a run stopped between the
    # two saves of a step resumes from the step before.
    shared = set(ckpt.all_steps()) & set(ema_ckpt.all_steps())
    if ckpt.all_steps() and not shared:
        raise FileNotFoundError(f"{run_dir} holds raw checkpoints but no step with "
                                "both a raw and an EMA checkpoint")
    if shared:
        resume_step = max(shared)
        print(f"resuming from step {resume_step}", flush=True)
        restored = restore(ckpt, resume_step, get_ckpt_state(model, optimizer, state))
        nn.update(model, restored["params"])
        nn.update(optimizer, restored["opt_state"])
        state = TrainState(step=restored["step"], base_rng=restored["base_rng"],
                           clock=AdaptiveClockState(**restored["clock"]))
        ema_params = restore(ema_ckpt, resume_step, {"params": ema_params})["params"]

    def save(step: int) -> None:
        # Orbax skips a step it already holds.
        for manager, item in ((ckpt, get_ckpt_state(model, optimizer, state)),
                              (ema_ckpt, {"params": ema_params})):
            manager.save(step, args=ocp.args.Composite(state=ocp.args.PyTreeSave(item)))

    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    def log(metrics: dict, step: int, split: str) -> None:
        logged = {f"{split}/{k}": float(v) for k, v in metrics.items()}
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": step, **logged}) + "\n")
        wandb.log(logged, step=step)
        print(f"[{split} step={step}] "
              + " ".join(f"{k}={float(v):.4e}" for k, v in metrics.items()), flush=True)

    # With wandb.mode=disabled every W&B call is a no-op.
    wandb.init(project=cfg.wandb.project, mode=cfg.wandb.mode, dir=run_dir,
               config=OmegaConf.to_container(cfg, resolve=True))
    start = int(state.step)
    if start == 0:
        log(validate(0), 0, "val")
    train_iter = task.batches(cfg, "train", t.global_batch_size, start)
    window = []
    for step in range(start + 1, t.num_steps + 1):
        state, metrics = train_step(model, optimizer, state, shard(next(train_iter)))
        ema_params = ema_update(ema_params, nn.state(model, nn.Param))
        window.append(metrics)
        if step % t.log_steps == 0:
            means = {k: sum(m[k] for m in window) / len(window) for k in window[0]}
            log({**jax.device_get(means), "lr": schedule(step - 1)}, step, "train")
            window = []
        if step % t.val_steps == 0:
            log(validate(step), step, "val")
        if step % t.save_every == 0:
            save(step)
    save(t.num_steps)
    ckpt.wait_until_finished()
    ema_ckpt.wait_until_finished()
    wandb.finish()


if __name__ == "__main__":
    main()
