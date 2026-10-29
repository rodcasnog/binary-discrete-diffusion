"""Checkpoint reads shared by training resume and evaluation."""
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import flax.nnx as nn
import orbax.checkpoint as ocp
from omegaconf import DictConfig, OmegaConf

from .data.factory import create_tokenizer
from .kernels import Kernel
from .models import build_model


def restore(manager: ocp.CheckpointManager, step: int, target: dict) -> dict:
    """Restore ``target`` from one step of ``manager``; Orbax raises unless the
    checkpoint holds exactly its tree and shapes."""
    return manager.restore(step, args=ocp.args.Composite(state=ocp.args.PyTreeRestore(
        target, restore_args=ocp.checkpoint_utils.construct_restore_args(target)))).state


def load_checkpoint(checkpoint_dir: str, overrides: Sequence[str]
                    ) -> tuple[nn.Module, DictConfig, Any, Kernel]:
    """Load ``<run>/checkpoints_ema/<step>`` with the run's ``.hydra/config.yaml``,
    updated by a dotlist of ``overrides`` such as ``["model.dtype=float32"]``.

    Returns ``(model, cfg, tokenizer, kernel)``.
    """
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    manager_dir = os.path.dirname(checkpoint_dir)
    cfg = OmegaConf.merge(
        OmegaConf.load(os.path.join(os.path.dirname(manager_dir), ".hydra", "config.yaml")),
        OmegaConf.from_dotlist(list(overrides)),
    )
    tokenizer = create_tokenizer(cfg)
    model = build_model(cfg, tokenizer, nn.Rngs(0))
    params = restore(ocp.CheckpointManager(manager_dir), int(os.path.basename(checkpoint_dir)),
                     {"params": nn.state(model, nn.Param)})["params"]
    nn.update(model, params)
    return model, cfg, tokenizer, Kernel.from_config(cfg, tokenizer)
