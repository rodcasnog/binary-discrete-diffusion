"""Reverse process: tempering, the kernels' exact bridges and the Psi-sampler; as a
script, text samples of a checkpoint in ``<out>/samples.json`` and ``<out>/tokens.npy``."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import partial

import flax.nnx as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from omegaconf import DictConfig, OmegaConf

from .checkpoints import load_checkpoint
from .data.text import decode, pad_anchors
from .kernels import (
    LOGIT_MASK,
    Kernel,
    _bernoulli,
    _clamp_bits,
    _randint,
    corrupt,
    denoiser_logits,
    log_or_mask,
    split_keys,
    uniform_shift,
)
from .schedule import survival_grid


def _categorical(keys, logits):
    # One key per row; high-dynamic-range Gumbels avoid implicit sharpening of small
    # probabilities.
    return jax.vmap(lambda key, row: random.categorical(key, row, mode="high"))(
        keys, logits)


@dataclass(frozen=True)
class Sampler:
    """Psi-sampler settings (``configs/eval/sampler/psi.yaml``); hashable, so the
    compiled levels take them as static. Every level takes the kernel's exact reverse
    bridge, except at the positions it re-corrupts (``eta``): there it decodes a clean
    token and corrupts it forward to the next level, so the state stays in the
    kernel's support. ``gamma``, ``lam`` and ``beta`` tilt that move by confidence,
    each keeping the untilted mean rate."""

    nfe: int
    spacing: str
    temperature: float
    top_p: float
    eta: float
    conf_coord: str
    gamma: float
    lam: float
    beta: float


def top_p_filter(logits: jnp.ndarray, top_p: float) -> jnp.ndarray:
    """Keep the smallest top-probability prefix whose mass reaches ``top_p``."""
    if top_p >= 1.0:
        return logits
    order = jnp.argsort(logits, axis=-1)[..., ::-1]
    sorted_logits = jnp.take_along_axis(logits, order, axis=-1)
    cdf = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
    # In descending order, keep the top token and every token whose predecessors hold
    # less than top_p; then scatter the mask back to vocabulary order.
    keep_sorted = jnp.concatenate(
        [jnp.ones_like(cdf[..., :1], dtype=jnp.bool_), cdf[..., :-1] < top_p],
        axis=-1,
    )
    keep = jnp.put_along_axis(
        jnp.zeros_like(keep_sorted), order, keep_sorted, axis=-1, inplace=False
    )
    return jnp.where(keep, logits, LOGIT_MASK)


def _tempered_logits(logits: jnp.ndarray, temperature: float, top_p: float):
    """Log belief for bridges, including an exact greedy limit."""
    if temperature <= 0.0:
        best = jnp.argmax(logits, axis=-1)
        return jnp.where(jax.nn.one_hot(best, logits.shape[-1], dtype=jnp.bool_),
                         0.0, LOGIT_MASK)
    belief = jnp.where(logits == LOGIT_MASK, LOGIT_MASK, logits / temperature)
    return top_p_filter(belief, top_p)


def decode_tokens(rng, logits, temperature: float, top_p: float):
    """Sample tokens, with non-positive temperature selecting greedy readout."""
    return _categorical(rng, _tempered_logits(logits, temperature, top_p))


@partial(nn.jit, static_argnames=("kernel",))
def predict(model, z, r, cond_mask, adjacency, *, kernel: Kernel):
    """Compiled model forward at survival ``r``, read as the denoiser and cavity
    logits ``[B, L, C]``."""
    t = 1.0 - r
    sigma = jnp.broadcast_to(kernel.time_cond(t), (z.shape[0],))
    logits = model(kernel.model_input(z), sigma, cond_mask=cond_mask, adjacency=adjacency)
    return denoiser_logits(kernel, logits, z, r, t), logits[..., :kernel.num_classes]


def _masked_step(kernel, sampler, dl, x, seq, cond, r_t, r_s, rng):
    """Unmask each masked position w.p. ``(r_s - r_t) / (1 - r_t)`` from the denoiser."""
    u_rng, t_rng = split_keys(rng)
    p_un = jnp.clip((r_s - r_t) / (1.0 - r_t), 0.0, 1.0)
    do_un = (x == kernel.mask_id) & _bernoulli(u_rng, p_un, x.shape)
    x = jnp.where(do_un, decode_tokens(t_rng, dl, sampler.temperature, sampler.top_p), x)
    return jnp.where(cond, seq, x)


def _uniform_step(kernel, sampler, dl, z, seq, cond, r_t, r_s, rng):
    """Exact finite bridge: ``z_s = y`` with probability proportional to
    ``q(z_t | y) m_s(y)``, where ``m_s = r_s mu + (1 - r_s) / C`` is the cavity's noisy
    marginal at ``r_s`` and the channel from ``s`` to ``t`` keeps a token w.p.
    ``r_t / r_s``, else redraws it uniformly. The cavity ``mu`` is the tempered
    denoiser with the forward likelihood divided out."""
    C = kernel.num_classes
    mu = jax.nn.softmax(
        _tempered_logits(dl, sampler.temperature, sampler.top_p)
        - uniform_shift(z, r_t, 1.0 - r_t, C),
        axis=-1,
    )
    r_ts = r_t / r_s
    m_s = r_s * mu + (1.0 - r_s) / C
    trans = (r_ts * jax.nn.one_hot(jnp.clip(z, 0, C - 1), C)
             + (1.0 - r_ts) * (1.0 / C))
    idx = _categorical(rng, log_or_mask(trans * m_s))
    return jnp.where(cond, seq, idx)


def _symmetric_step(kernel, sampler, dl, z, seq, cond, r_t, r_s, rng):
    """Exact bridge of the bits through a latent clean token ``c`` drawn from the
    denoiser: every bit is resampled from ``q(bit_s | bit_t, c)``."""
    C = kernel.num_classes
    zb = z[:, :, :C]
    c_rng, b_rng = split_keys(rng)
    c_oh = jax.nn.one_hot(
        decode_tokens(c_rng, dl, sampler.temperature, sampler.top_p), C, dtype=jnp.bool_
    )
    # q(bit_s = 1 | c): c's own bit is on w.p. (1 + r_s) / 2, every other w.p. (1 - r_s) / 2.
    m1 = jnp.where(c_oh, (1.0 + r_s) * 0.5, (1.0 - r_s) * 0.5)
    # From s to t a bit keeps its value w.p. (1 + rho) / 2.
    rho = r_t / r_s
    s1 = jnp.where(zb, (1.0 + rho) * 0.5, (1.0 - rho) * 0.5) * m1          # bit_s = 1
    s0 = jnp.where(zb, (1.0 - rho) * 0.5, (1.0 + rho) * 0.5) * (1.0 - m1)  # bit_s = 0
    bits = _bernoulli(b_rng, s1 / (s1 + s0), zb.shape)
    return _clamp_bits(z.at[:, :, :C].set(bits), cond, seq)


def _absorbing_step(kernel, sampler, dl, z, seq, cond, r_t, r_s, rng):
    """Exact bridge through a latent clean token: the token stays, every other
    active candidate stays w.p. ``t_s / t_t`` (second key unused)."""
    c_rng, _, k_rng = split_keys(rng, 3)
    t_t, t_s = 1.0 - r_t, 1.0 - r_s
    c = decode_tokens(c_rng, dl, sampler.temperature, sampler.top_p)
    c_oh = jax.nn.one_hot(c, kernel.vocab_size, dtype=jnp.bool_)
    keep = _bernoulli(k_rng, t_s / t_t, z.shape)
    return _clamp_bits(z & (keep | c_oh), cond, seq)


# The kernels' exact reverse transitions given the prediction.
BRIDGES = {"masked": _masked_step, "uniform": _uniform_step,
           "binary_symmetric": _symmetric_step, "binary_absorbing": _absorbing_step}


def _pad_bridge(kernel, z, seq, r_t, r_s, rng):
    """Exact q(z_s | z_t, x_0 = PAD) at every position. PAD lies outside the
    content alphabet: its content bits are all clean-off, and a uniform
    replacement never draws it back."""
    C = kernel.num_classes
    u_rng, k_rng, c_rng = split_keys(rng, 3)
    if kernel.name == "binary_symmetric":
        # _symmetric_step's bridge with every bit of c off.
        zb = z[:, :, :C]
        m1, rho = (1.0 - r_s) * 0.5, r_t / r_s
        s1 = jnp.where(zb, (1.0 + rho) * 0.5, (1.0 - rho) * 0.5) * m1
        s0 = jnp.where(zb, (1.0 - rho) * 0.5, (1.0 + rho) * 0.5) * (1.0 - m1)
        return z.at[:, :, :C].set(_bernoulli(k_rng, s1 / (s1 + s0), zb.shape))
    if kernel.name == "binary_absorbing":
        # Every active bit is a false candidate, kept w.p. t_s / t_t.
        keep = _bernoulli(k_rng, (1.0 - r_s) / (1.0 - r_t), z.shape)
        return z.at[:, :, :C].set(z[:, :, :C] & keep[:, :, :C])
    reveal = _bernoulli(u_rng, jnp.clip((r_s - r_t) / (1.0 - r_t), 0.0, 1.0), seq.shape)
    if kernel.name == "masked":
        return jnp.where((z == kernel.mask_id) & reveal, seq, z)
    # uniform: a replaced position returns to PAD w.p. reveal, else holds a content
    # token, redrawn w.p. 1 - r_t / r_s.
    redraw = _bernoulli(k_rng, 1.0 - r_t / r_s, seq.shape)
    content = jnp.where(redraw, _randint(c_rng, seq.shape, 0, C), z)
    return jnp.where((z == seq) | reveal, seq, content)


def _revision_probability(eta, strength, conf, eligible):
    """``eta`` on average over the eligible positions, spread among them in
    proportion to ``exp(-strength * conf)`` and capped at one, the cap refilled
    onto the rest so the average holds exactly; ``eta`` itself elsewhere.

    Per row this solves ``sum_i min(1, exp(c + logw_i)) = target`` for the log scale
    ``c`` by bisection."""
    # Log weights -strength * conf, shifted to 0 at the least confident eligible
    # position; -inf where not eligible.
    conf_min = jnp.where(eligible, conf, jnp.inf).min(1, keepdims=True)
    shift = jnp.where(jnp.isfinite(conf_min), conf_min, 0.0)
    logw = jnp.where(eligible, -strength * (conf - shift), -jnp.inf)
    target = eta * jnp.maximum(eligible.sum(1, keepdims=True), 1)
    # At lo the uncapped sum is exactly the target, so the capped one is at most it;
    # at hi every eligible position is capped at one.
    log_total = jnp.maximum(jax.nn.logsumexp(logw, axis=1, keepdims=True),
                            jnp.log(jnp.finfo(jnp.float32).tiny))
    lo = jnp.log(target) - log_total
    hi = jnp.maximum(lo, -jnp.where(eligible, logw, 0.0).min(1, keepdims=True))
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        short = jnp.minimum(1.0, jnp.exp(mid + logw)).sum(1, keepdims=True) < target
        lo, hi = jnp.where(short, mid, lo), jnp.where(short, hi, mid)
    return jnp.where(eligible, jnp.minimum(1.0, jnp.exp(hi + logw)), eta)


def _confident_bits(kernel, x0, r_s, p, rng, seq, cond, beta):
    """Re-corrupt ``x0`` to the per-position survival ``r_s``, opening the other
    content bits of a position as ``p ** beta`` with the cap and refill of
    ``_revision_probability``. The symmetric kernel drops the clean bit itself
    at the marginal rate."""
    C, V = kernel.num_classes, kernel.vocab_size
    x0_oh = jax.nn.one_hot(x0, C, dtype=jnp.bool_)
    rate = 1.0 - r_s
    if kernel.name == "binary_symmetric":
        rate = 0.5 * rate
    # With conf = -log p, the weights exp(-beta * conf) are p ** beta.
    neg_log_p = -jnp.log(jnp.maximum(p, jnp.finfo(jnp.float32).tiny))
    rho = _revision_probability(rate.reshape(-1, 1), beta,
                                neg_log_p.reshape(-1, C), (~x0_oh).reshape(-1, C)
                                ).reshape(x0_oh.shape)
    a_rng, d_rng = split_keys(rng)
    bits = x0_oh | (_bernoulli(a_rng, rho, x0_oh.shape) & ~x0_oh)
    if kernel.name == "binary_symmetric":
        bits = bits ^ (x0_oh & _bernoulli(d_rng, rate[..., None], x0_oh.shape))
    z = jnp.zeros(seq.shape + (V,), jnp.bool_).at[:, :, :C].set(bits)
    return _clamp_bits(z, cond, seq)


def _confident_token(kernel, x0, r_s, dl, rng, seq, cond, beta):
    """Uniform re-corruption of ``x0`` to the per-position survival ``r_s`` that
    draws the new token from the other content tokens as ``softmax(dl) ** beta``.
    A position changes at the forward kernel's rate ``(1 - r_s) (C - 1) / C``,
    since the forward draw can land back on ``x0``, so ``beta = 0`` is the
    forward corruption in law."""
    C = kernel.num_classes
    c_rng, u_rng = split_keys(rng)
    change = ~cond & _bernoulli(c_rng, (1.0 - r_s) * (C - 1) / C, seq.shape)
    logits = jnp.where(jax.nn.one_hot(x0, C, dtype=jnp.bool_), -jnp.inf,
                       beta * jax.nn.log_softmax(dl, axis=-1))
    return jnp.where(change, _categorical(u_rng, logits), x0)


@partial(nn.jit, static_argnames=("kernel", "sampler", "noisy_pad"))
def psi_level(model, z, stored, seq, cond, adjacency, r_t, r_s, rng, *,
              kernel: Kernel, sampler: Sampler, noisy_pad: bool):
    """One level from survival ``r_t`` to ``r_s``.

    ``stored`` is the sampler's memory. The masked kernel keeps a shown token's
    confidence from when it was unmasked: its loss carries a ``1{masked}`` factor,
    so the head is untrained wherever a token is already shown, and only shown
    tokens can be re-corrupted. The bit kernels keep each candidate's denoiser
    probability from when its bit was last on.
    """
    # The fourth key is unused.
    c_rng, n_rng, m_rng, _ = split_keys(
        jax.vmap(lambda key: random.fold_in(key, 1))(rng), 4)
    # The exact bridge, and a clean draw x0 with its confidence.
    dl, cav = predict(model, z, r_t, cond, adjacency, kernel=kernel)
    z_bridge = BRIDGES[kernel.name](kernel, sampler, dl, z, seq, cond, r_t, r_s, rng)
    x0 = jnp.where(cond, seq, decode_tokens(c_rng, dl, sampler.temperature, sampler.top_p))
    belief = jax.nn.softmax({"cavity": cav, "denoiser": dl}[sampler.conf_coord], axis=-1)
    conf = jnp.take_along_axis(belief, x0[..., None], -1)[..., 0]
    if kernel.name == "masked":
        conf = jnp.where(z == kernel.mask_id, conf, stored)
        eligible = (z != kernel.mask_id) & ~cond
    else:
        eligible = ~cond
    # The re-corruption of x0 to r_s: less confident positions go to a noisier level
    # (lam), and candidates reopen as p ** beta, p their denoiser probability (beta).
    r_fresh = jnp.full(seq.shape, r_s, jnp.float32)
    if sampler.lam > 0.0:
        r_fresh = 1.0 - _revision_probability(1.0 - r_s, sampler.lam, conf, ~cond)
    if sampler.beta > 0.0 and kernel.name == "uniform":
        z_fresh = _confident_token(kernel, x0, r_fresh, dl, n_rng, seq, cond, sampler.beta)
    elif sampler.beta > 0.0:
        stored = jnp.where(z[:, :, :kernel.num_classes],
                           jax.nn.softmax(dl, axis=-1), stored)
        z_fresh = _confident_bits(kernel, x0, r_fresh, stored, n_rng, seq, cond,
                                  sampler.beta)
    else:
        z_fresh, _ = corrupt(kernel, x0, r_fresh, n_rng, seq, cond)
    # A fraction eta of the eligible positions take the re-corruption instead of the
    # bridge, the less confident ones more often (gamma).
    rho = _revision_probability(sampler.eta, sampler.gamma, conf, eligible)
    revise = _bernoulli(m_rng, rho, seq.shape)
    z_new = jnp.where(revise[:, :, None] if kernel.is_bits else revise,
                      z_fresh, z_bridge)
    if noisy_pad:
        # The PAD anchors stay in the forward process, as training saw them.
        pad = _pad_bridge(kernel, z, seq, r_t, r_s,
                          jax.vmap(lambda key: random.fold_in(key, 2))(rng))
        z_new = jnp.where(cond[:, :, None] if kernel.is_bits else cond, pad, z_new)
    if kernel.name == "masked":
        # A newly unmasked token stores its current confidence.
        shown = jnp.take_along_axis(
            belief, jnp.clip(z_new, 0, kernel.num_classes - 1)[..., None], -1)[..., 0]
        stored = jnp.where((z == kernel.mask_id) & (z_new != kernel.mask_id), shown, stored)
    return z_new, stored


def sample(
    model: nn.Module,
    kernel: Kernel,
    sampler: Sampler,
    seq: jnp.ndarray,
    cond_mask: jnp.ndarray,
    keys: jnp.ndarray,
    *,
    adjacency: jnp.ndarray | None,
    noisy_pad: bool,
) -> jnp.ndarray:
    """Draw one sequence per row key of ``keys`` on the sampler's survival grid;
    the last network evaluation decodes the clean tokens. Anchors (``cond_mask``)
    are shown clean or, under ``noisy_pad``, are PAD tails kept in the forward
    process. The arrays are reused by every level, so pass them on device."""
    if sampler.beta > 0.0 and kernel.name == "masked":
        raise ValueError("beta reopens candidates, which the masked kernel does not have")
    grid = survival_grid(sampler.nfe, kernel.t_min, kernel.t_max, sampler.spacing
                         ).astype(np.float32)
    keys, init_keys = split_keys(keys)
    # The limiting prior with the anchors clamped: forward corruption to r = 0.
    z, _ = corrupt(kernel, jnp.where(cond_mask, seq, 0), 0.0, init_keys, seq, cond_mask)
    if noisy_pad:
        # The PAD anchors start forward-noised to the first level, on the init key.
        noisy, _ = corrupt(kernel, seq, grid[0], init_keys, seq, jnp.zeros_like(cond_mask))
        z = jnp.where(cond_mask[:, :, None] if kernel.is_bits else cond_mask, noisy, z)
    stored = (jnp.full(seq.shape + (kernel.num_classes,), 1.0 / kernel.num_classes)
              if kernel.is_bits else jnp.zeros(seq.shape, jnp.float32))
    for i in range(len(grid) - 1):
        keys, step_keys = split_keys(keys)
        z, stored = psi_level(model, z, stored, seq, cond_mask, adjacency, grid[i],
                              grid[i + 1], step_keys, kernel=kernel, sampler=sampler,
                              noisy_pad=noisy_pad)
    dl, _ = predict(model, z, grid[-1], cond_mask, adjacency, kernel=kernel)
    pred = decode_tokens(keys, dl, sampler.temperature, sampler.top_p)
    if kernel.name == "masked":
        pred = jnp.where(z == kernel.mask_id, pred, z)
    return jnp.where(cond_mask, seq, pred)


@hydra.main(config_path="../../configs/eval", config_name="sample", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Unconditional text samples, one key per sample. A padded corpus samples each
    text at the length of a validation record: its PAD tail is the anchor."""
    model, run_cfg, tokenizer, kernel = load_checkpoint(cfg.checkpoint_dir, cfg.overrides)
    if run_cfg.data.source != "text":
        raise ValueError("this samples text; logic tasks are solved by binary_diffusion.eval.solve")
    sampler = Sampler(**cfg.sampler)
    padded = run_cfg.data.padded
    if padded:
        seq, cond = pad_anchors(run_cfg, tokenizer, cfg.num_samples, cfg.seed)
    else:
        seq = np.zeros((cfg.num_samples, run_cfg.data.max_seq_len), np.int32)
        cond = np.zeros(seq.shape, bool)
    keys = random.split(random.PRNGKey(cfg.seed), cfg.num_samples)
    chunks = []
    for start in range(0, cfg.num_samples, cfg.batch_size):
        rows = slice(start, start + cfg.batch_size)
        chunks.append(np.asarray(sample(
            model, kernel, sampler, jnp.asarray(seq[rows]), jnp.asarray(cond[rows]), keys[rows],
            adjacency=None, noisy_pad=padded)))
        print(f"[sample] {start + len(chunks[-1])}/{cfg.num_samples}", flush=True)
    tokens = np.concatenate(chunks)
    np.save(os.path.join(cfg.out, "tokens.npy"), tokens)
    with open(os.path.join(cfg.out, "samples.json"), "w") as f:
        json.dump({"texts": decode(tokens, tokenizer, padded),
                   "run_config": OmegaConf.to_container(run_cfg, resolve=True)}, f, indent=2)


if __name__ == "__main__":
    main()
