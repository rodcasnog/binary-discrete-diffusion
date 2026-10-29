"""Corruption-time schedules, the adaptive time sampler and sampling grids, with
survival ``r = 1 - t``.

A schedule maps a unit coordinate ``u`` onto ``[t_min, t_max]``: ``linear`` in
``t``, ``log`` in ``log t``, and ``cosine`` as ``t_min + (t_max - t_min) sin^2(pi u / 2)``.
"""
from __future__ import annotations

import math

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

# Adaptive time sampler: equal-width bins of the schedule's unit coordinate, the
# decay of their statistics, and the proposal mass kept on the base schedule. The
# base schedule is used for the first 1/(1 - ema) updates.
CLOCK_BINS, CLOCK_EMA, CLOCK_FLOOR = 32, 0.99, 0.1
CLOCK_WARMUP = math.ceil(1.0 / (1.0 - CLOCK_EMA))


def _window(t_min: float, t_max: float, schedule: str) -> tuple[float, float, float]:
    """float64 ``(offset, scale, rate)`` of a schedule: ``offset`` and ``scale`` map
    unit draws to physical time, ``rate`` scales the Jacobian."""
    if schedule == "linear":
        return t_min, t_max - t_min, t_max - t_min
    if schedule == "log":
        lo, hi = math.log(t_min), math.log(t_max)
        return lo, hi - lo, math.log(t_max / t_min)
    if schedule == "cosine":
        return t_min, t_max - t_min, math.pi * (t_max - t_min)
    raise ValueError(f"unknown schedule {schedule!r} (use linear, log or cosine)")


def _time_from_unit(u, offset, scale, schedule: str, xp):
    if schedule == "linear":
        return offset + u * scale
    if schedule == "log":
        return xp.exp(offset + u * scale)
    return offset + scale * xp.sin(0.5 * math.pi * u) ** 2


def model_time_cond(t: jnp.ndarray, t_min: float, t_max: float, schedule: str) -> jnp.ndarray:
    """Model input: the normalized training coordinate of physical time ``t``."""
    t = jnp.asarray(t, jnp.float32)
    if schedule == "linear":
        return (t - t_min) / (t_max - t_min)
    if schedule == "log":
        return (jnp.log(t) - math.log(t_min)) / math.log(t_max / t_min)
    s = (t - t_min) / (t_max - t_min)
    return (2.0 / math.pi) * jnp.arcsin(jnp.sqrt(jnp.clip(s, 0.0, 1.0)))


def stratified_u(rng: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    """Unit draws, one per randomly shifted equal-width stratum.

    The random shift prevents a permanent association between batch position and
    noise level while retaining the variance reduction of stratification.
    """
    jitter_rng, shift_rng = random.split(rng)
    u = random.uniform(jitter_rng, (batch_size,), dtype=jnp.float32)
    shift = random.randint(shift_rng, (), 0, batch_size)
    strata = (jnp.arange(batch_size, dtype=jnp.int32) + shift) % batch_size
    return (strata.astype(jnp.float32) + u) / batch_size


@flax.struct.dataclass
class TimeBatch:
    """One batch of physical times and proposal Jacobians."""

    t: jnp.ndarray
    r: jnp.ndarray
    jac: jnp.ndarray              # dt / dv of the proposal drawn
    u: jnp.ndarray                # base-schedule unit coordinate
    is_weight: jnp.ndarray        # q_base(t) / p(t); ones for a fixed schedule


def warp_to_bins(
    v: jnp.ndarray,
    probs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Inverse-CDF map of unit draws through the piecewise-constant density
    ``K probs[k]`` on equal-width bins of ``[0, 1]``; returns the warped
    coordinate and its importance weight ``1 / (K probs[k])``."""
    K = probs.shape[0]
    cdf = jnp.cumsum(probs)
    k = jnp.minimum(jnp.sum(v[:, None] >= cdf[None, :], axis=1), K - 1)
    p_k = jnp.maximum(probs[k], jnp.finfo(jnp.float32).tiny)
    u = jnp.clip((k + (v - (cdf[k] - probs[k])) / p_k) / K, 0.0, 1.0)
    return u, 1.0 / (K * p_k)


def sample_time(rng: jnp.ndarray, batch_size: int, t_min: float, t_max: float,
                schedule: str, bin_probs: jnp.ndarray | None) -> TimeBatch:
    """Sample physical time with the exact Jacobian of the proposal drawn;
    ``bin_probs`` first warps the unit draws through :func:`warp_to_bins`."""
    offset, scale, rate = _window(t_min, t_max, schedule)
    u = stratified_u(rng, batch_size)
    is_weight = jnp.ones_like(u)
    if bin_probs is not None:
        u, is_weight = warp_to_bins(u, bin_probs)
    t = _time_from_unit(u, offset, scale, schedule, jnp)
    if schedule == "linear":
        jac = jnp.full_like(t, rate)
    elif schedule == "log":
        jac = t * rate
    else:
        # t = t_min + span sin^2(pi u / 2)  =>  dt/du = span pi sqrt(s (1 - s)).
        s = (t - offset) / scale
        jac = rate * jnp.sqrt(jnp.clip(s * (1.0 - s), 0.0))
    return TimeBatch(t=t, r=1.0 - t, jac=jac * is_weight, u=u, is_weight=is_weight)


def survival_grid(nfe: int, t_min: float, t_max: float, spacing: str) -> np.ndarray:
    """Return exactly ``nfe`` ascending survival levels from noisy to clean."""
    offset, scale, _ = _window(t_min, t_max, spacing)
    t = _time_from_unit(np.linspace(1.0, 0.0, nfe), offset, scale, spacing, xp=np)
    r = np.asarray(1.0 - t, dtype=np.float64)
    if nfe == 1:
        r[0] = 1.0 - t_max
    else:
        r[0], r[-1] = 1.0 - t_max, 1.0 - t_min
    return r


@flax.struct.dataclass
class TimeSample:
    """Per-sequence sampler statistics of one loss evaluation."""

    u: jnp.ndarray                # base-schedule unit coordinate
    is_weight: jnp.ndarray        # q_base(t) / p(t)
    summand: jnp.ndarray          # training loss == mean(summand)


@flax.struct.dataclass
class AdaptiveClockState:
    moments: jnp.ndarray          # (2, K) EMA bin sums: count, |dg/dlogits|^2
    updates: jnp.ndarray


def clock_init() -> AdaptiveClockState:
    return AdaptiveClockState(moments=jnp.zeros((2, CLOCK_BINS), jnp.float32),
                              updates=jnp.zeros((), jnp.int32))


def bin_probs(state: AdaptiveClockState) -> jnp.ndarray:
    """Proposal mass per bin of the adaptive time sampler: each bin weighted by
    ``sqrt(E |dg/dlogits|^2)`` of the objective's base-schedule integrand ``g`` and
    mixed with the base schedule at ``CLOCK_FLOOR``; uniform (the base schedule)
    during warmup and while any bin lacks a finite statistic. The importance
    weights leave the objective unchanged in expectation."""
    n, sg = state.moments
    d = jnp.maximum(n, jnp.finfo(jnp.float32).tiny)
    w = jnp.sqrt(jnp.maximum(sg / d, 0.0))
    total = jnp.sum(w)
    ready = ((state.updates >= CLOCK_WARMUP) & jnp.all(n > 0.0)
             & jnp.all(jnp.isfinite(w)) & (total > 0.0))
    p = (1.0 - CLOCK_FLOOR) * w / jnp.where(total > 0.0, total, 1.0) + CLOCK_FLOOR / CLOCK_BINS
    p = jnp.where(ready, p, 1.0 / CLOCK_BINS)
    return jax.lax.stop_gradient(p / jnp.sum(p))


def clock_update(state: AdaptiveClockState, sample: TimeSample,
                 probe_grad: jnp.ndarray) -> AdaptiveClockState:
    """EMA the batch's per-bin sums. ``probe_grad`` is the per-sequence
    ``|dloss/dlogits|^2`` from :func:`binary_diffusion.loss.grad_probe`, standing in
    for the parameter gradient of ``g = summand / is_weight``. Sequences with a
    non-finite statistic are left out."""
    g = sample.summand / sample.is_weight
    scale = sample.u.shape[0] / sample.is_weight
    g2 = probe_grad * scale * scale
    rows = jnp.stack([jnp.ones_like(g), g2], axis=1)
    rows = jnp.where((jnp.isfinite(g) & jnp.isfinite(g2))[:, None], rows, 0.0)
    k = jnp.clip(jnp.floor(sample.u * CLOCK_BINS), 0, CLOCK_BINS - 1).astype(jnp.int32)
    onehot = (k[:, None] == jnp.arange(CLOCK_BINS)[None, :]).astype(jnp.float32)
    sums = jax.lax.stop_gradient(onehot.T @ rows).T
    return AdaptiveClockState(moments=CLOCK_EMA * state.moments + sums,
                              updates=state.updates + 1)
