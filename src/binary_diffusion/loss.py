"""Path losses of the four kernels for a cavity head, and the ELBO's boundary terms.

The model's logits ``[..., :C]`` are the cavity law: the clean-token belief from
the context, before the current state's own forward likelihood. Every path loss
is integrated directly in corruption time ``t = 1 - r`` and estimated as
``mean(J_t * dt/du)`` under the schedule's proposal, ``J_t`` being the path rate
times the rate divergence ``Phi`` summed over the reverse targets. The objectives:

- ``elbo``: the path ELBO, ``J_t`` integrated over ``[t_min, t_max]``;
- ``xent``: the cavity cross-entropy averaged over the base schedule;
- ``wxent``: the cavity cross-entropy times the path rate, integrated like ``elbo``.
"""
import math
from collections.abc import Callable
from functools import partial

import flax.nnx as nn
import jax
import jax.numpy as jnp
import optax
from jax import random
from jax.scipy.special import xlogy
from omegaconf import DictConfig

from .kernels import Kernel, corrupt, denoiser_logits
from .schedule import TimeSample, sample_time


def _phi(true_ratio: jnp.ndarray, model_ratio: jnp.ndarray) -> jnp.ndarray:
    """Local rate divergence Phi(a, b) = b - a + a·(log a - log b), elementwise."""
    return (model_ratio - true_ratio
            + xlogy(true_ratio, true_ratio)
            - xlogy(true_ratio, model_ratio))


def _scored_positions(tokens, pad_token_id, cond_mask):
    """Non-padding positions, minus conditioning anchors (clue cells)."""
    non_padding = (tokens != pad_token_id).astype(jnp.float32)
    return non_padding * (~cond_mask).astype(non_padding.dtype)


def _non_padding_mean(per_token: jnp.ndarray, non_padding: jnp.ndarray) -> jnp.ndarray:
    """Nats per scored token (the count clipped to 1 for an all-padding batch)."""
    return jnp.sum(per_token * non_padding) / jnp.sum(non_padding).clip(1.0)


def _active_log_masses(
    logits: jnp.ndarray,
    active: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``log sum_A exp(logits)`` over the active set ``A`` and, for every ``u``,
    the leave-one-out ``log sum_{A minus u} exp(logits)``.

    The sums are taken relative to the active maximum, so leaving out any other
    entry keeps a sum of at least one. Leaving out the maximum itself would cancel
    nearly equal terms, so that sum is recomputed relative to the second maximum.
    This keeps the exact loss and its gradient without a probability floor.
    """
    masked = jnp.where(active, logits, -jnp.inf)
    top_index = jnp.argmax(masked, axis=-1)
    top = jnp.max(masked, axis=-1)
    top_oh = jax.nn.one_hot(top_index, logits.shape[-1], dtype=jnp.bool_)

    # Leaving out an entry other than the maximum.
    scaled = jnp.where(active, jnp.exp(masked - top[..., None]), 0.0)
    total_scaled = scaled.sum(axis=-1)
    remaining = total_scaled[..., None] - scaled
    log_without_regular = top[..., None] + jnp.log(
        jnp.where(remaining > 0.0, remaining, 1.0)
    )

    # Leaving out the maximum.
    without_top = active & ~top_oh
    second = jnp.max(jnp.where(without_top, logits, -jnp.inf), axis=-1)
    has_second = jnp.any(without_top, axis=-1)
    second_safe = jnp.where(has_second, second, 0.0)
    second_shift = jnp.where(without_top, logits - second_safe[..., None], 0.0)
    second_sum = jnp.sum(jnp.where(without_top, jnp.exp(second_shift), 0.0), axis=-1)
    log_without_top = second_safe + jnp.log(jnp.where(has_second, second_sum, 1.0))
    log_without = jnp.where(top_oh, log_without_top[..., None], log_without_regular)
    log_active = top + jnp.log(total_scaled)
    return log_active, log_without


@jax.custom_vjp
def grad_probe(logits: jnp.ndarray, probe: jnp.ndarray) -> jnp.ndarray:
    """Identity in ``logits`` whose cotangent on ``probe`` (zeros, shape ``[B]``)
    is the per-sequence squared norm of the logits cotangent: the adaptive time
    sampler's cheap stand-in for each sequence's parameter gradient."""
    return logits


def _grad_probe_fwd(logits, probe):
    return logits, None


def _grad_probe_bwd(_, ct):
    ct32 = ct.astype(jnp.float32)
    return ct, jnp.sum(ct32 * ct32, axis=tuple(range(1, ct.ndim)))


grad_probe.defvjp(_grad_probe_fwd, _grad_probe_bwd)


def _bags(tokens, labels, z, C):
    """The clean one-hot and the noisy bits of a bit state on the content alphabet.

    Out-of-alphabet tokens (padding, specials) get a harmless one-hot bag: their
    loss weight is zero, but ``logsumexp`` over an empty bag would still create NaN
    gradients before that multiplication. Valid content positions, including a
    symmetric state whose noisy bits happen to be all zero, are untouched.
    """
    truth = jax.nn.one_hot(tokens, C, dtype=jnp.bool)
    empty = ~jnp.any(truth, axis=-1, keepdims=True)
    canon = jax.nn.one_hot(labels, C, dtype=jnp.bool)
    return jnp.where(empty, canon, truth), jnp.where(empty, canon, z[..., :C])


# Per kernel: (path rate [B, 1], Phi summed over the reverse targets, the cross-entropy
# the xent objectives use), from the content logits, the clean tokens and labels, the
# noisy state z, the positions the kernel corrupted and the sampled times.

def _masked_terms(logits, tokens, labels, z, changed, times, xent, C):
    """Each token is masked w.p. ``t``. Denoiser and cavity coincide at masked
    positions, so ``J_t = 1/t · 1{masked} · xent``."""
    flat = changed.astype(jnp.float32) * xent
    return (1.0 / times.t)[:, None], flat, flat


def _uniform_terms(logits, tokens, labels, z, changed, times, xent, C):
    """Each token is replaced w.p. ``t`` by a uniform draw over the C content
    tokens, so ``q(y | x0) = r 1{y = x0} + t/C``.

    The reverse jump rate to ``y != z_t`` is ``(1/C)/r`` times the ratio
    ``g(y) = m(y)/m(z_t)`` of the cavity's noisy marginal ``m = r mu + t/C``. Against
    the true noisy marginal ``d``, the Phi-sum over those targets has the closed
    form ``(KL(d||m) + IS(d(z_t)||m(z_t))) / d(z_t)``, whose only row reduction is
    ``S = sum_y log m(y)``.
    """
    noisy_ids = jnp.minimum(z, C - 1)
    # The forward channel, one value per row: q_same at y = x0, q_other at y != x0.
    q_other = times.t / C
    q_same = times.r + q_other
    d_zt = jnp.where(noisy_ids == labels, q_same[:, None], q_other[:, None])
    lse = jax.nn.logsumexp(logits, axis=-1)
    logit_zt = jnp.take_along_axis(logits, noisy_ids[..., None], axis=-1)[..., 0]
    # log m(y) = logaddexp(log r + log mu(y), log q_other).
    log_r, log_q_other = jnp.log(times.r)[:, None], jnp.log(q_other)[:, None]
    S = jnp.sum(jnp.logaddexp(log_r[..., None] + logits - lse[..., None],
                              log_q_other[..., None]), axis=-1)
    logit_x0 = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    log_m_x0 = jnp.logaddexp(log_r + logit_x0 - lse, log_q_other)
    log_m_zt = jnp.logaddexp(log_r + logit_zt - lse, log_q_other)
    # KL(d||m) = sum d log d - sum d log m, where d is q_same at x0 and q_other
    # elsewhere, and q_same - q_other = r.
    d_log_d = q_same * jnp.log(q_same) + (C - 1) * q_other * jnp.log(q_other)
    kl = d_log_d[:, None] - times.r[:, None] * log_m_x0 - q_other[:, None] * S
    # IS(a||b) = a/b - log(a/b) - 1 at the current state.
    log_ratio = jnp.log(d_zt) - log_m_zt
    phi_sum = (kl + jnp.exp(log_ratio) - log_ratio - 1.0) / d_zt
    return (1.0 / C) / times.r[:, None], phi_sum, xent


def _symmetric_terms(logits, tokens, labels, z, changed, times, xent, C):
    """Each content bit flips w.p. ``t/2``. Thus ``J_t = (1/(2r)) · sum_u Phi(a_u,
    b_u)`` with the true per-bit rate ratio ``a_u = rho`` on mismatched bits and
    ``eta = 1/rho`` on matched ones, ``rho = (1+r)/(1-r)``. The cavity's ratio is
    ``b_u = gamma_u · (rho p^(u) + eta (1-p^(u))) / (rho p + eta (1-p))`` with
    ``p = <mu, z>``, ``p^(u)`` that mass after flipping bit ``u``, and
    ``gamma_u = rho`` on set bits, ``eta`` on clear ones.
    """
    truth, bits = _bags(tokens, labels, z, C)
    probs = jax.nn.softmax(logits, axis=-1)  # the cavity mu, [B, L, C]
    rho_t = (2.0 - times.t) / times.t  # (1 + r) / (1 - r)
    eta_t = times.t / (2.0 - times.t)  # 1 / rho
    rho_b, eta_b = rho_t[:, None, None], eta_t[:, None, None]
    # p = <mu, z> and, for every bit u, p^(u): that mass after flipping u.
    p_set = jnp.sum(probs, axis=-1, where=bits)  # [B, L]
    denom = rho_t[:, None] * p_set + eta_t[:, None] * (1.0 - p_set)
    p_flip = p_set[:, :, None] + jnp.where(bits, -probs, probs)
    gamma = jnp.where(bits, rho_b, eta_b)
    model_ratio = gamma * (rho_b * p_flip + eta_b * (1.0 - p_flip)) / denom[:, :, None]
    true_ratio = jnp.where(bits != truth, rho_b, eta_b)
    return 0.5 / times.r[:, None], jnp.sum(_phi(true_ratio, model_ratio), axis=-1), xent


def _absorbing_terms(logits, tokens, labels, z, changed, times, xent, C):
    """``z_t`` is a candidate set ``A`` that always holds the clean token, each other
    content bit active w.p. ``t``. The reverse process deletes false candidates,
    ``J_t = (1/t) · sum_{u in A} Phi(1{u != x0}, b_u)``, and the cavity's deletion
    ratio ``b_u = (p_A - p_u)/p_A`` (``p_A`` the cavity mass on ``A``) cancels the
    linear Phi terms, leaving ``sum_{u != x0} log(p_A / (p_A - p_u))``.
    """
    truth, bits = _bags(tokens, labels, z, C)
    log_set, log_without = _active_log_masses(logits, bits)
    phi_sum = jnp.sum(jnp.where(bits & ~truth, log_set[..., None] - log_without, 0.0), axis=-1)
    return 1.0 / times.t[:, None], phi_sum, xent


TERMS = {"masked": _masked_terms, "uniform": _uniform_terms,
         "binary_symmetric": _symmetric_terms, "binary_absorbing": _absorbing_terms}


def make_loss_fn(cfg: DictConfig, tokenizer) -> Callable:
    """The run's loss ``fn(model, batch, rng, *, bin_probs=None, probe=None)``,
    returning ``(loss, ({"loss", "elbo"}, time_sample))``. Validation passes no
    ``bin_probs`` (base schedule) and no ``probe``."""
    kernel = Kernel.from_config(cfg, tokenizer)
    terms, objective = TERMS[kernel.name], cfg.diffusion.objective
    C, pad_token_id = kernel.num_classes, tokenizer.pad_token_id

    def loss_fn(model, batch, rng, *, bin_probs=None, probe=None):
        tokens, cond = batch["input_ids"], batch["cond_mask"]
        B = tokens.shape[0]
        noise_rng, corrupt_rng = random.split(rng)
        times = sample_time(noise_rng, B, kernel.t_min, kernel.t_max, kernel.schedule, bin_probs)
        z, changed = corrupt(kernel, tokens, times.r, random.split(corrupt_rng, B), tokens, cond)
        logits = model(kernel.model_input(z), kernel.time_cond(times.t), cond_mask=cond,
                       adjacency=batch.get("adjacency"))
        if probe is not None:
            logits = grad_probe(logits, probe)
        # Out-of-alphabet tokens are never scored, so clamping their label is harmless.
        logits, labels = logits[..., :C], jnp.minimum(tokens, C - 1)
        xent = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        path_rate, phi_sum, xent = terms(logits, tokens, labels, z, changed, times, xent, C)
        scored = _scored_positions(tokens, pad_token_id, cond)
        # The time integrals carry the proposal's dt/du (jac, importance weight included);
        # xent, an average over the base schedule, carries only the importance weight.
        jac, is_w = times.jac[:, None], times.is_weight[:, None]
        weighted = {"elbo": path_rate * phi_sum * jac, "xent": xent * is_w,
                    "wxent": path_rate * xent * jac}
        loss = _non_padding_mean(weighted[objective], scored)
        metrics = {"loss": loss, "elbo": _non_padding_mean(weighted["elbo"], scored)}
        # Per-sequence summands at the times they were drawn; their batch mean is the loss.
        summand = jnp.sum(weighted[objective] * scored, axis=1) * (B / jnp.sum(scored).clip(1.0))
        return loss, (metrics, TimeSample(u=times.u, is_weight=times.is_weight,
                                          summand=jax.lax.stop_gradient(summand)))

    return loss_fn


@partial(nn.jit, static_argnames=("kernel", "pad_token_id"))
def reconstruction(model: nn.Module, batch: dict, rng: jnp.ndarray, *,
                   kernel: Kernel, pad_token_id: int) -> jnp.ndarray:
    """Monte Carlo endpoint term ``E[-log pi_theta(x0 | z_{t_min})]`` per scored
    token, decoded by the denoiser. A masked position shown clean costs nothing."""
    tokens, cond_mask = batch["input_ids"], batch["cond_mask"]
    t = kernel.t_min
    z, open_positions = corrupt(kernel, tokens, 1.0 - t,
                                random.split(rng, tokens.shape[0]), tokens, cond_mask)
    t_b = jnp.full((tokens.shape[0],), t, dtype=jnp.float32)
    logits = model(kernel.model_input(z), kernel.time_cond(t_b), cond_mask=cond_mask,
                   adjacency=batch.get("adjacency"))
    scored = _scored_positions(tokens, pad_token_id, cond_mask)
    if kernel.name != "masked":
        open_positions = scored
    labels = jnp.minimum(tokens, kernel.num_classes - 1)
    log_pi = jax.nn.log_softmax(denoiser_logits(kernel, logits, z, 1.0 - t, t), axis=-1)
    log_pi_truth = jnp.take_along_axis(log_pi, labels[..., None], axis=-1)[..., 0]
    return -_non_padding_mean(log_pi_truth * open_positions.astype(jnp.float32), scored)


def prior_kl(kernel: Kernel) -> float:
    """KL per scored token from the forward marginal at ``t_max`` to the sampler's
    limiting prior, in float64. Zero for the masked and binary absorbing kernels:
    their limiting prior is a point mass, whose KL stays infinite for ``t_max < 1``,
    and the ELBO omits it."""
    r, C = 1.0 - kernel.t_max, kernel.num_classes
    if kernel.name == "uniform":
        # Against the uniform law over the C content tokens: the forward marginal holds
        # r + (1 - r) / C on x0 and (1 - r) / C on each other content token.
        pi0 = 1.0 / C
        truth = r + (1.0 - r) * pi0
        return truth * math.log(truth / pi0) + (1.0 - r) * (1.0 - pi0) * math.log(1.0 - r)
    if kernel.name == "binary_symmetric":
        # Against independent fair bits; this form avoids subtracting two O(r)
        # terms to recover an O(r^2) KL.
        return C * (0.5 * (math.log1p(-r * r) + r * (math.log1p(r) - math.log1p(-r))))
    return 0.0
