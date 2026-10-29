"""Forward processes of the four kernels and the denoiser readout of a cavity head.

Survival is ``r = 1 - t``, and anchor positions (``cond``) stay clean. The noise
draws only content tokens ``[0, num_classes)``, or the masked kernel's mask; special
tokens and the mask take the ids above them. The bit kernels hold a state
``z in {0,1}^V`` per position, starting from the clean token's one-hot: binary
symmetric flips each content bit w.p. ``t/2``, binary absorbing switches each other
content bit on w.p. ``t``, so its state is a candidate set holding the clean token.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random
from omegaconf import DictConfig

from .schedule import model_time_cond

# Finite log-zero for softmax and categorical operations.
LOGIT_MASK = -1e9

BIT_KERNELS = ("binary_symmetric", "binary_absorbing")


# Random draws take one key per row and vmap over rows, so a row's draws do not
# depend on the batch it is sampled in.
def split_keys(keys: jnp.ndarray, count: int = 2) -> jnp.ndarray:
    return jnp.swapaxes(jax.vmap(lambda key: random.split(key, count))(keys), 0, 1)


def _bernoulli(keys, p, shape):
    p = jnp.asarray(p, jnp.float32)
    p = jnp.broadcast_to(p.reshape(p.shape + (1,) * (len(shape) - p.ndim)), shape)
    return jax.vmap(lambda key, row: random.bernoulli(key, row))(keys, p)


def _randint(keys, shape, minval, maxval):
    return jax.vmap(lambda key: random.randint(key, shape[1:], minval, maxval))(keys)


def _bcast(x: jnp.ndarray | float, target: jnp.ndarray) -> jnp.ndarray:
    """Add trailing axes so a time scalar broadcasts over ``target``."""
    x = jnp.asarray(x, dtype=jnp.float32)
    return x.reshape(x.shape + (1,) * (target.ndim - x.ndim))


def log_or_mask(x: jnp.ndarray) -> jnp.ndarray:
    """Elementwise log, representing exact zero by ``LOGIT_MASK``.

    Replacing zero by one in the operand keeps both branches' derivatives finite;
    unlike a probability floor, this does not change any positive mass.
    """
    positive = x > 0.0
    return jnp.where(positive, jnp.log(jnp.where(positive, x, 1.0)), LOGIT_MASK)


def encode_bits(bits: jnp.ndarray) -> jnp.ndarray:
    """Model input of a bit state: the bits scaled to unit l2 norm over all ``V``
    bits, anchor and special bits included."""
    bits_f = bits.astype(jnp.float32)
    return bits_f / bits_f.sum(-1, keepdims=True).clip(1.0) ** 0.5


@dataclass(frozen=True)
class Kernel:
    """A model's forward process; hashable, so compiled functions take it as static."""

    name: str
    vocab_size: int
    num_classes: int          # content alphabet: only [0, num_classes) diffuses
    mask_id: int
    t_min: float
    t_max: float
    schedule: str             # training time schedule, also the model's time input

    def __post_init__(self):
        # The path rates divide by t and by r.
        if not 0.0 < self.t_min < self.t_max < 1.0:
            raise ValueError(
                f"need 0 < t_min < t_max < 1, got t_min={self.t_min}, t_max={self.t_max}")

    @classmethod
    def from_config(cls, cfg: DictConfig, tokenizer) -> Kernel:
        d = cfg.diffusion
        return cls(
            name=d.kernel,
            vocab_size=len(tokenizer),
            num_classes=int(tokenizer.num_classes),
            mask_id=tokenizer.mask_token_id,
            t_min=float(d.t_min),
            t_max=float(d.t_max),
            schedule=d.schedule,
        )

    @property
    def is_bits(self) -> bool:
        return self.name in BIT_KERNELS

    def time_cond(self, t: jnp.ndarray) -> jnp.ndarray:
        """The model's time input: ``t`` in the training schedule's unit coordinate."""
        return model_time_cond(t, self.t_min, self.t_max, self.schedule)

    def model_input(self, z: jnp.ndarray) -> jnp.ndarray:
        return encode_bits(z) if self.is_bits else z


def _content(kernel: Kernel) -> jnp.ndarray:
    return jnp.arange(kernel.vocab_size) < kernel.num_classes


def _clamp_bits(z, cond, seq):
    seq_oh = jax.nn.one_hot(seq, z.shape[-1], dtype=jnp.bool_)
    return jnp.where(cond[:, :, None], seq_oh, z)


def corrupt(
    kernel: Kernel,
    x0: jnp.ndarray,
    r_s: jnp.ndarray | float,
    keys: jnp.ndarray,
    seq: jnp.ndarray,
    cond: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Forward-noise clean tokens ``x0`` to survival ``r_s`` (a scalar, one level per
    row or one per position), one key per row, leaving ``cond`` anchors clean. Also
    returns the positions the kernel corrupted: the masked or replaced ones, and every
    unanchored one for the bit kernels."""
    r_s = jnp.asarray(r_s, jnp.float32)
    rate = 1.0 - (r_s[:, None] if r_s.ndim == 1 else r_s)
    scored = ~cond
    if kernel.is_bits:
        x0_oh = jax.nn.one_hot(x0, kernel.vocab_size, dtype=jnp.bool_)
        bit_rate = rate[..., None]
    if kernel.name == "binary_symmetric":
        flips = _bernoulli(keys, 0.5 * bit_rate, x0_oh.shape) & _content(kernel)
        return _clamp_bits(x0_oh ^ flips, cond, seq), scored
    if kernel.name == "binary_absorbing":
        add = _bernoulli(keys, bit_rate, x0_oh.shape) & _content(kernel)
        return _clamp_bits(x0_oh | add, cond, seq), scored
    if kernel.name == "masked":
        remask = scored & _bernoulli(keys, rate, seq.shape)
        return jnp.where(remask, kernel.mask_id, x0), remask
    # uniform: replace w.p. 1 - r_s by a uniform content draw (third key unused).
    rep_rng, u_rng, _ = split_keys(keys, 3)
    rep = scored & _bernoulli(rep_rng, rate, seq.shape)
    repl = _randint(u_rng, seq.shape, 0, kernel.num_classes)
    return jnp.where(rep, repl, x0), rep


def uniform_shift(z: jnp.ndarray, r: jnp.ndarray | float, t: jnp.ndarray | float,
                  C: int) -> jnp.ndarray:
    """Uniform kernel's ``log q(z | x0 = l)`` up to an ``l``-independent constant:
    ``log(1 + r / (t/C))`` at ``l = z``, zero elsewhere. Positions holding a
    non-content token are never scored, so their clamped one-hot is harmless."""
    r_b = _bcast(r, z)
    t_b = _bcast(t, z)
    one_hot = jax.nn.one_hot(jnp.clip(z, 0, C - 1), C)
    return one_hot * jnp.log1p(r_b / (t_b / C))[..., None]


def denoiser_logits(kernel: Kernel, logits: jnp.ndarray, z: jnp.ndarray,
                    r: jnp.ndarray | float, t: jnp.ndarray | float) -> jnp.ndarray:
    """Clean-token posterior logits [B,L,C] of a cavity head: the cavity law
    ``logits[..., :C]`` times the forward likelihood of the current state ``z``."""
    C = kernel.num_classes
    if kernel.name == "masked":
        # A shown token is its own posterior.
        point = jnp.where(
            jax.nn.one_hot(jnp.clip(z, 0, C - 1), C, dtype=jnp.bool_),
            0.0,
            LOGIT_MASK,
        )
        return jnp.where((z != kernel.mask_id)[..., None], point, logits[:, :, :C])
    if kernel.name == "uniform":
        return logits[..., :C] + uniform_shift(z, r, t, C)
    if kernel.name == "binary_symmetric":
        # A set bit multiplies its token's odds by rho = (1 + r) / (1 - r), a clear
        # bit divides them by it.
        z_signed = 2.0 * z[:, :, :C].astype(jnp.float32) - 1.0
        log_rho = jnp.log1p(jnp.asarray(r, jnp.float32)) - jnp.log(t)
        return logits[:, :, :C] + _bcast(log_rho, z_signed) * z_signed
    # binary_absorbing: the likelihood is constant on the active set, zero off it.
    return jnp.where(z[:, :, :C], logits[:, :, :C], LOGIT_MASK)
