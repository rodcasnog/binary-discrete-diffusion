"""Diffusion transformers with adaLN time conditioning, compute in ``model.dtype`` and
float32 parameters, norms and logits: the DiT attends densely with rotary positions,
the GraphDiT along a graph. Adapted from the DiT of MDLM
(https://github.com/kuleshov-group/mdlm), licensed under Apache-2.0."""

import math

import flax.nnx as nn
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

FREQ_DIM = 256  # sinusoidal features of the diffusion time


def _rotary(dim: int, max_seq_len: int, theta: float = 10000.0):
    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    freqs = jnp.outer(jnp.arange(max_seq_len, dtype=jnp.float32), inv_freq)
    freqs = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(freqs), jnp.sin(freqs)


def _rotate(x, cos, sin):
    half = x.shape[-1] // 2
    rotated = jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return (x * cos[None, :, None, :]) + (rotated * sin[None, :, None, :])


def modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


def kaiming_uniform(key, shape, dtype=jnp.float32):
    bound = math.sqrt(5.0) * math.sqrt(1.0 / shape[0])
    return jax.random.uniform(key, shape, dtype, -bound, bound)


def lecun_normal(key, shape, dtype=jnp.float32):
    return jax.random.normal(key, shape, dtype) * (1.0 / math.sqrt(shape[0]))


class LayerNorm(nn.Module):
    def __init__(self, dim: int, rngs: nn.Rngs):
        self.scale = nn.Param(nn.initializers.ones(rngs.params(), (dim,)))

    def __call__(self, x):
        x = x.astype(jnp.float32)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return (x - mean) / jnp.sqrt(var + 1e-5) * self.scale.value


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, rngs: nn.Rngs):
        self.mlp_0 = nn.Linear(FREQ_DIM, dim, rngs=rngs)
        self.mlp_2 = nn.Linear(dim, dim, rngs=rngs)

    def __call__(self, t):
        half = FREQ_DIM // 2
        freqs = jnp.exp(-math.log(10000) * jnp.arange(0, half, dtype=jnp.float32) / half)
        args = t[:, None].astype(jnp.float32) * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return self.mlp_2(nn.silu(self.mlp_0(embedding)))


class EmbeddingLayer(nn.Module):
    def __init__(self, dim: int, vocab_size: int, rngs: nn.Rngs):
        self.embedding = nn.Param(kaiming_uniform(rngs.params(), (vocab_size, dim)))


class DiffusionConditioner(nn.Module):
    """Token and diffusion-time embeddings; bit states ``[B, L, V]`` mix the token rows,
    and ``model.clue_embedding`` adds a learned vector at the anchors."""

    def __init__(self, m: DictConfig, vocab_size: int, rngs: nn.Rngs):
        self.clue_embedding = m.clue_embedding
        self.vocab_embed = EmbeddingLayer(m.hidden_dim, vocab_size, rngs)
        if m.clue_embedding:
            self.clue_embed = nn.Param(jnp.zeros((m.hidden_dim,)))
        self.sigma_map = TimestepEmbedder(m.cond_dim, rngs)

    def __call__(self, tokens, sigma, cond_mask):
        table = self.vocab_embed.embedding.value
        x = table[tokens] if tokens.ndim == 2 else jnp.einsum("bsi,ic->bsc", tokens, table)
        if self.clue_embedding:
            x = x + cond_mask[..., None] * self.clue_embed.value
        return x, nn.silu(self.sigma_map(sigma))


class DDiTBlock(nn.Module):
    """DiT block: adaLN-modulated self-attention and MLP branches."""

    def __init__(self, m: DictConfig, rngs: nn.Rngs):
        dim, self.n_heads = m.hidden_dim, m.num_heads
        self.compute_dtype = jnp.dtype(m.dtype)
        self.adaLN_modulation = nn.Linear(m.cond_dim, 6 * dim, kernel_init=nn.initializers.zeros,
                                          rngs=rngs)
        self.norm1 = LayerNorm(dim, rngs)
        head_dim = dim // self.n_heads
        self.attn_qkv = nn.Param(lecun_normal(rngs.params(), (dim, self.n_heads, 3, head_dim)))
        self.attn_out = nn.Param(lecun_normal(rngs.params(), (self.n_heads, head_dim, dim)))
        self.norm2 = LayerNorm(dim, rngs)
        self.mlp_fc1 = nn.Linear(dim, m.mlp_dim, dtype=self.compute_dtype, rngs=rngs)
        self.mlp_fc2 = nn.Linear(m.mlp_dim, dim, dtype=self.compute_dtype, rngs=rngs)

    def _adaln(self, c):
        out = self.adaLN_modulation(c).astype(self.compute_dtype)
        return jnp.split(out[:, None, :], 6, axis=-1)

    def _attention(self, x, shift, scale, gate, cos=None, sin=None, bias=None, mask=None):
        cdt = self.compute_dtype
        x_mod = modulate(self.norm1(x), shift, scale).astype(cdt)
        # cuDNN attention needs a GPU, half precision, no dense mask and an even length
        # for its backward pass (Sudoku's 81 raises); everything else takes XLA's.
        cudnn = (jax.default_backend() == "gpu" and cdt == jnp.bfloat16 and mask is None
                 and x.shape[1] % 2 == 0)
        qkv = jnp.einsum("bsd,dhqk->bshqk", x_mod, self.attn_qkv.value.astype(cdt))
        q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]
        if cos is not None:
            cos, sin = cos[:x.shape[1]].astype(cdt), sin[:x.shape[1]].astype(cdt)
            q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        out = jax.nn.dot_product_attention(q, k, v, bias=bias, mask=mask,
                                           implementation="cudnn" if cudnn else None)
        out = jnp.einsum("bshd,hdo->bso", out, self.attn_out.value.astype(cdt))
        return x + gate * out

    def _mlp(self, x, shift, scale, gate):
        h = modulate(self.norm2(x), shift, scale).astype(self.compute_dtype)
        return x + gate * self.mlp_fc2(nn.gelu(self.mlp_fc1(h), approximate=True))

    def __call__(self, x, cos, sin, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._adaln(c)
        x = self._attention(x, shift_msa, scale_msa, gate_msa, cos=cos, sin=sin)
        return self._mlp(x, shift_mlp, scale_mlp, gate_mlp)


class DDitFinalLayer(nn.Module):
    """The head: logits of the content classes, padded to a multiple of 8 for its GEMM."""

    def __init__(self, m: DictConfig, num_classes: int, rngs: nn.Rngs):
        self.compute_dtype = jnp.dtype(m.dtype)
        zeros = nn.initializers.zeros
        self.adaLN_modulation = nn.Linear(m.cond_dim, 2 * m.hidden_dim, kernel_init=zeros,
                                          rngs=rngs)
        self.norm_final = LayerNorm(m.hidden_dim, rngs)
        self.linear = nn.Linear(m.hidden_dim, -(-num_classes // 8) * 8, kernel_init=zeros,
                                rngs=rngs)

    def __call__(self, x, c):
        shift, scale = jnp.split(self.adaLN_modulation(c)[:, None, :], 2, axis=-1)
        x = modulate(self.norm_final(x), shift, scale).astype(self.compute_dtype)
        # The head matmul runs in the compute dtype and accumulates fp32 logits.
        return jnp.einsum("bsd,dv->bsv", x, self.linear.kernel.value.astype(self.compute_dtype),
                          preferred_element_type=jnp.float32) + self.linear.bias.value


class DiT(nn.Module):
    """Diffusion transformer over token ids or bit states, attending densely."""

    def __init__(self, cfg: DictConfig, tokenizer, rngs: nn.Rngs):
        m = cfg.model
        self.compute_dtype = jnp.dtype(m.dtype)
        self.conditioner = DiffusionConditioner(m, len(tokenizer), rngs)
        self.rotary_cos, self.rotary_sin = _rotary(m.hidden_dim // m.num_heads,
                                                   cfg.data.max_seq_len)
        self.blocks = nn.List([DDiTBlock(m, rngs) for _ in range(m.num_layers)])
        self.output_layer = DDitFinalLayer(m, tokenizer.num_classes, rngs)

    def __call__(self, input_ids, sigma, *, cond_mask, adjacency):
        x, c = self.conditioner(input_ids, sigma, cond_mask)
        x = x.astype(self.compute_dtype)
        for block in self.blocks:
            x = block(x, self.rotary_cos, self.rotary_sin, c)
        return self.output_layer(x.astype(jnp.float32), c)


class GraphDDiTBlock(DDiTBlock):
    """DiT block with mixed local/global graph attention and relation biases."""

    def __init__(self, m: DictConfig, rngs: nn.Rngs):
        super().__init__(m, rngs)
        # 1-D, so weight decay skips them like the other biases.
        zeros = nn.initializers.zeros
        self.self_bias = nn.Param(zeros(rngs.params(), (self.n_heads,)))
        self.edge_bias = nn.Param(zeros(rngs.params(), (self.n_heads,)))
        self.nonedge_bias = nn.Param(zeros(rngs.params(), (self.n_heads,)))

    def __call__(self, x, c, graph_mask, edge_sel, eye_sel):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self._adaln(c)
        batch, n = edge_sel.shape[0], edge_sel.shape[-1]
        bias = jnp.broadcast_to(self.nonedge_bias.value[None, :, None, None],
                                (batch, self.n_heads, n, n))
        bias = jnp.where(edge_sel, self.edge_bias.value[None, :, None, None], bias)
        bias = jnp.where(eye_sel, self.self_bias.value[None, :, None, None], bias)
        x = self._attention(x, shift_attn, scale_attn, gate_attn,
                            bias=bias.astype(self.compute_dtype), mask=graph_mask)
        return self._mlp(x, shift_mlp, scale_mlp, gate_mlp)


class GraphDiT(nn.Module):
    """Graphormer-style denoiser over color tokens, without positional encoding.

    The bool adjacency ``[B, n, n]`` is the attention structure: the first half of the
    heads see only neighbors and the vertex itself, and every head adds per-head
    self/edge/non-edge biases. Vertex degrees are embedded; nothing else depends on
    ``n``, so a model runs on graphs of any size.
    """

    def __init__(self, cfg: DictConfig, tokenizer, rngs: nn.Rngs):
        m = cfg.model
        self.n, self.n_heads = cfg.data.graphcolor.n, m.num_heads
        self.compute_dtype = jnp.dtype(m.dtype)
        self.conditioner = DiffusionConditioner(m, len(tokenizer), rngs)
        self.degree_embed = EmbeddingLayer(m.hidden_dim, self.n, rngs)
        self.blocks = nn.List([GraphDDiTBlock(m, rngs) for _ in range(m.num_layers)])
        self.output_layer = DDitFinalLayer(m, tokenizer.num_classes, rngs)

    def __call__(self, input_ids, sigma, *, cond_mask, adjacency):
        x, c = self.conditioner(input_ids, sigma, cond_mask)
        # The degree table has graph_n rows; a larger graph clamps degrees past it.
        degrees = jnp.minimum(jnp.sum(adjacency, axis=-1), self.n - 1).astype(jnp.int32)
        x = x + self.degree_embed.embedding.value[degrees]
        batch, n, _ = adjacency.shape
        n_local = self.n_heads // 2
        local = jnp.broadcast_to((adjacency | jnp.eye(n, dtype=jnp.bool_)[None])[:, None],
                                 (batch, n_local, n, n))
        graph_mask = jnp.concatenate(
            [local, jnp.ones((batch, self.n_heads - n_local, n, n), dtype=jnp.bool_)], axis=1)
        eye_sel = jnp.eye(n, dtype=jnp.bool_)[None, None]
        x = x.astype(self.compute_dtype)
        for block in self.blocks:
            x = block(x, c, graph_mask, adjacency[:, None], eye_sel)
        return self.output_layer(x.astype(jnp.float32), c)


def build_model(cfg: DictConfig, tokenizer, rngs: nn.Rngs) -> nn.Module:
    """``model.architecture`` for the run's tokenizer: the embedding covers the whole
    vocabulary, mask included, and the head the content classes."""
    return {"dit": DiT, "graph_dit": GraphDiT}[cfg.model.architecture](cfg, tokenizer, rngs)
