"""Keras 3 port of the Timestral forecasting network (univariate, no covariates).

The released PyTorch package (Apache-2.0, github.com/mistralai/timestral) decodes one
block of ``h`` horizon patches after ``L`` context patches.  For a fixed ``(L, h)`` and a
fully observed univariate series every mask and every rotary table is a constant, so the
network becomes a static graph that suits quantisation-aware training and FPGA
compilation:

* host side (``scale_context`` / ``denormalize``): causal per-patch mean/std scaling with
  an ``asinh`` squash, and its inverse;
* network (``build_timestral``): patch embedding -> 8 pre-norm blocks (RMSNorm, sigmoid
  gated causal time attention with partial RoPE, variate attention, SwiGLU MLP) -> final
  RMSNorm -> residual MLP head -> monotone quantiles.

Variate attention over a single variate reduces to ``wO(V(x) * sigmoid(G(x)))`` because the
softmax over one key is 1; it is kept with that exact algebra.

Layer factories (``dense``, ``norm``, ...) are injectable so that the same builder produces
the float model and quantised variants (see ``qkeras.qtransformer``).

The model weights are licensed by Mistral AI under a Modified MIT licence (see the
LICENSE file shipped with the checkpoint); this module contains no weights.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import keras
import numpy as np
from keras import layers, ops


# ----------------------------------------------------------------------------- config
@dataclass
class TimestralSpec:
    patch_size: int = 32
    d_model: int = 192
    num_heads: int = 3
    mlp_hidden_dim: int = 512
    block_schedule: tuple = ("time",) * 3 + ("time_variate",) + ("time",) * 3 + ("time_variate",)
    quantile_knots: tuple = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                             0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)
    rope_theta: float = 10000.0
    rope_partial_factor: float = 0.5
    norm_eps: float = 1e-6
    minimum_scale: float = 1e-6
    use_asinh: bool = True
    quantile_width_factor: float = 3.0
    embed_hidden: int | None = None   # default 4 * d_model
    head_hidden: int | None = None    # default 4 * d_model

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def rope_dim(self) -> int:
        return int(self.head_dim * self.rope_partial_factor)

    @property
    def num_quantiles(self) -> int:
        return len(self.quantile_knots)

    @classmethod
    def from_json(cls, path: str | Path) -> "TimestralSpec":
        c = json.loads(Path(path).read_text())
        keys = cls.__dataclass_fields__.keys()
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items() if k in keys}
        return cls(**kw)


# ----------------------------------------------------------------------------- host side
def scale_context(x: np.ndarray, patch_size: int, minimum_scale: float = 1e-6, use_asinh: bool = True):
    """Causal per-patch scaling of fully observed rows ``x`` ``(B, T)``.

    Returns ``(scaled, anchor_loc, anchor_scale)``; ``anchor_*`` ``(B, 1)`` are the statistics
    of the last context patch, used to map forecasts back (``denormalize``).
    Mirrors ``timestral.scaler.compute_causal_statistics`` (Welford, float64, Bessel).
    """
    x64 = np.asarray(x, dtype=np.float64)
    b, t = x64.shape
    assert t % patch_size == 0
    n = np.arange(1, t + 1, dtype=np.float64)
    cum = np.cumsum(x64, axis=-1)
    means = cum / n
    shifted = np.concatenate([np.zeros((b, 1)), means[:, :-1]], axis=-1)
    m2 = np.cumsum((x64 - shifted) * (x64 - means), axis=-1)
    var = m2 / np.maximum(n - 1.0, 1.0)
    scales = np.maximum(np.sqrt(var), minimum_scale)
    loc = np.repeat(means[:, patch_size - 1 :: patch_size], patch_size, axis=-1).astype(np.float32)
    sc = np.repeat(scales[:, patch_size - 1 :: patch_size], patch_size, axis=-1).astype(np.float32)
    s = (np.asarray(x, np.float32) - loc) / np.maximum(sc, np.finfo(np.float32).eps)
    if use_asinh:
        s = np.arcsinh(s)
    return s.astype(np.float32), loc[:, -1:], sc[:, -1:]


def denormalize(q: np.ndarray, loc: np.ndarray, scale: np.ndarray, use_asinh: bool = True) -> np.ndarray:
    """Normalised quantiles ``(B, H, Q)`` -> raw space with the last context patch's frame."""
    q = np.asarray(q, np.float32)
    if use_asinh:
        q = np.sinh(q)
    return q * scale[..., None] + loc[..., None]


def rope_tables(spec: TimestralSpec, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """``(S, head_dim)`` cos/sin tables for positions ``0..S-1`` (partial rotation)."""
    hd, rd = spec.head_dim, spec.rope_dim
    inv = np.zeros(hd // 2, np.float32)
    idx = np.arange(0, rd, 2, dtype=np.float32)
    inv[: rd // 2] = 1.0 / (spec.rope_theta ** (idx / rd))
    freqs = np.arange(seq_len, dtype=np.float32)[:, None] * inv[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def forecast_mask(context_patches: int, horizon_patches: int) -> np.ndarray:
    """``(S, S)`` boolean mask (True = attend) of one decode block without covariates."""
    L, h = context_patches, horizon_patches
    s = L + h
    m = np.zeros((s, s), bool)
    m[:L, :L] = np.tril(np.ones((L, L), bool))
    m[L:, :L] = True
    m[L:, L:] = np.tril(np.ones((h, h), bool))
    return m


def quantile_increment_widths(knots, width_factor: float) -> np.ndarray:
    m = list(knots).index(0.5)
    w = [0.0] * len(knots)
    for i in range(m + 1, len(knots)):
        w[i] = knots[i] - knots[i - 1]
    for i in range(m):
        w[i] = knots[i + 1] - knots[i]
    return width_factor * np.asarray(w, np.float32)


def monotone_matrix(knots) -> np.ndarray:
    """``(Q, Q)`` matrix ``M`` with ``q = median * 1 + softplus_inc @ M`` (cumulative sums)."""
    q = len(knots)
    m = list(knots).index(0.5)
    M = np.zeros((q, q), np.float32)  # M[i_in, j_out]
    for j in range(q):
        if j > m:
            M[m + 1 : j + 1, j] = 1.0
        elif j < m:
            M[j:m, j] = -1.0
    return M


# ----------------------------------------------------------------------------- layers
@keras.saving.register_keras_serializable(package="timestral")
class RMSNorm(layers.Layer):
    def __init__(self, eps: float = 1e-6, **kw):
        super().__init__(**kw)
        self.eps = eps

    def build(self, shape):
        self.scale = self.add_weight(name="scale", shape=(shape[-1],), initializer="ones")

    def call(self, x):
        r = ops.rsqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + self.eps)
        return x * r * self.scale

    def get_config(self):
        return {**super().get_config(), "eps": self.eps}


@keras.saving.register_keras_serializable(package="timestral")
class AddToken(layers.Layer):
    """Appends ``h`` copies of a learned forecast-query token to the sequence."""

    def __init__(self, n_tokens: int, **kw):
        super().__init__(**kw)
        self.n_tokens = n_tokens

    def build(self, shape):
        self.token = self.add_weight(name="forecast_query", shape=(shape[-1],), initializer="zeros")

    def call(self, x):
        b = ops.shape(x)[0]
        t = ops.broadcast_to(ops.reshape(self.token, (1, 1, -1)), (b, self.n_tokens, ops.shape(x)[-1]))
        return ops.concatenate([x, ops.cast(t, x.dtype)], axis=1)

    def get_config(self):
        return {**super().get_config(), "n_tokens": self.n_tokens}


@keras.saving.register_keras_serializable(package="timestral")
class Rotary(layers.Layer):
    """Applies fixed RoPE tables to ``(B, S, H, D)`` tensors."""

    def __init__(self, cos, sin, **kw):
        super().__init__(**kw)
        self.cos_np = np.asarray(cos, np.float32)
        self.sin_np = np.asarray(sin, np.float32)

    def call(self, x):
        cos = ops.convert_to_tensor(self.cos_np[None, :, None, :])
        sin = ops.convert_to_tensor(self.sin_np[None, :, None, :])
        d = self.cos_np.shape[-1] // 2
        rot = ops.concatenate([-x[..., d:], x[..., :d]], axis=-1)
        return x * cos + rot * sin

    def get_config(self):
        return {**super().get_config(), "cos": self.cos_np.tolist(), "sin": self.sin_np.tolist()}


@keras.saving.register_keras_serializable(package="timestral")
class MaskedSoftmaxAttention(layers.Layer):
    """softmax(QK^T / sqrt(d) + mask) V for ``(B, S, H, D)`` inputs; returns ``(B, S, H*D)``."""

    def __init__(self, mask, **kw):
        super().__init__(**kw)
        self.mask_np = np.asarray(mask, bool)

    def call(self, q, k, v):
        d = ops.shape(q)[-1]
        s = ops.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(int(d))
        neg = np.where(self.mask_np, 0.0, -1e9).astype(np.float32)
        s = s + ops.convert_to_tensor(neg[None, None])
        a = ops.softmax(s, axis=-1)
        o = ops.einsum("bhqk,bkhd->bqhd", a, v)
        return ops.reshape(o, (-1, q.shape[1], q.shape[2] * q.shape[3]))

    def get_config(self):
        return {**super().get_config(), "mask": self.mask_np.tolist()}


# ----------------------------------------------------------------------------- builder
@dataclass
class LayerFactory:
    """Creates the trainable pieces; the default builds the float model."""

    def dense(self, units, use_bias, name, activation=None):
        return layers.Dense(units, use_bias=use_bias, activation=activation, name=name)

    def norm(self, eps, name):
        return RMSNorm(eps, name=name)

    def act(self, kind, name):
        return layers.Activation(kind, name=name)

    def attention(self, mask, name):
        return MaskedSoftmaxAttention(mask, name=name)

    def post(self, name):          # optional activation quantiser after sums
        return None


def _residual_block(x, hidden, out, use_bias, f: LayerFactory, name):
    h = f.dense(hidden, use_bias, f"{name}_hidden")(x)
    h = f.act("silu", f"{name}_silu")(h)
    o = f.dense(out, use_bias, f"{name}_output")(h)
    r = f.dense(out, use_bias, f"{name}_residual")(x)
    y = layers.Add(name=f"{name}_add")([o, r])
    p = f.post(f"{name}_q")
    return p(y) if p is not None else y


def _attention(x, spec, f, cos, sin, mask, is_time, name):
    D, H, hd = spec.d_model, spec.num_heads, spec.head_dim
    qkv = f.dense(3 * D, False, f"{name}_wQKV")(x)
    qkv = layers.Reshape((-1, 3, hd, H), name=f"{name}_qkv_r")(qkv)
    qkv = layers.Permute((1, 2, 4, 3), name=f"{name}_qkv_p")(qkv)  # (B,S,3,H,hd)
    q = layers.Lambda(lambda t: t[:, :, 0], name=f"{name}_q")(qkv)
    k = layers.Lambda(lambda t: t[:, :, 1], name=f"{name}_k")(qkv)
    v = layers.Lambda(lambda t: t[:, :, 2], name=f"{name}_v")(qkv)
    g = f.dense(D, False, f"{name}_gate")(x)
    g = f.act("sigmoid", f"{name}_gate_sig")(g)
    if is_time:
        rot = Rotary(cos, sin, name=f"{name}_rope")
        q, k = rot(q), rot(k)
        o = f.attention(mask, f"{name}_sdpa")(q, k, v)
    else:  # single variate: softmax over one key == 1
        o = layers.Reshape((-1, D), name=f"{name}_v_flat")(v)
    o = layers.Multiply(name=f"{name}_gated")([o, g])
    return f.dense(D, False, f"{name}_wO")(o)


def build_timestral(spec: TimestralSpec, context_patches: int, horizon_patches: int,
                    factory: LayerFactory | None = None, name: str = "timestral") -> keras.Model:
    """Static-shape Timestral: inputs ``scaled`` ``(B, L*P)`` -> quantiles ``(B, h*P, Q)``."""
    f = factory or LayerFactory()
    P, D, L, h = spec.patch_size, spec.d_model, context_patches, horizon_patches
    S = L + h
    cos, sin = rope_tables(spec, S)
    mask = forecast_mask(L, h)
    eh = spec.embed_hidden or 4 * D
    hh = spec.head_hidden or 4 * D

    inp = keras.Input((L * P,), name="scaled")
    vals = layers.Reshape((L, P), name="patches")(inp)
    miss = layers.Lambda(lambda t: ops.zeros_like(t), name="missing")(vals)  # fully observed
    x = layers.Concatenate(axis=-1, name="patch_in")([vals, miss])
    x = _residual_block(x, eh, D, True, f, "embed")
    x = AddToken(h, name="forecast_query")(x)
    for i, kind in enumerate(spec.block_schedule):
        axes = (True,) if kind == "time" else (True, False)
        for j, is_time in enumerate(axes):
            n = f.norm(spec.norm_eps, f"b{i}_attn_norm{j}")(x)
            a = _attention(n, spec, f, cos, sin, mask, is_time, f"b{i}_attn{j}")
            x = layers.Add(name=f"b{i}_res{j}")([x, a])
        n = f.norm(spec.norm_eps, f"b{i}_mlp_norm")(x)
        w12 = f.dense(2 * spec.mlp_hidden_dim, False, f"b{i}_w12")(n)
        gate = layers.Lambda(lambda t: t[..., : t.shape[-1] // 2], name=f"b{i}_gate")(w12)
        up = layers.Lambda(lambda t: t[..., t.shape[-1] // 2 :], name=f"b{i}_up")(w12)
        up = f.act("silu", f"b{i}_silu")(up)
        m = layers.Multiply(name=f"b{i}_swiglu")([gate, up])
        m = f.dense(D, False, f"b{i}_w3")(m)
        x = layers.Add(name=f"b{i}_res_mlp")([x, m])
    x = f.norm(spec.norm_eps, "out_norm")(x)
    x = layers.Lambda(lambda t: t[:, L:], name="decoder_tokens")(x)
    raw = _residual_block(x, hh, P * spec.num_quantiles, False, f, "head")
    raw = layers.Reshape((h * P, spec.num_quantiles), name="raw_quantiles")(raw)
    out = MonotoneQuantiles(spec.quantile_knots, spec.quantile_width_factor, name="quantiles")(raw)
    return keras.Model(inp, out, name=name)


@keras.saving.register_keras_serializable(package="timestral")
class MonotoneQuantiles(layers.Layer):
    """``median -/+ cumsum(width * softplus(raw))`` as a fixed matrix product."""

    def __init__(self, knots, width_factor, **kw):
        super().__init__(**kw)
        self.knots = tuple(knots)
        self.width_factor = float(width_factor)

    def call(self, raw):
        knots = self.knots
        m = knots.index(0.5)
        w = quantile_increment_widths(knots, self.width_factor)
        M = monotone_matrix(knots)
        inc = ops.softplus(raw) * ops.convert_to_tensor(w)
        med = raw[..., m : m + 1]
        return med + ops.matmul(inc, ops.convert_to_tensor(M))

    def get_config(self):
        return {**super().get_config(), "knots": list(self.knots), "width_factor": self.width_factor}


# ----------------------------------------------------------------------------- weights
def load_timestral_weights(model: keras.Model, weights: dict[str, np.ndarray], spec: TimestralSpec) -> None:
    """Copy released PyTorch weights (``safetensors`` dict) into ``model``."""

    def setw(name, *arrs):
        lyr = model.get_layer(name)
        cur = lyr.get_weights()
        assert len(cur) == len(arrs), (name, [c.shape for c in cur], [a.shape for a in arrs])
        for c, a in zip(cur, arrs):
            assert c.shape == a.shape, (name, c.shape, a.shape)
        lyr.set_weights([np.asarray(a, np.float32) for a in arrs])

    def lin(name, key, bias=True):
        wt = weights[f"{key}.weight"].T
        if bias and f"{key}.bias" in weights:
            setw(name, wt, weights[f"{key}.bias"])
        else:
            setw(name, wt)

    rp = "representation.patch_proj"
    for part in ("hidden", "output", "residual"):
        lin(f"embed_{part}", f"{rp}.{part}_layer")
    setw("forecast_query", weights["representation.forecast_query"])
    for i, kind in enumerate(spec.block_schedule):
        tp = f"transformer.layers.{i}"
        nax = 1 if kind == "time" else 2
        for j in range(nax):
            setw(f"b{i}_attn_norm{j}", weights[f"{tp}.attn_norms.{j}.scale"])
            lin(f"b{i}_attn{j}_wQKV", f"{tp}.attns.{j}.wQKV", bias=False)
            lin(f"b{i}_attn{j}_gate", f"{tp}.attns.{j}._gate", bias=False)
            lin(f"b{i}_attn{j}_wO", f"{tp}.attns.{j}.wO", bias=False)
        setw(f"b{i}_mlp_norm", weights[f"{tp}.mlp_norm.scale"])
        lin(f"b{i}_w12", f"{tp}.mlp.w12", bias=False)
        lin(f"b{i}_w3", f"{tp}.mlp.w3", bias=False)
    setw("out_norm", weights["transformer.out_norm.scale"])
    for part in ("hidden", "output", "residual"):
        lin(f"head_{part}", f"head.projection.{part}_layer", bias=False)


def forecast(model: keras.Model, context: np.ndarray, spec: TimestralSpec) -> np.ndarray:
    """Raw-space quantiles ``(B, h*P, Q)`` for fully observed contexts ``(B, L*P)``."""
    s, loc, sc = scale_context(context, spec.patch_size, spec.minimum_scale, spec.use_asinh)
    q = model.predict(s, verbose=0)
    return denormalize(q, loc, sc, spec.use_asinh)
