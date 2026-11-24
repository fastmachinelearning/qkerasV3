from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import keras
from keras import layers, ops
import numpy as np


# -----------------------------------------------------------------------------
# Model Specification
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TimestralSpec:
    patch_size: int = 32
    d_model: int = 192
    num_heads: int = 4
    mlp_hidden_dim: int = 512
    num_blocks: int = 8
    block_schedule: tuple[str, ...] = (
        "time",
        "time",
        "time",
        "time_variate",
        "time",
        "time",
        "time",
        "time_variate",
    )
    rope_base: float = 10000.0
    rope_fraction: float = 0.5
    norm_eps: float = 1e-6
    minimum_scale: float = 1e-6
    use_asinh: bool = True
    quantile_knots: tuple[float, ...] = (
        0.01,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        0.99,
    )
    quantile_width_factor: float = 0.05
    embed_hidden: int | None = None
    head_hidden: int | None = None

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def num_quantiles(self) -> int:
        return len(self.quantile_knots)

    @classmethod
    def from_json(cls, path: str | Path) -> TimestralSpec:
        cfg = json.loads(Path(path).read_text())
        q_knots = tuple(cfg.get("quantile_knots", cls.quantile_knots))
        schedule = tuple(cfg.get("block_schedule", cls.block_schedule))
        return cls(
            patch_size=int(cfg.get("patch_size", cls.patch_size)),
            d_model=int(cfg.get("d_model", cls.d_model)),
            num_heads=int(cfg.get("num_heads", cls.num_heads)),
            mlp_hidden_dim=int(cfg.get("mlp_hidden_dim", cls.mlp_hidden_dim)),
            num_blocks=int(cfg.get("num_blocks", len(schedule))),
            block_schedule=schedule,
            rope_base=float(cfg.get("rope_theta", cfg.get("rope_base", cls.rope_base))),
            rope_fraction=float(
                cfg.get("rope_partial_factor", cfg.get("rope_fraction", cls.rope_fraction))
            ),
            norm_eps=float(cfg.get("norm_eps", cls.norm_eps)),
            minimum_scale=float(cfg.get("minimum_scale", cls.minimum_scale)),
            use_asinh=bool(cfg.get("use_asinh", cls.use_asinh)),
            quantile_knots=q_knots,
            quantile_width_factor=float(
                cfg.get("quantile_width_factor", cls.quantile_width_factor)
            ),
            embed_hidden=cfg.get("embed_hidden", None),
            head_hidden=cfg.get("head_hidden", None),
        )


# -----------------------------------------------------------------------------
# Layer Factory (Standard Float32 default; override for QKeras)
# -----------------------------------------------------------------------------
class LayerFactory:
    def dense(self, units: int, use_bias: bool, name: str, activation=None):
        return layers.Dense(units, use_bias=use_bias, activation=activation, name=name)

    def norm(self, eps: float, name: str):
        return RMSNorm(epsilon=eps, name=name)

    def act(self, kind: str, name: str):
        return layers.Activation(kind, name=name)

    def attention(self, mask: np.ndarray, name: str):
        return StaticMaskAttention(attention_mask=mask, name=name)

    def post(self, name: str):
        return None


# -----------------------------------------------------------------------------
# Custom Layers (with explicit serialization + compute_output_shape for cloning)
# -----------------------------------------------------------------------------
class RMSNorm(layers.Layer):
    def __init__(self, epsilon: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.scale = self.add_weight(
            name="scale", shape=(dim,), initializer="ones", trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        ms = ops.mean(ops.square(x), axis=-1, keepdims=True)
        return x * ops.rsqrt(ms + self.epsilon) * self.scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg["epsilon"] = self.epsilon
        return cfg


class AddToken(layers.Layer):
    """Append ``num_tokens`` learned query tokens along the sequence axis of ``(..., L, D)``."""

    def __init__(self, num_tokens: int, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = int(num_tokens)

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.token = self.add_weight(
            name="token", shape=(1, 1, dim), initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        lead_shape = ops.shape(x)[:-2]
        dim = ops.shape(x)[-1]
        tok_shape = tuple(lead_shape) + (self.num_tokens, dim)
        tok = ops.broadcast_to(self.token, tok_shape)
        return ops.concatenate([x, tok], axis=-2)

    def compute_output_shape(self, input_shape):
        out = list(input_shape)
        if out[-2] is not None:
            out[-2] = out[-2] + self.num_tokens
        return tuple(out)

    def get_config(self):
        cfg = super().get_config()
        cfg["num_tokens"] = self.num_tokens
        return cfg


class Rotary(layers.Layer):
    """Partial rotary positional embedding on ``(B*, S, H, hd)``.

    Uses the "rotate-half" convention of the released Timestral weights: dimension ``r``
    is paired with ``r + hd/2`` for the first ``n = cos.shape[-1]`` frequencies
    (``n = rope_fraction * hd / 2``); all other dimensions pass through unchanged.
    """

    def __init__(self, cos: np.ndarray, sin: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self._cos_np = np.asarray(cos, dtype=np.float32)
        self._sin_np = np.asarray(sin, dtype=np.float32)

    def call(self, x):
        n = self._cos_np.shape[-1]
        half = x.shape[-1] // 2
        cos = ops.convert_to_tensor(self._cos_np)[None, :, None, :]
        sin = ops.convert_to_tensor(self._sin_np)[None, :, None, :]
        x1 = x[..., :n]
        x2 = x[..., half : half + n]
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return ops.concatenate([y1, x[..., n:half], y2, x[..., half + n :]], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg["cos"] = self._cos_np.tolist()
        cfg["sin"] = self._sin_np.tolist()
        return cfg


class StaticMaskAttention(layers.Layer):
    """Scaled dot-product attention with a fixed boolean mask ``(S, S)``."""

    def __init__(self, attention_mask: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self._mask_np = np.asarray(attention_mask, dtype=bool)

    def call(self, q, k, v):
        q_t = ops.transpose(q, (0, 2, 1, 3))
        k_t = ops.transpose(k, (0, 2, 1, 3))
        v_t = ops.transpose(v, (0, 2, 1, 3))
        scale = 1.0 / math.sqrt(float(q.shape[-1]))
        scores = ops.matmul(q_t, ops.transpose(k_t, (0, 1, 3, 2))) * scale
        mask = ops.convert_to_tensor(self._mask_np)[None, None, :, :]
        scores = ops.where(mask, scores, ops.convert_to_tensor(-1e9, dtype=scores.dtype))
        probs = ops.softmax(scores, axis=-1)
        out = ops.matmul(probs, v_t)
        out = ops.transpose(out, (0, 2, 1, 3))
        return ops.reshape(out, (ops.shape(out)[0], ops.shape(out)[1], -1))

    def compute_output_shape(self, q_shape, k_shape=None, v_shape=None):
        return (q_shape[0], q_shape[1], q_shape[2] * q_shape[3])

    def get_config(self):
        cfg = super().get_config()
        cfg["attention_mask"] = self._mask_np.tolist()
        return cfg


def quantile_increment_widths(quantile_knots, width_factor: float) -> np.ndarray:
    """``width_factor * dp`` per knot, ``dp`` = probability gap to the next knot towards 0.5."""
    k = [float(v) for v in quantile_knots]
    m = k.index(0.5)
    w = np.zeros(len(k), dtype=np.float32)
    for i in range(m + 1, len(k)):
        w[i] = k[i] - k[i - 1]
    for i in range(m):
        w[i] = k[i + 1] - k[i]
    return (width_factor * w).astype(np.float32)


class MonotoneQuantiles(layers.Layer):
    """Non-crossing quantiles: ``median -/+ cumsum(width_factor * dp * softplus(raw))``.

    ``dp`` is the probability gap between neighbouring knots (as in the released model),
    so unequal knot spacing is respected and the band width is calibrated.
    """

    def __init__(
        self,
        quantile_knots: tuple[float, ...],
        width_factor: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.quantile_knots = tuple(float(k) for k in quantile_knots)
        self.width_factor = float(width_factor)
        self.median_idx = self.quantile_knots.index(0.5)
        self._widths = quantile_increment_widths(self.quantile_knots, self.width_factor)

    def call(self, raw):
        m = self.median_idx
        med = raw[..., m : m + 1]
        w = ops.convert_to_tensor(self._widths)
        if m > 0:
            lower_deltas = ops.softplus(raw[..., :m]) * w[:m]
            lower = med - ops.flip(ops.cumsum(ops.flip(lower_deltas, axis=-1), axis=-1), axis=-1)
        else:
            lower = raw[..., :0]
        if m + 1 < len(self.quantile_knots):
            upper_deltas = ops.softplus(raw[..., m + 1 :]) * w[m + 1 :]
            upper = med + ops.cumsum(upper_deltas, axis=-1)
        else:
            upper = raw[..., :0]
        return ops.concatenate([lower, med, upper], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg["quantile_knots"] = list(self.quantile_knots)
        cfg["width_factor"] = self.width_factor
        return cfg


# -----------------------------------------------------------------------------
# Tables, Masks & Blocks
# -----------------------------------------------------------------------------
def rope_tables(spec: TimestralSpec, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    r_dim = int(spec.head_dim * spec.rope_fraction)
    half = r_dim // 2
    inv_freq = 1.0 / (
        spec.rope_base ** (np.arange(0, half, dtype=np.float32) / max(half, 1))
    )
    pos = np.arange(seq_len, dtype=np.float32)[:, None]
    angles = pos * inv_freq[None, :]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def forecast_mask(context_patches: int, horizon_patches: int) -> np.ndarray:
    """Causal mask across context patches; forecast tokens attend to all context + causal forecast."""
    s = context_patches + horizon_patches
    mask = np.tril(np.ones((s, s), dtype=bool))
    mask[context_patches:, :context_patches] = True
    return mask


def _residual_block(x, hidden_dim: int, out_dim: int, use_bias: bool, f: LayerFactory, name: str):
    h = f.dense(hidden_dim, use_bias, f"{name}_hidden")(x)
    h = f.act("silu", f"{name}_silu")(h)
    h = f.dense(out_dim, use_bias, f"{name}_output")(h)
    res = f.dense(out_dim, use_bias, f"{name}_residual")(x)
    out = layers.Add(name=f"{name}_add")([h, res])
    post = f.post(f"{name}_post")
    return post(out) if post is not None else out


# -----------------------------------------------------------------------------
# Univariate Builder
# -----------------------------------------------------------------------------
def _attention_1d(x, spec: TimestralSpec, f: LayerFactory, cos, sin, mask, seq_len: int, is_time: bool, name: str):
    D, H, hd = spec.d_model, spec.num_heads, spec.head_dim
    qkv = f.dense(3 * D, False, f"{name}_wQKV")(x)
    qkv = layers.Reshape((seq_len, 3, hd, H), name=f"{name}_qkv_r")(qkv)
    qkv = layers.Permute((1, 2, 4, 3), name=f"{name}_qkv_p")(qkv)
    q = layers.Lambda(lambda t: t[:, :, 0], output_shape=(seq_len, H, hd), name=f"{name}_q")(qkv)
    k = layers.Lambda(lambda t: t[:, :, 1], output_shape=(seq_len, H, hd), name=f"{name}_k")(qkv)
    v = layers.Lambda(lambda t: t[:, :, 2], output_shape=(seq_len, H, hd), name=f"{name}_v")(qkv)

    g = f.dense(D, False, f"{name}_gate")(x)
    g = f.act("sigmoid", f"{name}_gate_sig")(g)

    if is_time:
        rot = Rotary(cos, sin, name=f"{name}_rope")
        q, k = rot(q), rot(k)
        o = f.attention(mask, f"{name}_sdpa")(q, k, v)
    else:
        o = layers.Reshape((seq_len, D), name=f"{name}_v_flat")(v)

    o = layers.Multiply(name=f"{name}_gated")([o, g])
    o = f.dense(D, False, f"{name}_wO")(o)
    post = f.post(f"{name}_post")
    return post(o) if post is not None else o


def build_timestral(
    spec: TimestralSpec,
    context_patches: int,
    horizon_patches: int,
    factory: LayerFactory | None = None,
    name: str = "timestral",
) -> keras.Model:
    """Static-shape Univariate Timestral: ``(B, L*P)`` -> ``(B, h*P, Q)``."""
    f = factory or LayerFactory()
    P, D, L, h = spec.patch_size, spec.d_model, context_patches, horizon_patches
    S = L + h
    cos, sin = rope_tables(spec, S)
    mask = forecast_mask(L, h)
    eh = spec.embed_hidden or 4 * D
    hh = spec.head_hidden or 4 * D

    inp = keras.Input((L * P,), name="scaled")
    vals = layers.Reshape((L, P), name="patches")(inp)
    miss = layers.Lambda(lambda t: ops.zeros_like(t), output_shape=(L, P), name="missing")(vals)
    x = layers.Concatenate(axis=-1, name="patch_in")([vals, miss])
    x = _residual_block(x, eh, D, True, f, "embed")
    x = AddToken(h, name="forecast_query")(x)

    for i, kind in enumerate(spec.block_schedule):
        axes = (True,) if kind == "time" else (True, False)
        for j, is_time in enumerate(axes):
            n = f.norm(spec.norm_eps, f"b{i}_attn_norm{j}")(x)
            a = _attention_1d(n, spec, f, cos, sin, mask, S, is_time, f"b{i}_attn{j}")
            x = layers.Add(name=f"b{i}_res{j}")([x, a])
        n = f.norm(spec.norm_eps, f"b{i}_mlp_norm")(x)
        w12 = f.dense(2 * spec.mlp_hidden_dim, False, f"b{i}_w12")(n)
        gate = layers.Lambda(
            lambda t: t[..., : t.shape[-1] // 2],
            output_shape=(S, spec.mlp_hidden_dim),
            name=f"b{i}_gate",
        )(w12)
        up = layers.Lambda(
            lambda t: t[..., t.shape[-1] // 2 :],
            output_shape=(S, spec.mlp_hidden_dim),
            name=f"b{i}_up",
        )(w12)
        up = f.act("silu", f"b{i}_silu")(up)
        m = layers.Multiply(name=f"b{i}_swiglu")([gate, up])
        m = f.dense(D, False, f"b{i}_w3")(m)
        post_m = f.post(f"b{i}_mlp_post")
        if post_m is not None:
            m = post_m(m)
        x = layers.Add(name=f"b{i}_res_mlp")([x, m])

    x = f.norm(spec.norm_eps, "out_norm")(x)
    x = layers.Lambda(lambda t: t[:, L:, :], output_shape=(h, D), name="decoder_tokens")(x)
    raw = _residual_block(x, hh, P * spec.num_quantiles, False, f, "head")
    raw = layers.Reshape((h * P, spec.num_quantiles), name="raw_quantiles")(raw)
    out = MonotoneQuantiles(spec.quantile_knots, spec.quantile_width_factor, name="quantiles")(raw)
    return keras.Model(inp, out, name=name)


# -----------------------------------------------------------------------------
# Multivariate Builder
# -----------------------------------------------------------------------------
def _attention_mv(
    x,
    spec: TimestralSpec,
    f: LayerFactory,
    cos,
    sin,
    time_mask,
    num_variates: int,
    seq_len: int,
    is_time: bool,
    name: str,
):
    D, H, hd = spec.d_model, spec.num_heads, spec.head_dim
    V, S = num_variates, seq_len

    if is_time:
        # Fold (B, V, S, D) -> (B*V, S, D)
        x_2d = layers.Lambda(
            lambda t: ops.reshape(t, (-1, S, D)),
            output_shape=(S, D),
            name=f"{name}_in_flat",
        )(x)
        axis_len = S
    else:
        # Permute (B, V, S, D) -> (B, S, V, D), then fold -> (B*S, V, D)
        x_perm = layers.Permute((2, 1, 3), name=f"{name}_perm_in")(x)
        x_2d = layers.Lambda(
            lambda t: ops.reshape(t, (-1, V, D)),
            output_shape=(V, D),
            name=f"{name}_in_flat",
        )(x_perm)
        axis_len = V

    qkv = f.dense(3 * D, False, f"{name}_wQKV")(x_2d)
    qkv = layers.Reshape((axis_len, 3, hd, H), name=f"{name}_qkv_r")(qkv)
    qkv = layers.Permute((1, 2, 4, 3), name=f"{name}_qkv_p")(qkv)
    q = layers.Lambda(lambda t: t[:, :, 0], output_shape=(axis_len, H, hd), name=f"{name}_q")(qkv)
    k = layers.Lambda(lambda t: t[:, :, 1], output_shape=(axis_len, H, hd), name=f"{name}_k")(qkv)
    v = layers.Lambda(lambda t: t[:, :, 2], output_shape=(axis_len, H, hd), name=f"{name}_v")(qkv)

    g = f.dense(D, False, f"{name}_gate")(x_2d)
    g = f.act("sigmoid", f"{name}_gate_sig")(g)

    if is_time:
        rot = Rotary(cos, sin, name=f"{name}_rope")
        q, k = rot(q), rot(k)
        o = f.attention(time_mask, f"{name}_sdpa")(q, k, v)
    else:
        variate_mask = np.ones((V, V), dtype=bool)
        o = f.attention(variate_mask, f"{name}_sdpa")(q, k, v)

    o = layers.Multiply(name=f"{name}_gated")([o, g])
    o = f.dense(D, False, f"{name}_wO")(o)
    post = f.post(f"{name}_post")
    if post is not None:
        o = post(o)

    if is_time:
        return layers.Lambda(
            lambda t: ops.reshape(t, (-1, V, S, D)),
            output_shape=(V, S, D),
            name=f"{name}_out_unflat",
        )(o)
    o_4d = layers.Lambda(
        lambda t: ops.reshape(t, (-1, S, V, D)),
        output_shape=(S, V, D),
        name=f"{name}_out_unflat",
    )(o)
    return layers.Permute((2, 1, 3), name=f"{name}_perm_out")(o_4d)


def build_timestral_multivariate(
    spec: TimestralSpec,
    num_variates: int,
    context_patches: int,
    horizon_patches: int,
    factory: LayerFactory | None = None,
    name: str = "timestral_mv",
) -> keras.Model:
    """Static-shape Multivariate Timestral: ``(B, V, L*P)`` -> ``(B, V, h*P, Q)``."""
    f = factory or LayerFactory()
    P, D, V, L, h = spec.patch_size, spec.d_model, num_variates, context_patches, horizon_patches
    S = L + h
    cos, sin = rope_tables(spec, S)
    time_mask = forecast_mask(L, h)
    eh = spec.embed_hidden or 4 * D
    hh = spec.head_hidden or 4 * D

    inp = keras.Input((V, L * P), name="scaled")
    vals = layers.Reshape((V, L, P), name="patches")(inp)
    miss = layers.Lambda(lambda t: ops.zeros_like(t), output_shape=(V, L, P), name="missing")(vals)
    x = layers.Concatenate(axis=-1, name="patch_in")([vals, miss])
    x = _residual_block(x, eh, D, True, f, "embed")

    x = AddToken(h, name="forecast_query")(x)  # (B, V, L, D) -> (B, V, S, D)

    for i, kind in enumerate(spec.block_schedule):
        axes = (True,) if kind == "time" else (True, False)
        for j, is_time in enumerate(axes):
            n = f.norm(spec.norm_eps, f"b{i}_attn_norm{j}")(x)
            a = _attention_mv(n, spec, f, cos, sin, time_mask, V, S, is_time, f"b{i}_attn{j}")
            x = layers.Add(name=f"b{i}_res{j}")([x, a])
        n = f.norm(spec.norm_eps, f"b{i}_mlp_norm")(x)
        w12 = f.dense(2 * spec.mlp_hidden_dim, False, f"b{i}_w12")(n)
        gate = layers.Lambda(
            lambda t: t[..., : t.shape[-1] // 2],
            output_shape=(V, S, spec.mlp_hidden_dim),
            name=f"b{i}_gate",
        )(w12)
        up = layers.Lambda(
            lambda t: t[..., t.shape[-1] // 2 :],
            output_shape=(V, S, spec.mlp_hidden_dim),
            name=f"b{i}_up",
        )(w12)
        up = f.act("silu", f"b{i}_silu")(up)
        m = layers.Multiply(name=f"b{i}_swiglu")([gate, up])
        m = f.dense(D, False, f"b{i}_w3")(m)
        post_m = f.post(f"b{i}_mlp_post")
        if post_m is not None:
            m = post_m(m)
        x = layers.Add(name=f"b{i}_res_mlp")([x, m])

    x = f.norm(spec.norm_eps, "out_norm")(x)
    x = layers.Lambda(lambda t: t[:, :, L:, :], output_shape=(V, h, D), name="decoder_tokens")(x)
    raw = _residual_block(x, hh, P * spec.num_quantiles, False, f, "head")
    raw = layers.Reshape((V, h * P, spec.num_quantiles), name="raw_quantiles")(raw)
    out = MonotoneQuantiles(spec.quantile_knots, spec.quantile_width_factor, name="quantiles")(raw)
    return keras.Model(inp, out, name=name)


# -----------------------------------------------------------------------------
# Weight Loading (works for both float and prunable QKeras models)
# -----------------------------------------------------------------------------
def load_timestral_weights(
    model: keras.Model, weights: dict[str, np.ndarray], spec: TimestralSpec
) -> None:
    """Load safetensors weights directly into kernel/bias/scale/token variables."""

    def setw(name: str, kernel_or_scale: np.ndarray, bias: np.ndarray | None = None):
        lyr = model.get_layer(name)
        if hasattr(lyr, "kernel"):
            assert lyr.kernel.shape == kernel_or_scale.shape, (
                name,
                lyr.kernel.shape,
                kernel_or_scale.shape,
            )
            lyr.kernel.assign(np.asarray(kernel_or_scale, np.float32))
            if bias is not None:
                assert lyr.bias is not None and lyr.bias.shape == bias.shape
                lyr.bias.assign(np.asarray(bias, np.float32))
        else:
            target = lyr.scale if hasattr(lyr, "scale") else lyr.token
            arr = np.asarray(kernel_or_scale, np.float32)
            if arr.ndim != len(target.shape) and arr.size == int(np.prod(target.shape)):
                arr = arr.reshape(tuple(target.shape))
            assert tuple(target.shape) == arr.shape, (
                name,
                target.shape,
                arr.shape,
            )
            target.assign(arr)

    def lin(name: str, key: str, bias: bool = True):
        wt = weights[f"{key}.weight"].T
        b = weights[f"{key}.bias"] if (bias and f"{key}.bias" in weights) else None
        setw(name, wt, b)

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


# -----------------------------------------------------------------------------
# Causal Scaling, Denormalization & Forecasting
# -----------------------------------------------------------------------------
def scale_context(
    x: np.ndarray, patch_size: int, minimum_scale: float = 1e-6, use_asinh: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal per-patch scaling of univariate rows ``x`` ``(B, L*P)``.

    Every patch is normalised with the mean and (Bessel-corrected) std of all values up
    to the end of that patch, computed in float64 in Welford form exactly like
    ``timestral.scaler.compute_causal_statistics``.  float32 ``E[x^2] - mean^2`` loses the
    variance of near-constant signals at a large level (e.g. a temperature of 980 or a
    lambda of 1.1), which collapses the scale to ``minimum_scale``.
    Returns ``(scaled, anchor_loc, anchor_scale)``; the anchors (last patch) are ``(B, 1)``.
    """
    x64 = np.asarray(x, dtype=np.float64)
    b, t = x64.shape
    assert t % patch_size == 0, f"length {t} is not a multiple of patch_size {patch_size}"
    n = np.arange(1, t + 1, dtype=np.float64)
    means = np.cumsum(x64, axis=-1) / n
    shifted = np.concatenate([np.zeros((b, 1)), means[:, :-1]], axis=-1)
    m2 = np.cumsum((x64 - shifted) * (x64 - means), axis=-1)
    scales = np.maximum(np.sqrt(m2 / np.maximum(n - 1.0, 1.0)), minimum_scale)
    loc = np.repeat(means[:, patch_size - 1 :: patch_size], patch_size, axis=-1)
    scl = np.repeat(scales[:, patch_size - 1 :: patch_size], patch_size, axis=-1)
    scaled = (x64 - loc) / scl
    if use_asinh:
        scaled = np.arcsinh(scaled)
    return (
        scaled.astype(np.float32),
        loc[:, -1:].astype(np.float32),
        scl[:, -1:].astype(np.float32),
    )


def denormalize(
    q: np.ndarray, loc: np.ndarray, scale: np.ndarray, use_asinh: bool = True
) -> np.ndarray:
    """Normalised quantiles ``(B, H, Q)`` -> raw space ``(B, H, Q)``."""
    q = np.asarray(q, dtype=np.float32)
    if use_asinh:
        q = np.sinh(q)
    return q * scale[..., None] + loc[..., None]


def forecast(model: keras.Model, context: np.ndarray, spec: TimestralSpec) -> np.ndarray:
    """Raw-space quantiles ``(B, h*P, Q)`` for univariate contexts ``(B, L*P)``."""
    s, loc, sc = scale_context(context, spec.patch_size, spec.minimum_scale, spec.use_asinh)
    q = model(s, training=False).numpy()
    return denormalize(q, loc, sc, spec.use_asinh)


def scale_context_mv(
    x: np.ndarray, patch_size: int, minimum_scale: float = 1e-6, use_asinh: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal per-patch scaling of multivariate rows ``x`` ``(B, V, L*P)``."""
    x = np.asarray(x, dtype=np.float32)
    b, v, t = x.shape
    s_flat, loc_flat, sc_flat = scale_context(
        x.reshape(b * v, t), patch_size, minimum_scale, use_asinh
    )
    return s_flat.reshape(b, v, t), loc_flat.reshape(b, v, 1), sc_flat.reshape(b, v, 1)


def denormalize_mv(
    q: np.ndarray, loc: np.ndarray, scale: np.ndarray, use_asinh: bool = True
) -> np.ndarray:
    """Normalised quantiles ``(B, V, H, Q)`` -> raw space ``(B, V, H, Q)``."""
    q = np.asarray(q, dtype=np.float32)
    if use_asinh:
        q = np.sinh(q)
    return q * scale[..., None] + loc[..., None]


def forecast_mv(model: keras.Model, context: np.ndarray, spec: TimestralSpec) -> np.ndarray:
    """Raw-space quantiles ``(B, V, h*P, Q)`` for multivariate contexts ``(B, V, L*P)``."""
    s, loc, sc = scale_context_mv(context, spec.patch_size, spec.minimum_scale, spec.use_asinh)
    q = model(s, training=False).numpy()
    return denormalize_mv(q, loc, sc, spec.use_asinh)


def normalize_horizon_mv(
    y: np.ndarray, loc: np.ndarray, scale: np.ndarray, use_asinh: bool = True
) -> np.ndarray:
    """Raw-space horizon ``(B, V, H)`` -> normalized space matching model output."""
    y = np.asarray(y, dtype=np.float32)
    normed = (y - loc) / scale
    if use_asinh:
        normed = np.arcsinh(normed)
    return normed.astype(np.float32)