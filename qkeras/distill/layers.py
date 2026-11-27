"""Float transformer building blocks shared by teachers and students.

Every layer has ``get_config`` and ``compute_output_shape`` so models can be cloned and
serialised.  Their quantised counterparts live in ``qkeras.qtransformer`` (``QRMSNorm``,
``QDotProductAttention``, ...); a ``LayerFactory`` (``qkeras.distill.factory``) picks one or
the other, so one builder function yields teacher, float student and QAT student.
"""

from __future__ import annotations

import math

import keras
import numpy as np
from keras import layers, ops

_PKG = "qkeras.distill"


@keras.saving.register_keras_serializable(package=_PKG)
class RMSNorm(layers.Layer):
    """``x * rsqrt(mean(x^2) + eps) * scale`` over the last axis."""

    def __init__(self, epsilon: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.scale = self.add_weight(name="scale", shape=(dim,), initializer="ones", trainable=True)
        super().build(input_shape)

    def call(self, x):
        ms = ops.mean(ops.square(x), axis=-1, keepdims=True)
        return x * ops.rsqrt(ms + self.epsilon) * self.scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "epsilon": self.epsilon}


@keras.saving.register_keras_serializable(package=_PKG)
class AddToken(layers.Layer):
    """Append ``num_tokens`` copies of one learned token along the sequence axis of ``(..., L, D)``."""

    def __init__(self, num_tokens: int, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = int(num_tokens)

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.token = self.add_weight(name="token", shape=(1, 1, dim), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        lead_shape = ops.shape(x)[:-2]
        dim = ops.shape(x)[-1]
        tok = ops.broadcast_to(self.token, tuple(lead_shape) + (self.num_tokens, dim))
        return ops.concatenate([x, tok], axis=-2)

    def compute_output_shape(self, input_shape):
        out = list(input_shape)
        if out[-2] is not None:
            out[-2] = out[-2] + self.num_tokens
        return tuple(out)

    def get_config(self):
        return {**super().get_config(), "num_tokens": self.num_tokens}


def rope_tables(head_dim: int, seq_len: int, fraction: float = 1.0, base: float = 10000.0):
    """``(seq_len, n)`` cos/sin tables for partial rotate-half RoPE, ``n = fraction * head_dim / 2``."""
    half = int(head_dim * fraction) // 2
    inv_freq = 1.0 / (base ** (np.arange(0, half, dtype=np.float32) / max(half, 1)))
    angles = np.arange(seq_len, dtype=np.float32)[:, None] * inv_freq[None, :]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


@keras.saving.register_keras_serializable(package=_PKG)
class Rotary(layers.Layer):
    """Partial rotary positional embedding on ``(B*, S, H, hd)`` (rotate-half convention).

    Dimension ``r`` is paired with ``r + hd/2`` for the first ``n = cos.shape[-1]``
    frequencies; all other dimensions pass through unchanged.
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
        x2 = x[..., half: half + n]
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return ops.concatenate([y1, x[..., n:half], y2, x[..., half + n:]], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "cos": self._cos_np.tolist(), "sin": self._sin_np.tolist()}


@keras.saving.register_keras_serializable(package=_PKG)
class StaticMaskAttention(layers.Layer):
    """Scaled dot-product attention ``(q, k, v) (B, S, H, hd) -> (B, S, H*hd)`` with a fixed
    boolean mask ``(S, S)`` (True = attend)."""

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
        out = ops.transpose(ops.matmul(probs, v_t), (0, 2, 1, 3))
        return ops.reshape(out, (ops.shape(out)[0], ops.shape(out)[1], -1))

    def compute_output_shape(self, q_shape, k_shape=None, v_shape=None):
        return (q_shape[0], q_shape[1], q_shape[2] * q_shape[3])

    def get_config(self):
        return {**super().get_config(), "attention_mask": self._mask_np.tolist()}


def causal_mask(seq_len: int) -> np.ndarray:
    """Lower-triangular ``(S, S)`` mask."""
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


def prefix_query_mask(prefix: int, queries: int) -> np.ndarray:
    """Causal over ``prefix`` tokens; the ``queries`` appended tokens see the whole prefix and
    are causal among themselves (forecast / decoder-query tokens)."""
    mask = causal_mask(prefix + queries)
    mask[prefix:, :prefix] = True
    return mask


# ----------------------------------------------------------------------------- quantile heads
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


@keras.saving.register_keras_serializable(package=_PKG)
class MonotoneQuantiles(layers.Layer):
    """Non-crossing quantiles: ``median -/+ cumsum(width_factor * dp * softplus(raw))``.

    ``dp`` is the probability gap between neighbouring knots, so unequal knot spacing is
    respected and the band width is calibrated.
    """

    def __init__(self, quantile_knots: tuple[float, ...], width_factor: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.quantile_knots = tuple(float(k) for k in quantile_knots)
        self.width_factor = float(width_factor)
        self.median_idx = self.quantile_knots.index(0.5)
        self._widths = quantile_increment_widths(self.quantile_knots, self.width_factor)

    def call(self, raw):
        m = self.median_idx
        med = raw[..., m: m + 1]
        w = ops.convert_to_tensor(self._widths)
        if m > 0:
            lower_deltas = ops.softplus(raw[..., :m]) * w[:m]
            lower = med - ops.flip(ops.cumsum(ops.flip(lower_deltas, axis=-1), axis=-1), axis=-1)
        else:
            lower = raw[..., :0]
        if m + 1 < len(self.quantile_knots):
            upper = med + ops.cumsum(ops.softplus(raw[..., m + 1:]) * w[m + 1:], axis=-1)
        else:
            upper = raw[..., :0]
        return ops.concatenate([lower, med, upper], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "quantile_knots": list(self.quantile_knots),
                "width_factor": self.width_factor}
