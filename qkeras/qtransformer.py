# Copyright 2026 The QKeras Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Quantised building blocks for Transformer models.

The layers decompose attention the same way hls4ml lowers it for FPGAs: projections are
(Q)EinsumDense/QDense layers, scores and the value mix are einsums, and every
non-linearity (softmax exponent and reciprocal, sigmoid, SiLU, rsqrt, ...) is applied to
a quantised input and produces a quantised output, i.e. it can be realised as a lookup
table indexed by the input bits.

All quantizers are optional; with ``None`` a layer reproduces its float counterpart
exactly, which makes post-training quantisation and quantisation-aware fine-tuning of a
pretrained float model a matter of setting quantizer strings.

Layers
------
QEinsumDense          EinsumDense with kernel/bias quantizers.
QUnary                element-wise function with input/output quantizers (LUT semantics).
QRMSNorm              RMS normalisation with quantised mean-square, rsqrt, gain and output.
QSoftmax              max-subtracted softmax with quantised exp, sum reciprocal and output.
QRotary               fixed rotary position embedding (partial rotation supported).
QDotProductAttention  scaled dot-product attention with a static mask.
"""

import math

import keras
import numpy as np
from keras import layers, ops

from .quantizers import get_quantizer
from .safe_eval import safe_eval  # noqa: F401  (kept for parity with other modules)

_UNARY = {
    "sigmoid": lambda x: ops.sigmoid(x),
    "silu": lambda x: x * ops.sigmoid(x),
    "softplus": lambda x: ops.softplus(x),
    "exp": lambda x: ops.exp(x),
    "rsqrt": lambda x: ops.rsqrt(x),
    "reciprocal": lambda x: 1.0 / x,
    "tanh": lambda x: ops.tanh(x),
    "gelu": lambda x: keras.activations.gelu(x, approximate=False),
}


def _q(quantizer):
    return get_quantizer(quantizer) if quantizer is not None else None


def _apply(q, x):
    return q(x) if q is not None else x


def _ser(q):
    return keras.saving.serialize_keras_object(q) if q is not None else None


class QEinsumDense(layers.EinsumDense):
    """``keras.layers.EinsumDense`` with quantised kernel and bias."""

    def __init__(self, equation, output_shape, activation=None, bias_axes=None,
                 kernel_quantizer=None, bias_quantizer=None, **kwargs):
        self.kernel_quantizer = kernel_quantizer
        self.bias_quantizer = bias_quantizer
        self.kernel_quantizer_internal = _q(kernel_quantizer)
        self.bias_quantizer_internal = _q(bias_quantizer)
        self.quantizers = [self.kernel_quantizer_internal, self.bias_quantizer_internal]
        if activation is not None and not callable(activation):
            activation = get_quantizer(activation) if "quantized" in str(activation) else activation
        super().__init__(equation, output_shape, activation=activation, bias_axes=bias_axes, **kwargs)

    def call(self, inputs, training=None):
        kernel = _apply(self.kernel_quantizer_internal, self.kernel)
        x = ops.einsum(self.equation, inputs, kernel)
        if self.bias is not None:
            x = x + _apply(self.bias_quantizer_internal, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update(kernel_quantizer=_ser(self.kernel_quantizer_internal),
                   bias_quantizer=_ser(self.bias_quantizer_internal))
        return cfg

    def get_quantizers(self):
        return self.quantizers

    def get_prunable_weights(self):
        return [self.kernel]


class QUnary(layers.Layer):
    """``output_quantizer(fn(input_quantizer(x)))`` for a named element-wise function.

    With both quantizers set the layer is exactly a lookup table indexed by the input
    bits, which is how hls4ml implements it (``UnaryLUT``).
    """

    def __init__(self, function, input_quantizer=None, output_quantizer=None, epsilon=0.0, **kwargs):
        super().__init__(**kwargs)
        if function not in _UNARY:
            raise ValueError(f"unsupported function {function!r}; choose from {sorted(_UNARY)}")
        self.function = function
        self.epsilon = float(epsilon)
        self.input_quantizer = input_quantizer
        self.output_quantizer = output_quantizer
        self.input_quantizer_internal = _q(input_quantizer)
        self.output_quantizer_internal = _q(output_quantizer)
        self.quantizers = [self.input_quantizer_internal, self.output_quantizer_internal]

    def call(self, x):
        x = _apply(self.input_quantizer_internal, x)
        if self.epsilon:
            x = x + self.epsilon
        return _apply(self.output_quantizer_internal, _UNARY[self.function](x))

    def get_config(self):
        return {**super().get_config(), "function": self.function, "epsilon": self.epsilon,
                "input_quantizer": _ser(self.input_quantizer_internal),
                "output_quantizer": _ser(self.output_quantizer_internal)}

    def get_quantizers(self):
        return self.quantizers


class QRMSNorm(layers.Layer):
    """``x * rsqrt(mean(x^2) + eps) * gain`` with quantised intermediate values.

    ``ms_quantizer`` quantises the mean square (the rsqrt table index), ``rsqrt_quantizer``
    the reciprocal root, ``gain_quantizer`` the learned per-channel gain and
    ``output_quantizer`` the result.
    """

    def __init__(self, epsilon=1e-6, ms_quantizer=None, rsqrt_quantizer=None, gain_quantizer=None,
                 output_quantizer=None, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)
        self.ms_quantizer = ms_quantizer
        self.rsqrt_quantizer = rsqrt_quantizer
        self.gain_quantizer = gain_quantizer
        self.output_quantizer = output_quantizer
        self.ms_quantizer_internal = _q(ms_quantizer)
        self.rsqrt_quantizer_internal = _q(rsqrt_quantizer)
        self.gain_quantizer_internal = _q(gain_quantizer)
        self.output_quantizer_internal = _q(output_quantizer)
        self.quantizers = [self.ms_quantizer_internal, self.rsqrt_quantizer_internal,
                           self.gain_quantizer_internal, self.output_quantizer_internal]

    def build(self, input_shape):
        self.scale = self.add_weight(name="scale", shape=(input_shape[-1],), initializer="ones")

    def call(self, x):
        ms = _apply(self.ms_quantizer_internal, ops.mean(ops.square(x), axis=-1, keepdims=True))
        r = _apply(self.rsqrt_quantizer_internal, ops.rsqrt(ms + self.epsilon))
        g = _apply(self.gain_quantizer_internal, self.scale)
        return _apply(self.output_quantizer_internal, x * r * g)

    def get_config(self):
        return {**super().get_config(), "epsilon": self.epsilon,
                "ms_quantizer": _ser(self.ms_quantizer_internal),
                "rsqrt_quantizer": _ser(self.rsqrt_quantizer_internal),
                "gain_quantizer": _ser(self.gain_quantizer_internal),
                "output_quantizer": _ser(self.output_quantizer_internal)}

    def get_quantizers(self):
        return self.quantizers


class QSoftmax(layers.Layer):
    """Stable softmax ``e_i / sum(e)`` with ``e = exp(q_in(x - max x))``.

    ``input_quantizer`` sets the exp-table index width, ``exp_quantizer`` the table output,
    ``inv_quantizer`` the reciprocal of the sum and ``output_quantizer`` the probabilities.
    ``mask`` (optional, boolean, broadcastable to the input) removes keys before the max.
    """

    def __init__(self, axis=-1, input_quantizer=None, exp_quantizer=None, inv_quantizer=None,
                 output_quantizer=None, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.input_quantizer = input_quantizer
        self.exp_quantizer = exp_quantizer
        self.inv_quantizer = inv_quantizer
        self.output_quantizer = output_quantizer
        self.input_quantizer_internal = _q(input_quantizer)
        self.exp_quantizer_internal = _q(exp_quantizer)
        self.inv_quantizer_internal = _q(inv_quantizer)
        self.output_quantizer_internal = _q(output_quantizer)
        self.quantizers = [self.input_quantizer_internal, self.exp_quantizer_internal,
                           self.inv_quantizer_internal, self.output_quantizer_internal]

    def call(self, x, mask=None):
        if mask is not None:
            x = ops.where(mask, x, ops.cast(-1e9, x.dtype))
        x = x - ops.stop_gradient(ops.max(x, axis=self.axis, keepdims=True))
        x = _apply(self.input_quantizer_internal, x)
        e = _apply(self.exp_quantizer_internal, ops.exp(x))
        if mask is not None:
            e = ops.where(mask, e, ops.zeros_like(e))
        inv = _apply(self.inv_quantizer_internal, 1.0 / ops.sum(e, axis=self.axis, keepdims=True))
        return _apply(self.output_quantizer_internal, e * inv)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        return {**super().get_config(), "axis": self.axis,
                "input_quantizer": _ser(self.input_quantizer_internal),
                "exp_quantizer": _ser(self.exp_quantizer_internal),
                "inv_quantizer": _ser(self.inv_quantizer_internal),
                "output_quantizer": _ser(self.output_quantizer_internal)}

    def get_quantizers(self):
        return self.quantizers


def rotary_matrices(seq_len, head_dim, rotary_dim=None, theta=10000.0):
    """``(S, D, D)`` rotation matrices ``R_s`` with ``x_rot[s] = x[s] @ R_s``.

    Matches the "rotate-half" convention (``x*cos + rotate_half(x)*sin``) with the first
    ``rotary_dim / 2`` frequencies active and the rest identity.
    """
    rotary_dim = head_dim if rotary_dim is None else rotary_dim
    half = head_dim // 2
    inv = np.zeros(half, np.float64)
    idx = np.arange(0, rotary_dim, 2, dtype=np.float64)
    inv[: rotary_dim // 2] = 1.0 / (theta ** (idx / rotary_dim))
    ang = np.arange(seq_len)[:, None] * inv[None, :]
    c, s = np.cos(ang), np.sin(ang)
    R = np.zeros((seq_len, head_dim, head_dim), np.float64)
    for j in range(half):
        # out[j] = x[j] c - x[j+half] s ; out[j+half] = x[j+half] c + x[j] s
        R[:, j, j] = c[:, j]
        R[:, j + half, j] = -s[:, j]
        R[:, j + half, j + half] = c[:, j]
        R[:, j, j + half] = s[:, j]
    return R.astype(np.float32)


class QRotary(layers.Layer):
    """Fixed rotary embedding for ``(B, S, H, D)`` tensors as a per-position matrix.

    The rotation tables are constants for a fixed sequence length, so on hardware RoPE is
    a sparse constant matrix product (two multiplications and one addition per element).
    ``table_quantizer`` quantises cos/sin, ``output_quantizer`` the result.
    """

    def __init__(self, seq_len, head_dim, rotary_dim=None, theta=10000.0, table_quantizer=None,
                 output_quantizer=None, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = int(seq_len)
        self.head_dim = int(head_dim)
        self.rotary_dim = int(rotary_dim if rotary_dim is not None else head_dim)
        self.theta = float(theta)
        self.table_quantizer = table_quantizer
        self.output_quantizer = output_quantizer
        self.table_quantizer_internal = _q(table_quantizer)
        self.output_quantizer_internal = _q(output_quantizer)
        self.quantizers = [self.table_quantizer_internal, self.output_quantizer_internal]
        self._R = rotary_matrices(self.seq_len, self.head_dim, self.rotary_dim, self.theta)

    def rotation(self):
        R = ops.convert_to_tensor(self._R)
        return _apply(self.table_quantizer_internal, R)

    def call(self, x):
        y = ops.einsum("bshd,sde->bshe", x, self.rotation())
        return _apply(self.output_quantizer_internal, y)

    def get_config(self):
        return {**super().get_config(), "seq_len": self.seq_len, "head_dim": self.head_dim,
                "rotary_dim": self.rotary_dim, "theta": self.theta,
                "table_quantizer": _ser(self.table_quantizer_internal),
                "output_quantizer": _ser(self.output_quantizer_internal)}

    def get_quantizers(self):
        return self.quantizers


class QDotProductAttention(layers.Layer):
    """``softmax(Q K^T / sqrt(D) | mask) V`` for ``(B, S, H, D)`` inputs -> ``(B, S, H*D)``.

    ``attention_mask`` is a static boolean ``(S_q, S_k)`` array (True = attend).  (It is not
    called ``mask`` because ``qkeras.utils.quantized_model_from_json`` strips ``mask`` keys.)  ``score_quantizer``
    quantises the scaled scores (= softmax input), the ``softmax_*`` quantizers configure
    the internal ``QSoftmax`` and ``output_quantizer`` the mixed values.
    """

    def __init__(self, attention_mask=None, score_quantizer=None, softmax_exp_quantizer=None,
                 softmax_inv_quantizer=None, prob_quantizer=None, output_quantizer=None, **kwargs):
        super().__init__(**kwargs)
        self.attention_mask = None if attention_mask is None else np.asarray(attention_mask, bool)
        self.score_quantizer = score_quantizer
        self.output_quantizer = output_quantizer
        self.softmax = QSoftmax(axis=-1, input_quantizer=score_quantizer,
                                exp_quantizer=softmax_exp_quantizer,
                                inv_quantizer=softmax_inv_quantizer,
                                output_quantizer=prob_quantizer, name="softmax")
        self.output_quantizer_internal = _q(output_quantizer)
        self.quantizers = self.softmax.quantizers + [self.output_quantizer_internal]

    def build(self, q_shape, k_shape=None, v_shape=None):
        self.built = True

    def call(self, q, k, v):
        d = q.shape[-1]
        s = ops.einsum("bqhd,bkhd->bhqk", q, k) * (1.0 / math.sqrt(int(d)))
        am = self.attention_mask
        m = None if am is None else ops.convert_to_tensor(am[None, None])
        a = self.softmax(s, mask=m)
        o = ops.einsum("bhqk,bkhd->bqhd", a, v)
        o = ops.reshape(o, (-1, q.shape[1], q.shape[2] * q.shape[3]))
        return _apply(self.output_quantizer_internal, o)

    def get_config(self):
        sm = self.softmax
        return {**super().get_config(),
                "attention_mask": None if self.attention_mask is None else self.attention_mask.tolist(),
                "score_quantizer": _ser(sm.input_quantizer_internal),
                "softmax_exp_quantizer": _ser(sm.exp_quantizer_internal),
                "softmax_inv_quantizer": _ser(sm.inv_quantizer_internal),
                "prob_quantizer": _ser(sm.output_quantizer_internal),
                "output_quantizer": _ser(self.output_quantizer_internal)}

    def get_quantizers(self):
        return self.quantizers


QTRANSFORMER_OBJECTS = {
    "QEinsumDense": QEinsumDense,
    "QUnary": QUnary,
    "QRMSNorm": QRMSNorm,
    "QSoftmax": QSoftmax,
    "QRotary": QRotary,
    "QDotProductAttention": QDotProductAttention,
}
