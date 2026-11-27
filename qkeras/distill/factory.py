"""Layer factories: the single hook between a model builder and quantisation.

A model builder that creates its layers only through a factory --

    ``f.dense(units, use_bias, name)``   ``f.norm(eps, name)``   ``f.act(kind, name)``
    ``f.attention(mask, name)``          ``f.post(name)`` (optional re-quantisation, may be None)

-- gives, from one graph definition, the float teacher (``LayerFactory``), a float student
with hardware-friendly activations (``FloatFactory``) and the QAT student (``QuantFactory``).
Layer names are identical in all three, so weights move between them with
``qkeras.distill.weights.transfer_weights``.

``act`` kinds the builders use are the *reference* functions (``"silu"``, ``"sigmoid"``,
``"gelu"`` ...).  ``act_map`` substitutes hardware-friendly ones, e.g.
``{"silu": "relu", "sigmoid": "hard_sigmoid"}`` removes all lookup tables.
"""

from __future__ import annotations

from typing import Callable, Iterable

import keras
import numpy as np

from . import presets as P
from .layers import RMSNorm, StaticMaskAttention


class LayerFactory:
    """Float reference layers (the teacher)."""

    def dense(self, units: int, use_bias: bool, name: str, activation=None):
        return keras.layers.Dense(units, use_bias=use_bias, activation=activation, name=name)

    def norm(self, eps: float, name: str):
        return RMSNorm(epsilon=eps, name=name)

    def act(self, kind: str, name: str):
        return keras.layers.Activation(kind, name=name)

    def attention(self, mask: np.ndarray, name: str):
        return StaticMaskAttention(attention_mask=mask, name=name)

    def post(self, name: str):
        return None


class FloatFactory(LayerFactory):
    """Float layers with activation substitution (``act_map``)."""

    def __init__(self, act_map: dict | None = None):
        self.act_map = dict(act_map or {})

    def kind(self, kind: str) -> str:
        return self.act_map.get(kind, kind)

    def act(self, kind, name):
        return keras.layers.Activation(self.kind(kind), name=name)


class QuantFactory(FloatFactory):
    """QAT layers (qkeras + qkeras.qtransformer) for a preset.

    Args:
      preset: preset name (``qkeras.distill.presets.PRESETS``) or a roles dict.
      act_map: activation substitution, see module docstring.
      pruner: ``qkeras.pruning.Pruner`` attached to dense kernels (or None).
      prune_filter: ``name -> bool``; dense layers for which it is False get no pruner
        (e.g. keep the embedding and head dense).
      min_norm_eps: lower bound of the RMSNorm epsilon (a tiny eps makes the quantised
        rsqrt table explode for near-zero inputs).
      skip_post: layer-name substrings for which ``post`` returns None.
      kernel_initializer: passed to ``QDense`` if not None.
    """

    def __init__(self, preset, act_map: dict | None = None, pruner=None,
                 prune_filter: Callable[[str], bool] | None = None, min_norm_eps: float | None = None,
                 skip_post: Iterable[str] = (), kernel_initializer: str | None = None):
        super().__init__(act_map)
        self.p = P.resolve(preset)
        self.pruner = pruner
        self.prune_filter = prune_filter
        self.min_norm_eps = min_norm_eps
        self.skip_post = tuple(skip_post)
        self.kernel_initializer = kernel_initializer

    def dense(self, units, use_bias, name, activation=None):
        from qkeras import QDense

        kw = {}
        if self.pruner is not None and (self.prune_filter is None or self.prune_filter(name)):
            kw["kernel_pruner"] = self.pruner
        if self.kernel_initializer is not None:
            kw["kernel_initializer"] = self.kernel_initializer
        return QDense(units, use_bias=use_bias, activation=activation, kernel_quantizer=self.p["w"],
                      bias_quantizer=self.p["b"] if use_bias else None, name=name, **kw)

    def norm(self, eps, name):
        from qkeras.qtransformer import QRMSNorm

        p = self.p
        if self.min_norm_eps is not None:
            eps = max(eps, self.min_norm_eps)
        return QRMSNorm(epsilon=eps, ms_quantizer=p["ms"], rsqrt_quantizer=p["rsqrt"], gain_quantizer=p["gain"],
                        output_quantizer=p["a"], name=name)

    def act(self, kind, name):
        from qkeras import QActivation
        from qkeras.qtransformer import QUnary

        k = self.kind(kind)
        if k == "relu":
            bits, integer = P.bits_int(self.p["lut_out"])
            return QActivation(f"quantized_relu({bits},{integer})", name=name)
        if k == "hard_sigmoid":
            act = keras.layers.Activation("hard_sigmoid", name=name)
            q = QActivation(self.p["sig_out"], name=f"{name}_q")
            return lambda t: q(act(t))
        if k == "linear":
            return keras.layers.Activation("linear", name=name)
        out = self.p["sig_out"] if k == "sigmoid" else self.p["lut_out"]
        return QUnary(function=k, input_quantizer=self.p["lut_in"], output_quantizer=out, name=name)

    def attention(self, mask, name):
        from qkeras.qtransformer import QDotProductAttention

        p = self.p
        return QDotProductAttention(attention_mask=mask, score_quantizer=p["score"], softmax_exp_quantizer=p["exp"],
                                    softmax_inv_quantizer=p["inv"], prob_quantizer=p["prob"],
                                    output_quantizer=p["a"], name=name)

    def post(self, name):
        from qkeras import QActivation

        if not self.p["a"] or any(s in name for s in self.skip_post):
            return None
        return QActivation(self.p["a"], name=name)


def make_factory(preset=None, act_map: dict | None = None, **quant_kw) -> LayerFactory:
    """``FloatFactory`` for a float preset (``None``/``"float"``), else ``QuantFactory``."""
    if not P.is_quantized(preset):
        if quant_kw.get("pruner") is not None:
            raise ValueError("pruning is applied in the QAT phase; use a quantised preset")
        return FloatFactory(act_map)
    return QuantFactory(preset, act_map, **quant_kw)
