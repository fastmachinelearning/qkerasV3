"""Student models: Timestral-style multivariate transformers sized for small FPGAs.

A student reuses ``keras_timestral.build_timestral_multivariate`` with a small spec, so the
float student, the quantised (QAT) student and the teacher share one graph definition.

``variate_every``: 0 -> channel-independent (time attention only; every variate is
processed alone with shared weights), k -> variate attention in every k-th block (the
teacher uses k = 4), 1 -> in every block.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import keras
import numpy as np

import keras_timestral as kt
from qkeras import QActivation, QDense
from qkeras.qtransformer import QDotProductAttention, QRMSNorm, QUnary

# ----------------------------------------------------------------------------- presets
# Quantizer presets per deployment class.  Strings are qkeras quantizer specs.
#   w   weights            a   activations (block outputs, residual stream, projections)
#   lut_in/out  inputs/outputs of element-wise functions (silu, sigmoid) -> table size 2^bits
#   ms/rsqrt    RMSNorm mean square and reciprocal root,  gain  RMSNorm gain
#   score/exp/inv/prob  softmax pipeline
PRESETS = {
    "float": None,
    # class A: folded design, weights in BRAM, shared multipliers
    "A8": dict(w="quantized_bits(8,1,alpha=1)", b="quantized_bits(16,6,alpha=1)",
               a="quantized_bits(16,6,alpha=1)", lut_in="quantized_bits(12,5,alpha=1)",
               lut_out="quantized_bits(12,5,alpha=1)", sig_out="quantized_bits(12,0,keep_negative=0,alpha=1)",
               ms="quantized_bits(16,8,keep_negative=0,alpha=1)", rsqrt="quantized_bits(16,8,keep_negative=0,alpha=1)",
               gain="quantized_bits(10,3,alpha=1)", score="quantized_bits(14,6,alpha=1)",
               exp="quantized_bits(16,1,keep_negative=0,alpha=1)", inv="quantized_bits(16,1,keep_negative=0,alpha=1)",
               prob="quantized_bits(12,1,keep_negative=0,alpha=1)"),
    "A6": dict(w="quantized_bits(6,1,alpha='auto_po2')", b="quantized_bits(16,6,alpha=1)",
               a="quantized_bits(12,5,alpha=1)", lut_in="quantized_bits(10,4,alpha=1)",
               lut_out="quantized_bits(10,4,alpha=1)", sig_out="quantized_bits(10,0,keep_negative=0,alpha=1)",
               ms="quantized_bits(14,7,keep_negative=0,alpha=1)", rsqrt="quantized_bits(14,7,keep_negative=0,alpha=1)",
               gain="quantized_bits(8,3,alpha=1)", score="quantized_bits(12,5,alpha=1)",
               exp="quantized_bits(12,1,keep_negative=0,alpha=1)", inv="quantized_bits(12,1,keep_negative=0,alpha=1)",
               prob="quantized_bits(10,1,keep_negative=0,alpha=1)"),
    # class B: constant-weight dataflow (shift-add / LUT), narrow everything
    "B4": dict(w="quantized_bits(4,1,alpha='auto_po2')", b="quantized_bits(10,4,alpha=1)",
               a="quantized_bits(8,3,alpha=1)", lut_in="quantized_bits(7,3,alpha=1)",
               lut_out="quantized_bits(8,3,alpha=1)", sig_out="quantized_bits(6,0,keep_negative=0,alpha=1)",
               ms="quantized_bits(10,5,keep_negative=0,alpha=1)", rsqrt="quantized_bits(10,5,keep_negative=0,alpha=1)",
               gain="quantized_bits(6,2,alpha=1)", score="quantized_bits(8,4,alpha=1)",
               exp="quantized_bits(8,1,keep_negative=0,alpha=1)", inv="quantized_bits(8,1,keep_negative=0,alpha=1)",
               prob="quantized_bits(7,1,keep_negative=0,alpha=1)"),
    "B3": dict(w="quantized_bits(3,1,alpha='auto_po2')", b="quantized_bits(8,3,alpha=1)",
               a="quantized_bits(7,3,alpha=1)", lut_in="quantized_bits(6,3,alpha=1)",
               lut_out="quantized_bits(7,3,alpha=1)", sig_out="quantized_bits(5,0,keep_negative=0,alpha=1)",
               ms="quantized_bits(9,5,keep_negative=0,alpha=1)", rsqrt="quantized_bits(9,5,keep_negative=0,alpha=1)",
               gain="quantized_bits(5,2,alpha=1)", score="quantized_bits(7,4,alpha=1)",
               exp="quantized_bits(7,1,keep_negative=0,alpha=1)", inv="quantized_bits(7,1,keep_negative=0,alpha=1)",
               prob="quantized_bits(6,1,keep_negative=0,alpha=1)"),
}


def preset_bits(preset: str | None) -> dict:
    """Total bit widths of the preset's quantizers (for the cost model)."""
    if not preset or PRESETS.get(preset) is None:
        return {k: 32 for k in ("w", "a", "lut_in", "lut_out", "score", "prob", "gain")}
    out = {}
    for k, s in PRESETS[preset].items():
        out[k] = int(s.split("(")[1].split(",")[0])
    return out


# ----------------------------------------------------------------------------- config
@dataclass
class StudentConfig:
    name: str = "student"
    patch_size: int = 16
    d_model: int = 32
    num_heads: int = 2
    mlp_hidden: int = 64
    n_blocks: int = 2
    variate_every: int = 1
    embed_hidden: int | None = None   # default 2 * d_model
    head_hidden: int | None = None    # default 2 * d_model
    ctx_steps: int = 256
    hor_steps: int = 64
    knots: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    width_factor: float = 3.0
    rope_fraction: float = 0.5
    preset: str = "float"              # QAT preset used in phase 2
    deployment: str = "A"             # A = folded/BRAM, B = streamed constant-weight
    act_fn: str = "silu"              # "silu" (table) or "relu" (no table)
    gate_fn: str = "sigmoid"          # "sigmoid" (table) or "hard_sigmoid" (shift + clamp)

    @classmethod
    def from_dict(cls, d: dict) -> "StudentConfig":
        d = dict(d)
        if "knots" in d:
            d["knots"] = tuple(d["knots"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def ctx_patches(self) -> int:
        return self.ctx_steps // self.patch_size

    @property
    def hor_patches(self) -> int:
        return self.hor_steps // self.patch_size

    @property
    def eh(self) -> int:
        return self.embed_hidden or 2 * self.d_model

    @property
    def hh(self) -> int:
        return self.head_hidden or 2 * self.d_model

    def schedule(self) -> tuple:
        k = self.variate_every
        return tuple("time_variate" if (k and (i + 1) % k == 0) else "time" for i in range(self.n_blocks))

    def spec(self) -> kt.TimestralSpec:
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        hd = self.d_model // self.num_heads
        assert hd % 2 == 0 and int(hd * self.rope_fraction) >= 2, "head_dim too small for RoPE"
        assert self.ctx_steps % self.patch_size == 0 and self.hor_steps % self.patch_size == 0
        assert 0.5 in self.knots
        return kt.TimestralSpec(
            patch_size=self.patch_size, d_model=self.d_model, num_heads=self.num_heads,
            mlp_hidden_dim=self.mlp_hidden, num_blocks=self.n_blocks, block_schedule=self.schedule(),
            rope_fraction=self.rope_fraction, quantile_knots=tuple(self.knots),
            quantile_width_factor=self.width_factor, embed_hidden=self.eh, head_hidden=self.hh)


# ----------------------------------------------------------------------------- factory
class FloatFactory(kt.LayerFactory):
    """Float factory with optional hardware-friendly activations."""

    def __init__(self, act_fn: str = "silu", gate_fn: str = "sigmoid"):
        self.act_fn, self.gate_fn = act_fn, gate_fn

    def _kind(self, kind):
        if kind == "silu":
            return self.act_fn
        if kind == "sigmoid":
            return self.gate_fn
        return kind

    def act(self, kind, name):
        return keras.layers.Activation(self._kind(kind), name=name)


class QuantFactory(FloatFactory):
    """QAT layer factory for a preset (qkeras + qkeras.qtransformer)."""

    def __init__(self, preset: str, act_fn: str = "silu", gate_fn: str = "sigmoid", pruner=None):
        super().__init__(act_fn, gate_fn)
        self.p = PRESETS[preset]
        self.pruner = pruner              # qkeras.pruning.Pruner for every QDense kernel, or None

    def dense(self, units, use_bias, name, activation=None):
        extra = {"kernel_pruner": self.pruner} if self.pruner is not None else {}
        return QDense(units, use_bias=use_bias, activation=activation, kernel_quantizer=self.p["w"],
                      bias_quantizer=self.p["b"] if use_bias else None, kernel_initializer="glorot_uniform",
                      name=name, **extra)

    def norm(self, eps, name):
        p = self.p
        return QRMSNorm(epsilon=max(eps, 1e-4), ms_quantizer=p["ms"], rsqrt_quantizer=p["rsqrt"],
                        gain_quantizer=p["gain"], output_quantizer=p["a"], name=name)

    def act(self, kind, name):
        k = self._kind(kind)
        if k == "relu":
            bits, integer = _bits_int(self.p["lut_out"])
            return QActivation(f"quantized_relu({bits},{integer})", name=name)
        if k == "hard_sigmoid":
            act = keras.layers.Activation("hard_sigmoid", name=name)
            q = QActivation(self.p["sig_out"], name=f"{name}_q")
            return lambda t: q(act(t))
        out = self.p["sig_out"] if k == "sigmoid" else self.p["lut_out"]
        return QUnary(function=k, input_quantizer=self.p["lut_in"], output_quantizer=out, name=name)

    def attention(self, mask, name):
        p = self.p
        return QDotProductAttention(attention_mask=mask, score_quantizer=p["score"], softmax_exp_quantizer=p["exp"],
                                    softmax_inv_quantizer=p["inv"], prob_quantizer=p["prob"],
                                    output_quantizer=p["a"], name=name)

    def post(self, name):
        return QActivation(self.p["a"], name=name)


def _bits_int(spec: str) -> tuple[int, int]:
    a = spec.split("(")[1].split(")")[0].split(",")
    return int(a[0]), int(a[1])


def build_student(cfg: StudentConfig, num_variates: int, preset: str | None = None, pruner=None) -> keras.Model:
    """Float or QAT student; ``pruner`` (QAT presets only) adds magnitude-pruning masks to every QDense."""
    preset = preset or "float"
    if PRESETS.get(preset) is None:
        if pruner is not None:
            raise ValueError("pruning is applied in the QAT phase; use a quantised preset")
        factory = FloatFactory(cfg.act_fn, cfg.gate_fn)
    else:
        factory = QuantFactory(preset, cfg.act_fn, cfg.gate_fn, pruner)
    return kt.build_timestral_multivariate(cfg.spec(), num_variates, cfg.ctx_patches, cfg.hor_patches,
                                           factory=factory, name=f"{cfg.name}_{preset}")


def _own_vars(layer):
    """Layer variables without qkeras.pruning masks (they live in a ``pruning_masks`` sublayer)."""
    return [v for v in layer.weights if "_pruning_mask" not in (getattr(v, "path", None) or v.name)]


def transfer_weights(src: keras.Model, dst: keras.Model) -> int:
    """Copy weights layer by layer (matching names and shapes, pruning masks ignored on both sides).

    A pruned source must have its masks baked in (``UpdatePruningStep`` does that at the end of
    ``fit``) for the destination to be exactly sparse.  Returns the number of layers copied.
    """
    n = 0
    for lyr in dst.layers:
        try:
            s = src.get_layer(lyr.name)
        except ValueError:
            continue
        sv, dv = _own_vars(s), _own_vars(lyr)
        if sv and len(sv) == len(dv) and all(tuple(a.shape) == tuple(b.shape) for a, b in zip(sv, dv)):
            for a, b in zip(sv, dv):
                b.assign(a)
            n += 1
    return n


def dense_sparsity(qmodel) -> dict:
    """Zero fraction of the *quantised* kernels (pruned + quantised-to-zero weights)."""
    z = t = 0
    for lyr in qmodel.layers:
        q = getattr(lyr, "kernel_quantizer_internal", None)
        if q is None or not hasattr(lyr, "kernel"):
            continue
        k = np.asarray(q(lyr.kernel))
        z += int(np.sum(k == 0)); t += k.size
    return {"zero_fraction": z / t if t else 0.0, "nonzero": t - z, "total": t}


# ----------------------------------------------------------------------------- counting
def count_ops(cfg: StudentConfig, V: int) -> dict:
    """Analytic per-forecast counts: parameters, constant-weight MACs, activation x activation
    MACs (attention), and element-wise function evaluations (table lookups)."""
    P, D, H, M = cfg.patch_size, cfg.d_model, cfg.num_heads, cfg.mlp_hidden
    hd, Q = D // H, len(cfg.knots)
    L, h = cfg.ctx_patches, cfg.hor_patches
    S, T = L + h, V * (L + h)
    eh, hh = cfg.eh, cfg.hh
    params = {"embed": 2 * P * eh + eh + eh * D + D + 2 * P * D + D, "query": D,
              "head": D * hh + hh * P * Q + D * P * Q, "norm": D}
    w_mac = V * L * (2 * P * eh + eh * D + 2 * P * D) + V * h * (D * hh + hh * P * Q + D * P * Q)
    aa_mac, lut = 0, {"silu": V * L * eh + V * h * hh, "sigmoid": 0, "exp": 0, "rsqrt": T, "softplus": V * h * P * (Q - 1)}
    blocks = 0
    for kind in cfg.schedule():
        axes = [True] if kind == "time" else [True, False]
        for is_time in axes:
            params["norm"] += D
            blocks += 5 * D * D
            w_mac += T * 5 * D * D            # QKV (3D^2) + gate + output projection
            lut["sigmoid"] += T * D
            lut["rsqrt"] += T
            if is_time:
                aa_mac += V * H * 2 * S * S * hd
                lut["exp"] += V * H * S * S
            else:
                aa_mac += S * H * 2 * V * V * hd
                lut["exp"] += S * H * V * V
        params["norm"] += D
        blocks += 3 * D * M
        w_mac += T * 3 * D * M
        lut["silu"] += T * M
        lut["rsqrt"] += T
    params["blocks"] = blocks
    n_params = sum(params.values())
    return {"params": int(n_params), "params_by_part": params, "weight_macs": int(w_mac),
            "act_macs": int(aa_mac), "lookups": lut, "tokens": T, "seq_len": S}
