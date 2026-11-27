"""Student models: Timestral-style multivariate transformers sized for small FPGAs.

A student reuses ``model.build_timestral_multivariate`` with a small spec, so the float
student, the quantised (QAT) student and the teacher share one graph definition; the layers
come from a ``qkeras.distill`` factory for the student's preset.

``variate_every``: 0 -> channel-independent (time attention only; every variate is
processed alone with shared weights), k -> variate attention in every k-th block (the
teacher uses k = 4), 1 -> in every block.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import keras

from qkeras.distill import presets as _presets
from qkeras.distill.factory import make_factory
from qkeras.distill.presets import PRESETS, preset_bits  # noqa: F401  (re-exported)
from qkeras.distill.weights import dense_sparsity, transfer_weights  # noqa: F401  (re-exported)

from . import model as kt

# QAT options shared by every Timestral student (see qkeras.distill.factory.QuantFactory)
STUDENT_QUANT_KW = dict(min_norm_eps=1e-4, kernel_initializer="glorot_uniform")


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


# ----------------------------------------------------------------------------- build
def build_student(cfg: StudentConfig, num_variates: int, preset: str | None = None, pruner=None) -> keras.Model:
    """Float or QAT student; ``pruner`` (QAT presets only) adds magnitude-pruning masks to every QDense."""
    preset = preset or "float"
    act_map = {"silu": cfg.act_fn, "sigmoid": cfg.gate_fn}
    kw = STUDENT_QUANT_KW if _presets.is_quantized(preset) else {}
    factory = make_factory(preset, act_map, pruner=pruner, **kw)
    return kt.build_timestral_multivariate(cfg.spec(), num_variates, cfg.ctx_patches, cfg.hor_patches,
                                           factory=factory, name=f"{cfg.name}_{preset}")


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
