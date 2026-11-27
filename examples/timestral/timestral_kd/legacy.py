"""Adapter for the first Timestral PTQ / range-analysis scripts (``scripts/``).

Those scripts configured quantisation through keyword arguments named per tensor
(``weight_q``, ``act_q``, ``rms_rsqrt_q`` ...).  ``legacy_factory`` maps them onto preset roles
and returns a ``qkeras.distill.QuantFactory``.  The couplings of the old classes are the
preset defaults: RMSNorm gain = ``weight_q``, softmax exp = ``softmax_prob_q``, sigmoid
output = ``lut_out_q``.
"""

from __future__ import annotations

from qkeras.distill.factory import QuantFactory
from qkeras.distill.presets import PRESETS

LEGACY_ROLE = {"weight_q": "w", "bias_q": "b", "act_q": "a", "lut_in_q": "lut_in", "lut_out_q": "lut_out",
               "rms_ms_q": "ms", "rms_rsqrt_q": "rsqrt", "attn_score_q": "score", "softmax_inv_q": "inv",
               "softmax_prob_q": "prob"}


def legacy_roles(base: str | None = "W8A14", **legacy) -> dict:
    roles = dict(PRESETS[base]) if base else {}
    for k, v in legacy.items():
        roles[LEGACY_ROLE[k]] = v
    return roles


def legacy_factory(base: str | None = "W8A14", skip_head_post: bool = True, pruner=None,
                   prune_embed_and_head: bool = False, **legacy) -> QuantFactory:
    """``QuantFactory`` equivalent of the old ``QTimestralFactory`` / ``PrunableQTimestralFactory``.

    ``base=None`` starts from an empty preset (every tensor float unless given), as the
    quantisation sweep did; ``skip_head_post`` drops the re-quantisation after the head.
    """
    prune_filter = None if prune_embed_and_head else (lambda n: not n.startswith(("embed_", "head_")))
    return QuantFactory(legacy_roles(base, **legacy), pruner=pruner, prune_filter=prune_filter,
                        skip_post=("head",) if skip_head_post else ())
