"""Quantisation presets: one qkeras quantizer spec per tensor *role* of a transformer.

A preset is a plain ``dict`` ``role -> quantizer spec string``.  Roles
--------------------------------------------------------------------
``w``        dense kernels                      ``b``        dense biases
``a``        activations (block outputs, residual stream, projections, attention output)
``lut_in``   input of an element-wise function (silu, sigmoid, gelu, ...) -> table index
``lut_out``  output of an element-wise function (also sets the ``quantized_relu`` format)
``sig_out``  output of the attention gate (sigmoid / hard_sigmoid); default ``lut_out``
``ms``       RMSNorm mean square                ``rsqrt``    RMSNorm reciprocal root
``gain``     RMSNorm gain; default ``w``
``score``    attention scores                   ``exp``      softmax exponent; default ``prob``
``inv``      softmax reciprocal sum             ``prob``     attention probabilities

A missing role (or ``None``) leaves that tensor unquantised.  ``resolve`` fills the defaults
above, so every factory sees all roles.  Register your own presets with ``register_preset``.

The ``A*`` presets target a *folded* design (weights in BRAM, shared multipliers), the ``B*``
presets a *streamed constant-weight* design (every dense is a shift-add block, narrow
everything).  ``W8A14`` is the wide preset of the first Timestral PTQ / range-analysis studies.
"""

from __future__ import annotations

import ast

ROLES = ("w", "b", "a", "lut_in", "lut_out", "sig_out", "ms", "rsqrt", "gain", "score", "exp", "inv",
         "prob")

PRESETS: dict[str, dict | None] = {
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
    # wide preset of the first Timestral PTQ / range-analysis scripts (gain = w, exp = prob,
    # sig_out = lut_out); used with ``QuantFactory(..., skip_post=("head",))``.
    "W8A14": dict(w="quantized_bits(8, 2, alpha=None)", b="quantized_bits(12, 4, alpha=None)",
                  a="quantized_bits(14, 5, symmetric=1)", lut_in="quantized_bits(12, 4, symmetric=1)",
                  lut_out="quantized_bits(12, 3, symmetric=1)",
                  ms="quantized_bits(16, 6, keep_negative=False)",
                  rsqrt="quantized_bits(16, 10, keep_negative=False)",
                  score="quantized_bits(12, 5, symmetric=1)",
                  inv="quantized_bits(14, 0, keep_negative=False)",
                  prob="quantized_bits(12, 0, keep_negative=False)"),
}


def register_preset(name: str, roles: dict | None, overwrite: bool = False) -> None:
    """Add a preset (``roles`` may be partial; ``None`` = float)."""
    if name in PRESETS and not overwrite:
        raise ValueError(f"preset {name!r} exists (pass overwrite=True)")
    if roles is not None:
        unknown = set(roles) - set(ROLES)
        if unknown:
            raise ValueError(f"unknown roles {sorted(unknown)}; valid: {ROLES}")
    PRESETS[name] = None if roles is None else dict(roles)


def is_quantized(preset) -> bool:
    """True for a quantised preset name or a roles dict."""
    if isinstance(preset, dict):
        return True
    return bool(preset) and PRESETS.get(preset) is not None


def resolve(preset) -> dict:
    """All roles of a preset (name or dict) with the documented defaults filled in."""
    p = dict(PRESETS[preset] if isinstance(preset, str) else preset)
    out = {r: p.get(r) for r in ROLES}
    out["sig_out"] = p.get("sig_out", out["lut_out"])
    out["gain"] = p.get("gain", out["w"])
    out["exp"] = p.get("exp", out["prob"])
    return out


def parse_spec(spec: str) -> tuple[str, list, dict]:
    """``"quantized_bits(7,3,alpha=1)"`` -> ``("quantized_bits", [7, 3], {"alpha": 1})``."""
    tree = ast.parse(spec, mode="eval").body
    return tree.func.id, [ast.literal_eval(a) for a in tree.args], \
        {kw.arg: ast.literal_eval(kw.value) for kw in tree.keywords}


def bits_int(spec: str) -> tuple[int, int]:
    """Total and integer bits of a quantizer spec."""
    _, args, _ = parse_spec(spec)
    return int(args[0]), int(args[1]) if len(args) > 1 else 0


def preset_bits(preset) -> dict:
    """Total bit width per role (32 for a float preset); input for cost models."""
    if not is_quantized(preset):
        return {k: 32 for k in ("w", "a", "lut_in", "lut_out", "score", "prob", "gain")}
    roles = PRESETS[preset] if isinstance(preset, str) else preset
    return {k: bits_int(s)[0] for k, s in roles.items() if s is not None}
