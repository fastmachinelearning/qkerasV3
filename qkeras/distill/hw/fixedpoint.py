"""Bit-exact integer references of the quantised transformer ops (golden models for RTL).

Everything the RTL does is defined here first in integer arithmetic and checked against the
qkeras quantizer objects of the trained model, so "matches the model" is proven on the
arithmetic.  Formats are *derived* from quantizer spec strings, with the qkeras value <-> code
convention:

  * quantized_bits(bits, integer, keep_negative=True)  -> signed, f = bits-1-integer
  * quantized_bits(bits, integer, keep_negative=0)     -> unsigned, f = bits-integer
  * quantized_relu(bits, integer)                      -> unsigned, f = bits-integer
  * rounding is qkeras ``_round_through`` = floor(x + 0.5)  (NOT numpy banker's rounding)

``P`` is a resolved preset (``qkeras.distill.presets.resolve``).  Integer datapaths (D = width):

  rmsnorm:  MS = clip(round(sum(X^2) * 2^(fms-2*fa) / D))          # mean square
            R  = RSQRT_LUT[MS]                                      # 1/sqrt(ms+eps), f=frs
            OUT_i = clip(round(X_i * R * G_i * 2^-(frs+fg)))        # * gain, to a-format
  softmax:  S = clip(round(score * 2^fs));  S -= max(S)
            E  = EXP_LUT[S]                                         # exp, f=fe
            INV = INV_LUT[sum(E)]                                   # 1/sum, f=fi
            P  = clip(round(E * INV * 2^-(fe+fi-fp)))               # prob, f=fp
  rope:     partial rotate-half with cos/sin in a chosen ``cs_spec`` format
"""

from __future__ import annotations

import numpy as np

from ..presets import parse_spec


def fmt(spec: str) -> dict:
    """``bits, f, signed, lo, hi, name, integer`` of an activation quantizer spec (alpha assumed 1)."""
    name, args, kw = parse_spec(spec)
    bits = int(args[0])
    integer = int(args[1]) if len(args) > 1 else 0
    if name == "quantized_bits":
        keep_neg = bool(kw.get("keep_negative", True))
        f = bits - 1 - integer if keep_neg else bits - integer
        signed = keep_neg
    elif name == "quantized_relu":
        f = bits - integer
        signed = False
    else:
        raise ValueError(name)
    lo = -(2 ** (bits - 1)) if signed else 0
    hi = 2 ** (bits - 1) - 1 if signed else 2 ** bits - 1
    return dict(bits=bits, f=f, signed=signed, lo=lo, hi=hi, name=name, integer=integer)


def qround(x):
    """qkeras _round_through: floor(x + 0.5)."""
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5)


def to_code(value, F: dict) -> np.ndarray:
    return np.clip(qround(np.asarray(value) * 2.0 ** F["f"]), F["lo"], F["hi"]).astype(np.int64)


def to_value(code, F: dict) -> np.ndarray:
    return np.asarray(code, dtype=np.float64) * 2.0 ** (-F["f"])


def make_quantizer(spec: str):
    """qkeras quantizer object from a spec string."""
    from qkeras.quantizers import quantized_bits, quantized_relu

    name, args, kw = parse_spec(spec)
    return {"quantized_bits": quantized_bits, "quantized_relu": quantized_relu}[name](*args, **kw)


def build_luts(P: dict, D: int, eps: float, nkeys_max: int = 32) -> dict:
    """ROMs the RTL instantiates, indexed by integer codes (all f-consistent)."""
    fms, frs = fmt(P["ms"]), fmt(P["rsqrt"])
    fe, fi, fs = fmt(P["exp"]), fmt(P["inv"]), fmt(P["score"])
    MS = np.arange(fms["lo"], fms["hi"] + 1)                      # rsqrt: index = MS code
    RSQRT = to_code(1.0 / np.sqrt(to_value(MS, fms) + eps), frs)
    S = np.arange(fs["lo"], fs["hi"] + 1)                         # exp: index = score code
    EXP = to_code(np.exp(to_value(S, fs)), fe)
    SUM = np.arange(1, nkeys_max * fe["hi"] + 2)                  # inv: index = sum of E codes
    INV = to_code(1.0 / (SUM * 2.0 ** (-fe["f"])), fi)
    return dict(MS=MS, RSQRT=RSQRT, EXP=EXP, SUM=SUM, INV=INV, D=D, eps=eps)


def rmsnorm_ref(X: np.ndarray, G: np.ndarray, luts: dict, P: dict) -> np.ndarray:
    """X (...,D) a-format codes, G (D,) gain codes -> OUT (...,D) a-format codes."""
    fa, fms, frs, fg = fmt(P["a"]), fmt(P["ms"]), fmt(P["rsqrt"]), fmt(P["gain"])
    D = X.shape[-1]
    sum_sq = (X.astype(np.int64) ** 2).sum(-1, keepdims=True)    # units 2^-2fa
    MS = np.clip(qround(sum_sq * 2.0 ** (fms["f"] - 2 * fa["f"]) / D), fms["lo"], fms["hi"]).astype(np.int64)
    R = luts["RSQRT"][MS - fms["lo"]]
    out = qround(X.astype(np.float64) * R * G * 2.0 ** -(frs["f"] + fg["f"]))
    return np.clip(out, fa["lo"], fa["hi"]).astype(np.int64)


def softmax_ref(scores: np.ndarray, luts: dict, P: dict, mask: np.ndarray | None = None) -> np.ndarray:
    """scores (...,S) real, softmax over the last axis -> prob codes (f=fp).  Mask True = attend."""
    fs, fe, fi, fp = fmt(P["score"]), fmt(P["exp"]), fmt(P["inv"]), fmt(P["prob"])
    x = np.asarray(scores, dtype=np.float64)
    if mask is not None:
        x = np.where(mask, x, -1e9)
    x = x - x.max(-1, keepdims=True)
    S = np.clip(qround(x * 2.0 ** fs["f"]), fs["lo"], fs["hi"]).astype(np.int64)
    E = luts["EXP"][S - fs["lo"]]
    if mask is not None:
        E = np.where(mask, E, 0)
    sumE = E.sum(-1, keepdims=True).astype(np.int64)
    INV = luts["INV"][np.clip(sumE - 1, 0, len(luts["SUM"]) - 1)]
    P_ = np.clip(qround(E * INV * 2.0 ** -(fe["f"] + fi["f"] - fp["f"])), fp["lo"], fp["hi"])
    return P_.astype(np.int64)


def rope_ref(x_codes, cos_codes, sin_codes, hd: int, n: int, P: dict, cs_spec: str):
    """x_codes (...,hd) a-format -> rotated a-format codes (partial rotate-half, ``n`` pairs)."""
    fa, fc = fmt(P["a"]), fmt(cs_spec)
    half = hd // 2
    x = to_value(x_codes, fa)
    c = to_value(cos_codes, fc)
    s = to_value(sin_codes, fc)
    y = x.copy()
    x1, x2 = x[..., :n], x[..., half:half + n]
    y[..., :n] = x1 * c - x2 * s
    y[..., half:half + n] = x2 * c + x1 * s
    return np.clip(qround(y * 2.0 ** fa["f"]), fa["lo"], fa["hi"]).astype(np.int64)
