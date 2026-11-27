"""Bit-exact fixed-point foundation for the full-model GateMate RTL (phase P0).

The generic integer references (rmsnorm, softmax, rope, LUTs) live in
``qkeras.distill.hw.fixedpoint``; this module fixes the B3 formats of the Timestral student and
adds the quantile head (softplus LUT + monotone cumsum) and the validation against qkeras layers.

Everything the RTL does is defined here first, in *integer* arithmetic, and validated
against the exact qkeras quantizer objects the trained student uses. The RTL generators
(P1+) reproduce these integer datapaths; the Verilator testbenches compare against these
references, so "matches the model" is proven on the arithmetic, not assumed.

Formats are *derived* from the B3 quantizer strings (``qkeras.distill.presets.PRESETS["B3"]``) rather than
hard-coded, and the value<->code convention matches qkeras exactly:

  * quantized_bits(bits, integer, keep_negative=True)  -> signed, f = bits-1-integer
  * quantized_bits(bits, integer, keep_negative=0)     -> unsigned, f = bits-integer
  * quantized_relu(bits, integer)                      -> unsigned, f = bits-integer
  * rounding is qkeras ``_round_through`` = floor(x + 0.5)  (NOT numpy banker's rounding)

Integer datapaths (D = d_model):
  rmsnorm:  MS = clip(round(sum(X^2) * 2^(fms-2*fa) / D))           # mean square
            R  = RSQRT_LUT[MS]                                       # 1/sqrt(ms+eps), f=frs
            OUT_i = clip(round(X_i * R * G_i * 2^-(frs+fg)))         # * gain, to a-format
  softmax:  S = clip(round(score * 2^fs));  S -= max(S)
            E  = EXP_LUT[S]                                          # exp, f=fe
            INV = INV_LUT[sum(E)]                                    # 1/sum, f=fi
            P  = clip(round(E * INV * 2^-(fe+fi-fp)))                # prob, f=fp
  quantiles: q = median +/- cumsum(width * softplus(raw))            # softplus via LUT

Run:  python -m timestral_kd.hw.fixedpoint            (validates against qkeras layers)
"""

from __future__ import annotations

import json
import os

import numpy as np

from qkeras.distill.hw import fixedpoint as _fp
from qkeras.distill.hw.fixedpoint import fmt, qround, to_code, to_value  # noqa: F401  (re-exported)
from qkeras.distill.hw.fixedpoint import make_quantizer as _make_quantizer
from qkeras.distill.presets import PRESETS

# ----------------------------------------------------------------------------- B3 specs
B3 = PRESETS["B3"]          # single source of truth: the preset the student was trained with

EPS = 1e-4  # QRMSNorm(epsilon=max(norm_eps, 1e-4))

# Hardware formats for the two blocks whose model form is float (so they are *chosen*,
# then validated to within 1 LSB): partial rotate-half tables and the quantile head output.
ROPE_CS = "quantized_bits(14,2,alpha=1)"   # cos/sin: signed, f=11, range [-4, 4); ~5e-4 LSB
QOUT = "quantized_bits(12,5,alpha=1)"     # quantile head output: signed, f=6, range [-32, 32)
ROPE_N = 2                                # n = rope_fraction * hd/2 = 0.5*8/2
HEAD_DIM = 8


# ----------------------------------------------------------------------------- generic refs, B3 defaults
def build_luts(D: int = 16, eps: float = EPS, P: dict = B3) -> dict:
    """ROMs the RTL instantiates (``qkeras.distill.hw.fixedpoint.build_luts``)."""
    return _fp.build_luts(P, D, eps)


def rmsnorm_ref(X: np.ndarray, G: np.ndarray, luts: dict, P: dict = B3) -> np.ndarray:
    return _fp.rmsnorm_ref(X, G, luts, P)


def softmax_ref(scores: np.ndarray, luts: dict, P: dict = B3, mask: np.ndarray | None = None) -> np.ndarray:
    return _fp.softmax_ref(scores, luts, P, mask)


def rope_ref(x_codes, cos_codes, sin_codes, hd: int = HEAD_DIM, n: int = ROPE_N, P: dict = B3):
    return _fp.rope_ref(x_codes, cos_codes, sin_codes, hd, n, P, ROPE_CS)


# ----------------------------------------------------------------------------- Timestral quantile head
def quantile_increment_widths(knots, width_factor: float) -> np.ndarray:
    k = [float(v) for v in knots]
    m = k.index(0.5)
    w = np.zeros(len(k), np.float64)
    for i in range(m + 1, len(k)):
        w[i] = k[i] - k[i - 1]
    for i in range(m):
        w[i] = k[i + 1] - k[i]
    return (width_factor * w).astype(np.float64)


def quantiles_ref(raw_codes: np.ndarray, knots, width_factor: float, P: dict = B3) -> np.ndarray:
    """raw_codes (...,Q) a-format -> float normalised quantiles (matches MonotoneQuantiles)."""
    fa = fmt(P["a"])
    raw = to_value(raw_codes, fa)
    w = quantile_increment_widths(knots, width_factor)
    m = list(knots).index(0.5)
    med = raw[..., m:m + 1]
    lower = med - np.flip(np.cumsum(np.flip(np.logaddexp(0.0, raw[..., :m]) * w[:m], -1), -1), -1) \
        if m > 0 else raw[..., :0]
    upper = med + np.cumsum(np.logaddexp(0.0, raw[..., m + 1:]) * w[m + 1:], -1) \
        if m + 1 < len(knots) else raw[..., :0]
    return np.concatenate([lower, med, upper], -1)


# ----------------------------------------------------------------------------- rope + head
def rope_tables_ref(seq_len: int = 20, base: float = 10000.0, n: int = ROPE_N):
    """(S, n) float cos/sin, matching timestral_kd.model.rope_tables for hd=8, fraction 0.5."""
    half = n
    inv_freq = 1.0 / (base ** (np.arange(0, half, dtype=np.float32) / max(half, 1)))
    pos = np.arange(seq_len, dtype=np.float32)[:, None]
    angles = pos * inv_freq[None, :]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def softplus_lut(nbits: int = 18, f: int = None, G: int = 8) -> np.ndarray:
    """LUT over a-format raw codes -> round(softplus(raw) * 2^(f+G)); G guard bits.

    Stored at higher precision than the QOUT output so the monotone cumsum does not
    accumulate > 1 LSB of rounding error.
    """
    fa = fmt(B3["a"])
    f = fmt(QOUT)["f"] if f is None else f
    codes = np.arange(fa["lo"], fa["hi"] + 1)
    val = np.logaddexp(0.0, to_value(codes, fa)) * 2.0 ** (f + G)
    return np.clip(qround(val), 0, 2 ** nbits - 1).astype(np.int64)


def quantiles_hw_ref(raw_codes, knots, width_factor: float, splut: np.ndarray,
                     G: int = 8, P: dict = B3):
    """raw_codes (...,Q) a-format -> QOUT-format integer codes (integer softplus + cumsum).

    median passes through to QOUT; offsets = width * softplus(raw) accumulated in QOUT
    fixed point. splut holds softplus*2^(f+G); width stored as w*2^G, so each increment is
    round(splut*woff / 2^(2G)) in QOUT units, rounding only once per knot.
    """
    fa, fq = fmt(P["a"]), fmt(QOUT)
    f = fq["f"]
    w = quantile_increment_widths(knots, width_factor)
    m = list(knots).index(0.5)
    Q = raw_codes.shape[-1]
    woff = qround(w * 2.0 ** G).astype(np.int64)           # width with G guard bits
    half = 1 << (2 * G - 1)
    out = np.zeros(raw_codes.shape, np.int64)
    out[..., m] = np.clip(qround(to_value(raw_codes[..., m], fa) * 2.0 ** f), fq["lo"], fq["hi"])
    for i in range(m + 1, Q):
        d = (splut[raw_codes[..., i] - fa["lo"]] * woff[i] + half) >> (2 * G)
        out[..., i] = out[..., i - 1] + d
    for i in range(m - 1, -1, -1):
        d = (splut[raw_codes[..., i] - fa["lo"]] * woff[i] + half) >> (2 * G)
        out[..., i] = out[..., i + 1] - d
    return np.clip(out, fq["lo"], fq["hi"]).astype(np.int64)


# ----------------------------------------------------------------------------- validation
def _validate():
    from qkeras.qtransformer import QRMSNorm, QSoftmax
    rng = np.random.default_rng(0)
    luts = build_luts(D=16)
    worst = {}

    # --- RMSNorm vs QRMSNorm
    ms_q, rs_q, g_q, out_q = (_make_quantizer(B3[k]) for k in ("ms", "rsqrt", "gain", "a"))
    layer = QRMSNorm(epsilon=EPS, ms_quantizer=ms_q, rsqrt_quantizer=rs_q, gain_quantizer=g_q,
                     output_quantizer=out_q)
    layer.build((None, 16))
    fg = fmt(B3["gain"])
    for _ in range(50):
        X = rng.integers(-60, 60, (1, 16))
        x = to_value(X, fmt(B3["a"]))
        scale = rng.uniform(0.3, 1.7, 16)
        layer.scale.assign(scale.astype(np.float32))
        ref = np.asarray(layer(x)).reshape(16)
        G = to_code(scale, fg)
        got = to_value(rmsnorm_ref(X, G, luts), fmt(B3["a"]))
        worst["rmsnorm"] = max(worst.get("rmsnorm", 0.0), float(np.abs(ref - got).max()))

    # --- softmax vs QSoftmax
    sm = QSoftmax(input_quantizer=_make_quantizer(B3["score"]), exp_quantizer=_make_quantizer(B3["exp"]),
                  inv_quantizer=_make_quantizer(B3["inv"]), output_quantizer=_make_quantizer(B3["prob"]))
    fp = fmt(B3["prob"])
    for _ in range(50):
        s = rng.normal(0, 2, (1, 1, 1, 20)).astype(np.float32)
        ref = np.asarray(sm(s)).reshape(20)
        got = to_value(softmax_ref(s.reshape(20), luts), fp)
        worst["softmax"] = max(worst.get("softmax", 0.0), float(np.abs(ref - got).max()))

    # --- quantiles vs MonotoneQuantiles
    from qkeras.distill.layers import MonotoneQuantiles
    knots = (0.1, 0.5, 0.9)
    mq = MonotoneQuantiles(knots, 3.0)
    for _ in range(50):
        R = rng.integers(-40, 40, (1, 1, 1, 3))
        ref = np.asarray(mq(to_value(R, fmt(B3["a"])))).reshape(3)
        got = quantiles_ref(R.reshape(3), knots, 3.0)
        worst["quantiles"] = max(worst.get("quantiles", 0.0), float(np.abs(ref - got).max()))

    # --- rope vs float Rotary (partial rotate-half), tables quantised to ROPE_CS
    from qkeras.distill.layers import Rotary

    from ..model import TimestralSpec, rope_tables
    spec = TimestralSpec(d_model=16, num_heads=2)   # head_dim = 8, rope_fraction = 0.5
    cos_f, sin_f = rope_tables(spec, 20)
    fc = fmt(ROPE_CS)
    cos_c = to_code(cos_f, fc)
    sin_c = to_code(sin_f, fc)
    rot = Rotary(cos_f, sin_f)
    fa = fmt(B3["a"])
    for _ in range(50):
        s = int(rng.integers(0, 20))
        X = rng.integers(fa["lo"], fa["hi"] + 1, (1, 20, 2, 8))   # full seq so cos broadcasts
        ref = to_code(np.asarray(rot(to_value(X, fa))).reshape(20, 2, 8)[s], fa)   # quantise ref to a
        got = np.stack([rope_ref(X[0, s, h], cos_c[s], sin_c[s]) for h in range(2)])
        worst["rope"] = max(worst.get("rope", 0.0),
                            float(np.abs(to_value(ref, fa) - to_value(got, fa)).max()))

    # --- quantile head vs MonotoneQuantiles, integer softplus LUT + QOUT output
    splut = softplus_lut()
    fq = fmt(QOUT)
    for _ in range(50):
        R = rng.integers(-40, 40, (1, 1, 1, 3))
        ref = np.asarray(mq(to_value(R, fa))).reshape(3)
        got = to_value(quantiles_hw_ref(R.reshape(3), knots, 3.0, splut), fq)
        worst["quantiles_hw"] = max(worst.get("quantiles_hw", 0.0), float(np.abs(ref - got).max()))

    print(json.dumps(worst, indent=1))
    # tolerances: <= 1 LSB of each output format
    assert worst["rope"] <= 2.0 ** -fa["f"] + 1e-9, "rope mismatch"
    assert worst["quantiles_hw"] <= 2.0 ** -fq["f"] + 1e-9, "quantile head mismatch"
    assert worst["rmsnorm"] <= 2.0 ** -fmt(B3["a"])["f"] + 1e-9, "rmsnorm mismatch"
    assert worst["softmax"] <= 2.0 ** -fp["f"] + 1e-9, "softmax mismatch"
    assert worst["quantiles"] <= 1e-4, "quantiles mismatch"
    print("P0 fixed-point references validated against qkeras layers (<= 1 LSB).")
    return worst


def dump_luts(path: str, D: int = 16):
    luts = build_luts(D)
    np.savez(os.path.join(path) if path.endswith(".npz") else path + ".npz",
             **{k: v for k, v in luts.items() if isinstance(v, np.ndarray)})
    json.dump({k: fmt(B3[k]) for k in B3}, open((path[:-4] if path.endswith(".npz") else path) + ".formats.json", "w"),
              indent=1)


if __name__ == "__main__":
    _validate()
