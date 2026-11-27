"""Integer program of a trained class-B student: the golden model the GateMate engine must match.

``extract(run)`` reads a trained QAT student (``qat.weights.h5``) and turns every constant into
integers with an explicit number of fractional bits:

* dense kernels / biases: the exact QAT-quantised values (``auto_po2`` scales included),
* RMSNorm gains, the forecast-query token, RoPE cos/sin and ``1/sqrt(head_dim)``,
* lookup tables (rsqrt over mean-square codes, exp over score codes, reciprocal over sums of
  exp codes, the element-wise activation when it is a table) are produced by running the
  model's own TensorFlow ops and qkeras quantizers on every possible input code, so they are
  bit-identical to the QAT model.

``forward(prog, x_codes)`` runs the student in integer arithmetic for a batch of windows and
returns the ``head_post`` codes ``(N, V, h*P, Q)`` -- the hardware output.  The monotone
quantile transform and de-normalisation (a few hundred float ops per window) stay on the CPU
(``quantiles_from_raw``).

Rounding: qkeras quantizers round half to *even* (``rne``); hardware-chosen intermediate
precisions (``HWFormats``) round half up (``rhu``, add-half-and-truncate).  Exactness vs the
Keras model: every op between two qkeras quantizers is exact fixed-point arithmetic, except the
float input, the float forecast-query token, RoPE, ``1/sqrt(head_dim)`` and ``hard_sigmoid``'s
``/6``; those are carried at the widths in ``HWFormats`` and ``check`` measures how often a
quantised value differs from the Keras model.

Supported students: ``variate_every = 0`` (channel-independent), any ``n_blocks``, ``act_fn``
relu or silu (table), ``gate_fn`` hard_sigmoid or sigmoid (table), class-B presets.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

import numpy as np

from qkeras.distill.hw.fixedpoint import fmt
from qkeras.distill.presets import resolve


# ----------------------------------------------------------------------------- rounding
def rne_shift(x, s: int):
    """Round ``x / 2**s`` half to even (``s`` may be <= 0)."""
    x = np.asarray(x, np.int64)
    if s <= 0:
        return x << (-s)
    q = x >> s
    r = x - (q << s)
    half = np.int64(1) << (s - 1)
    return q + ((r > half) | ((r == half) & ((q & 1) == 1)))


def rhu_shift(x, s: int):
    """Round ``x / 2**s`` half up (add half, arithmetic shift)."""
    x = np.asarray(x, np.int64)
    if s <= 0:
        return x << (-s)
    return (x + (np.int64(1) << (s - 1))) >> s


def rne_div(num, den: int):
    """Round ``num / den`` half to even, ``den > 0``."""
    num = np.asarray(num, np.int64)
    q = num // den
    r2 = 2 * (num - q * den)
    return q + ((r2 > den) | ((r2 == den) & ((q & 1) == 1)))


def clip(x, F: dict):
    return np.clip(x, F["lo"], F["hi"])


def requant(x, f_from: int, F: dict):
    """qkeras quantizer ``F`` applied to an exact value with ``f_from`` fractional bits."""
    return clip(rne_shift(x, f_from - F["f"]), F)


# ----------------------------------------------------------------------------- formats
@dataclass
class HWFormats:
    """Widths the hardware chooses where the Keras model computes in float."""

    in_bits: int = 12          # input: quantized_bits(in_bits, in_int), signed
    in_int: int = 3            #   asinh-scaled values stay within +-8
    tok_f: int = 24            # forecast-query token, fractional bits (residual stream)
    rope_f: int = 16           # cos / sin
    qrot_f: int = 16           # rotated q / k
    score_f: int = 16          # q.k and scaled scores before max-subtraction
    invsqrt_f: int = 20        # 1 / sqrt(head_dim)
    ms_f: int = 16             # residual precision used for the RMSNorm mean square

    @property
    def in_f(self) -> int:
        return self.in_bits - 1 - self.in_int


@dataclass
class Program:
    cfg: dict                                   # StudentConfig.to_dict()
    hw: dict                                    # HWFormats
    q: dict                                     # role -> quantizer spec (resolved preset)
    dense: dict = field(default_factory=dict)   # name -> {"W": int (in,out), "fw": int, "b": int|None, "fb": int}
    gains: dict = field(default_factory=dict)   # norm name -> int codes (gain format)
    luts: dict = field(default_factory=dict)    # name -> int array
    token: np.ndarray | None = None             # (D,) codes at tok_f
    cos: np.ndarray | None = None               # (S, n) codes at rope_f
    sin: np.ndarray | None = None
    invsqrt: int = 0
    mask: np.ndarray | None = None              # (S, S) bool

    # sizes
    @property
    def dims(self) -> dict:
        c = self.cfg
        P, D, H = c["patch_size"], c["d_model"], c["num_heads"]
        L, h = c["ctx_steps"] // P, c["hor_steps"] // P
        hd = D // H
        return dict(P=P, D=D, H=H, hd=hd, M=c["mlp_hidden"], L=L, h=h, S=L + h, Q=len(c["knots"]),
                    EH=c["embed_hidden"] or 2 * D, HH=c["head_hidden"] or 2 * D, B=c["n_blocks"],
                    n=int(hd * c["rope_fraction"]) // 2)

    def F(self, role: str) -> dict:
        return fmt(self.q[role])

    def save(self, path: str) -> None:
        arrs = {}
        for n, d in self.dense.items():
            arrs[f"dense/{n}/W"] = d["W"]
            if d["b"] is not None:
                arrs[f"dense/{n}/b"] = d["b"]
        for n, g in self.gains.items():
            arrs[f"gain/{n}"] = g
        for n, t in self.luts.items():
            arrs[f"lut/{n}"] = t
        meta = dict(cfg=self.cfg, hw=self.hw, q=self.q, invsqrt=self.invsqrt,
                    dense={n: dict(fw=d["fw"], fb=d["fb"]) for n, d in self.dense.items()})
        import json
        np.savez_compressed(path, token=self.token, cos=self.cos, sin=self.sin, mask=self.mask,
                            meta=json.dumps(meta), **arrs)

    @classmethod
    def load(cls, path: str) -> "Program":
        import json
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        p = cls(cfg=meta["cfg"], hw=meta["hw"], q=meta["q"], invsqrt=int(meta["invsqrt"]))
        for n, fb in meta["dense"].items():
            p.dense[n] = dict(W=z[f"dense/{n}/W"], b=z[f"dense/{n}/b"] if f"dense/{n}/b" in z else None, **fb)
        for k in z.files:
            if k.startswith("gain/"):
                p.gains[k[5:]] = z[k]
            elif k.startswith("lut/"):
                p.luts[k[4:]] = z[k]
        p.token, p.cos, p.sin, p.mask = z["token"], z["cos"], z["sin"], z["mask"]
        return p


# ----------------------------------------------------------------------------- extraction
def _exact_int(values, max_f: int = 30) -> tuple[np.ndarray, int]:
    v = np.asarray(values, np.float64)
    for f in range(max_f + 1):
        s = v * 2.0 ** f
        if np.array_equal(s, np.round(s)):
            return np.round(s).astype(np.int64), f
    raise ValueError("not a fixed-point value within 2**-%d" % max_f)


def _code_range(F: dict) -> np.ndarray:
    return np.arange(F["lo"], F["hi"] + 1, dtype=np.int64)


def _keras_lut(fn, in_codes, f_in: int, quantizer: str) -> np.ndarray:
    """``quantizer(fn(code * 2**-f_in))`` evaluated with the model's float32 ops -> output codes."""
    from keras import ops

    from qkeras.quantizers import get_quantizer

    x = ops.convert_to_tensor((np.asarray(in_codes, np.float64) * 2.0 ** -f_in).astype(np.float32))
    y = np.asarray(get_quantizer(quantizer)(fn(x)), np.float64)
    F = fmt(quantizer)
    codes = np.round(y * 2.0 ** F["f"])
    assert np.array_equal(codes, y * 2.0 ** F["f"]), "LUT output not on the quantizer grid"
    return codes.astype(np.int64)


def extract(run: str, hw: HWFormats | None = None) -> tuple[Program, object]:
    """Program (+ the loaded QAT Keras model) of a trained student directory."""
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    from keras import ops

    from ...model import forecast_mask, rope_tables
    from .. import dense as rtl

    hw = hw or HWFormats()
    cfg, _, model, preset = rtl._student(run)
    if cfg.variate_every != 0:
        raise NotImplementedError("the engine supports channel-independent students (variate_every = 0)")
    q = resolve(preset)
    prog = Program(cfg=cfg.to_dict(), hw=asdict(hw), q=q)
    d = prog.dims

    for lyr in model.layers:
        kq = getattr(lyr, "kernel_quantizer_internal", None)
        if kq is None:
            continue
        W, fw = _exact_int(kq(lyr.kernel))
        b, fb = None, 0
        if lyr.use_bias:
            b, fb = _exact_int(lyr.bias_quantizer_internal(lyr.bias))
        prog.dense[lyr.name] = dict(W=W, fw=fw, b=b, fb=fb)

    Fg = prog.F("gain")
    from qkeras.quantizers import get_quantizer
    for lyr in model.layers:
        if type(lyr).__name__ == "QRMSNorm":
            g = np.asarray(get_quantizer(q["gain"])(lyr.scale), np.float64)
            prog.gains[lyr.name] = np.round(g * 2.0 ** Fg["f"]).astype(np.int64)
            eps = float(lyr.epsilon)

    # lookup tables, bit-identical to the model's float32 ops + quantizers
    Fms = prog.F("ms")
    prog.luts["rsqrt"] = _keras_lut(lambda x: ops.rsqrt(x + eps), _code_range(Fms), Fms["f"], q["rsqrt"])
    Fs = prog.F("score")
    s_codes = np.arange(Fs["lo"], 1, dtype=np.int64)                   # after max-subtraction: <= 0
    prog.luts["exp"] = _keras_lut(ops.exp, s_codes, Fs["f"], q["exp"])
    Fe = prog.F("exp")
    sums = np.arange(1, d["S"] * Fe["hi"] + 1, dtype=np.int64)       # sum of unmasked exp codes
    prog.luts["inv"] = _keras_lut(lambda x: 1.0 / x, sums, Fe["f"], q["inv"])
    if cfg.act_fn not in ("relu",):
        Fi = prog.F("lut_in")
        prog.luts["act"] = _keras_lut(lambda x: x * ops.sigmoid(x), _code_range(Fi), Fi["f"], q["lut_out"])
    if cfg.gate_fn not in ("hard_sigmoid",):
        Fi = prog.F("lut_in")
        prog.luts["gate"] = _keras_lut(ops.sigmoid, _code_range(Fi), Fi["f"], q["sig_out"])

    tok = np.asarray(model.get_layer("forecast_query").token, np.float64).reshape(-1)
    prog.token = rhu_shift(np.round(tok * 2.0 ** (hw.tok_f + 20)).astype(np.int64), 20)
    cos, sin = rope_tables(cfg.spec(), d["S"])
    prog.cos = rhu_shift(np.round(np.asarray(cos, np.float64) * 2.0 ** (hw.rope_f + 20)).astype(np.int64), 20)
    prog.sin = rhu_shift(np.round(np.asarray(sin, np.float64) * 2.0 ** (hw.rope_f + 20)).astype(np.int64), 20)
    prog.invsqrt = int(np.round(2.0 ** hw.invsqrt_f / np.sqrt(d["hd"])))
    prog.mask = np.asarray(forecast_mask(d["L"], d["h"]), bool)
    return prog, model


# ----------------------------------------------------------------------------- forward
def quantize_input(scaled: np.ndarray, hw: dict) -> np.ndarray:
    """Model input (asinh-scaled floats) -> input codes (the hardware input word)."""
    h = HWFormats(**hw)
    F = dict(f=h.in_f, lo=-(1 << (h.in_bits - 1)), hi=(1 << (h.in_bits - 1)) - 1)
    return np.clip(rne_shift(np.round(np.asarray(scaled, np.float64) * 2.0 ** (F["f"] + 24)).astype(np.int64), 24),
                   F["lo"], F["hi"])


def _dense(prog, name, x, fx):
    """Exact ``x @ W + b``; returns (values, fractional bits)."""
    d = prog.dense[name]
    y = np.einsum("...i,io->...o", x, d["W"])
    fy = fx + d["fw"]
    if d["b"] is not None:
        y = y + (d["b"] << (fy - d["fb"]) if fy >= d["fb"] else rne_shift(d["b"], d["fb"] - fy))
    return y, fy


def _align(x, fx, f):
    return x << (f - fx) if f >= fx else rhu_shift(x, fx - f)


def _act(prog, x, fx, kind: str):
    """Element-wise activation as the QAT model quantises it; returns (codes, f)."""
    if kind == "relu":
        bits, integer = (int(v) for v in prog.q["lut_out"].split("(")[1].split(")")[0].split(",")[:2])
        F = dict(f=bits - integer, lo=0, hi=(1 << bits) - 1)
        return clip(rne_shift(np.maximum(x, 0), fx - F["f"]), F), F["f"]
    Fi = prog.F("lut_in")
    idx = requant(x, fx, Fi) - Fi["lo"]
    Fo = prog.F("lut_out")
    return prog.luts["act"][idx], Fo["f"]


def _gate(prog, x, fx, kind: str):
    Fs = prog.F("sig_out")
    if kind == "hard_sigmoid":            # keras 3: clip(x / 6 + 0.5, 0, 1), then sig_out
        num = (x << Fs["f"]) + (3 << (fx + Fs["f"]))      # (x/6 + 1/2) * 2^f * 6 * 2^fx
        return clip(rne_div(num, 6 << fx), Fs), Fs["f"]
    Fi = prog.F("lut_in")
    return prog.luts["gate"][requant(x, fx, Fi) - Fi["lo"]], Fs["f"]


def _rmsnorm(prog, name, x, fx):
    Fa, Fms, Frs, Fg = prog.F("a"), prog.F("ms"), prog.F("rsqrt"), prog.F("gain")
    D = x.shape[-1]
    fm = min(fx, prog.hw["ms_f"])
    xm = rhu_shift(x, fx - fm)                                         # exact for a-grid rows
    ss = np.sum(xm * xm, axis=-1, keepdims=True)                       # f = 2 fm
    ms = clip(rne_div(ss, D << (2 * fm - Fms["f"])), Fms)          # needs 2 fm >= f_ms
    r = prog.luts["rsqrt"][ms - Fms["lo"]]
    y = x * r * prog.gains[name]                                       # f = fx + frs + fg
    return requant(y, fx + Frs["f"] + Fg["f"], Fa), Fa["f"]


def _rope(prog, x, fx):
    """x (..., S, H, hd) -> rotated at qrot_f."""
    hw, n = prog.hw, prog.dims["n"]
    half = x.shape[-1] // 2
    c = prog.cos[:, None, :]
    s = prog.sin[:, None, :]
    y = x << hw["rope_f"]
    x1, x2 = x[..., :n], x[..., half:half + n]
    y[..., :n] = x1 * c - x2 * s
    y[..., half:half + n] = x2 * c + x1 * s
    return rhu_shift(y, fx + hw["rope_f"] - hw["qrot_f"]), hw["qrot_f"]


def _attention(prog, qh, kh, vh, fq, fv):
    """q, k at qrot_f, v at fv: (N, S, H, hd) -> (N, S, H*hd) a-format codes."""
    hw = prog.hw
    Fs, Fe, Fi, Fp, Fa = prog.F("score"), prog.F("exp"), prog.F("inv"), prog.F("prob"), prog.F("a")
    s = rhu_shift(np.einsum("nqhd,nkhd->nhqk", qh, kh), 2 * fq - hw["score_f"])   # q.k at score_f
    s = rhu_shift(s * prog.invsqrt, hw["invsqrt_f"])                   # / sqrt(hd), at score_f
    m = prog.mask[None, None]
    smax = np.max(np.where(m, s, np.iinfo(np.int64).min // 4), axis=-1, keepdims=True)
    sc = np.where(m, clip(rne_shift(s - smax, hw["score_f"] - Fs["f"]), Fs), Fs["lo"])   # <= 0
    e = np.where(m, prog.luts["exp"][sc - Fs["lo"]], 0)
    inv = prog.luts["inv"][np.sum(e, axis=-1, keepdims=True) - 1]
    p = clip(rne_shift(e * inv, Fe["f"] + Fi["f"] - Fp["f"]), Fp)
    o = np.einsum("nhqk,nkhd->nqhd", p, vh)                             # f = fp + fv
    o = requant(o, Fp["f"] + fv, Fa)
    return o.reshape(o.shape[0], o.shape[1], -1), Fa["f"]


def forward(prog: Program, x_codes: np.ndarray, trace: dict | None = None) -> np.ndarray:
    """Input codes ``(N, V, ctx)`` -> ``head_post`` codes ``(N, V, h*P, Q)``.

    ``trace`` (optional dict) collects intermediate codes by Keras layer name, as
    ``(codes (N*V, ...), f)``, for ``check``.
    """
    d, hw, c = prog.dims, prog.hw, prog.cfg
    Fa = prog.F("a")
    N, V, _ = x_codes.shape
    P, L, S, D, H, hd, M = d["P"], d["L"], d["S"], d["D"], d["H"], d["hd"], d["M"]
    t = trace if trace is not None else {}
    fin = HWFormats(**hw).in_f
    x = np.asarray(x_codes, np.int64).reshape(N * V, L, P)
    pin = np.concatenate([x, np.zeros_like(x)], axis=-1)

    # embedding residual block
    h1, f1 = _dense(prog, "embed_hidden", pin, fin)
    a1, fa1 = _act(prog, h1, f1, c["act_fn"])
    o1, fo1 = _dense(prog, "embed_output", a1, fa1)
    r1, fr1 = _dense(prog, "embed_residual", pin, fin)
    f = max(fo1, fr1)
    e = requant(_align(o1, fo1, f) + _align(r1, fr1, f), f, Fa)
    t["embed_post"] = (e, Fa["f"])

    # residual stream at tok_f; forecast-query tokens appended
    fres = hw["tok_f"]
    xs = np.concatenate([e << (fres - Fa["f"]), np.broadcast_to(prog.token, (N * V, d["h"], D))], axis=1)

    for b in range(d["B"]):
        n, fn = _rmsnorm(prog, f"b{b}_attn_norm0", xs, fres)
        t[f"b{b}_attn_norm0"] = (n, fn)
        qkv, fq = _dense(prog, f"b{b}_attn0_wQKV", n, fn)
        qkv = qkv.reshape(N * V, S, 3, hd, H).transpose(0, 1, 2, 4, 3)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        qr, fr = _rope(prog, q, fq)
        kr, _ = _rope(prog, k, fq)
        o, fo = _attention(prog, qr, kr, v, fr, fq)
        t[f"b{b}_attn0_sdpa"] = (o, fo)
        g, fgd = _dense(prog, f"b{b}_attn0_gate", n, fn)
        gs, fgs = _gate(prog, g, fgd, c["gate_fn"])
        t[f"b{b}_attn0_gate_sig_q"] = (gs, fgs)
        wo, fwo = _dense(prog, f"b{b}_attn0_wO", o * gs, fo + fgs)
        po = requant(wo, fwo, Fa)
        t[f"b{b}_attn0_post"] = (po, Fa["f"])
        xs = xs + (po << (fres - Fa["f"]))

        n, fn = _rmsnorm(prog, f"b{b}_mlp_norm", xs, fres)
        t[f"b{b}_mlp_norm"] = (n, fn)
        w12, f12 = _dense(prog, f"b{b}_w12", n, fn)
        gate, up = w12[..., :M], w12[..., M:]
        ua, fua = _act(prog, up, f12, c["act_fn"])
        t[f"b{b}_silu"] = (ua, fua)
        w3, f3 = _dense(prog, f"b{b}_w3", gate * ua, f12 + fua)
        pm = requant(w3, f3, Fa)
        t[f"b{b}_mlp_post"] = (pm, Fa["f"])
        xs = xs + (pm << (fres - Fa["f"]))

    n, fn = _rmsnorm(prog, "out_norm", xs, fres)
    t["out_norm"] = (n, fn)
    dt = n[:, L:, :]
    hh, fh = _dense(prog, "head_hidden", dt, fn)
    ha, fha = _act(prog, hh, fh, c["act_fn"])
    ho, fho = _dense(prog, "head_output", ha, fha)
    hr, fhr = _dense(prog, "head_residual", dt, fn)
    f = max(fho, fhr)
    raw = requant(_align(ho, fho, f) + _align(hr, fhr, f), f, Fa)
    t["head_post"] = (raw, Fa["f"])
    return raw.reshape(N, V, d["h"] * P, d["Q"])


# ----------------------------------------------------------------------------- CPU side
def quantiles_from_raw(raw_codes: np.ndarray, prog: Program) -> np.ndarray:
    """``head_post`` codes -> normalised quantiles (``MonotoneQuantiles`` in float64)."""
    from qkeras.distill.layers import quantile_increment_widths

    knots = list(prog.cfg["knots"])
    raw = np.asarray(raw_codes, np.float64) * 2.0 ** -prog.F("a")["f"]
    w = quantile_increment_widths(knots, prog.cfg["width_factor"]).astype(np.float64)
    m = knots.index(0.5)
    med = raw[..., m:m + 1]
    sp = np.logaddexp(0.0, raw) * w
    lower = med - np.flip(np.cumsum(np.flip(sp[..., :m], -1), -1), -1)
    upper = med + np.cumsum(sp[..., m + 1:], -1)
    return np.concatenate([lower, med, upper], -1)


# ----------------------------------------------------------------------------- check vs Keras
def _ties(x, f_from: int, f_to: int):
    """Elements whose exact value lies exactly half-way between two codes of ``f_to``."""
    s = f_from - f_to
    if s <= 0:
        return np.zeros(np.shape(x), bool)
    x = np.asarray(x, np.int64)
    return (x - ((x >> s) << s)) == (np.int64(1) << (s - 1))


def _cmp(name, mine, ref_float, f, tie=None) -> dict:
    ref = np.round(np.asarray(ref_float, np.float64) * 2.0 ** f).astype(np.int64).reshape(mine.shape)
    bad = ref != mine
    tie = np.zeros_like(bad) if tie is None else np.broadcast_to(tie, bad.shape)
    return dict(n=int(bad.size), differ=int(bad.sum()), tie_flips=int((bad & tie).sum()),
                other=int((bad & ~tie).sum()), max_lsb=int(np.abs(ref - mine).max()))


def check_local(prog: Program, model, x_scaled: np.ndarray) -> dict:
    """Every stage of the integer program fed with the Keras model's own input to that stage.

    Differences are split into ``tie_flips`` (the exact value sits half-way between two codes;
    qkeras' tie handling depends on graph compilation) and ``other``.
    """
    import keras

    d, hw, c = prog.dims, prog.hw, prog.cfg
    Fa = prog.F("a")
    codes = quantize_input(x_scaled, hw)
    xq = (codes * 2.0 ** -HWFormats(**hw).in_f).astype(np.float32)
    B = d["B"]
    names = ["embed_post", "out_norm", "head_post"]
    for b in range(B):
        names += [f"b{b}_attn_norm0", f"b{b}_attn0_sdpa", f"b{b}_attn0_gate_sig_q", f"b{b}_attn0_post",
                  f"b{b}_mlp_norm", f"b{b}_silu", f"b{b}_mlp_post", f"b{b}_res_mlp" if b else "b0_res_mlp",
                  f"b{b}_res0"]
    names += ["forecast_query"]
    names = list(dict.fromkeys(names))
    outs = dict(zip(names, keras.Model(model.inputs, [model.get_layer(n).output for n in names])
                    .predict(xq, batch_size=256, verbose=0)))
    N, V, _ = codes.shape
    S, D, H, hd, M, L = d["S"], d["D"], d["H"], d["hd"], d["M"], d["L"]
    fres = hw["tok_f"]

    def resid(name):          # Keras residual stream (a-grid rows exact, token rows float) -> tok_f codes
        v = np.asarray(outs[name], np.float64).reshape(N * V, S, D)
        return np.round(v * 2.0 ** fres).astype(np.int64)

    def acodes(name, shape=None):
        v = np.asarray(outs[name], np.float64)
        return np.round(v * 2.0 ** Fa["f"]).astype(np.int64).reshape(shape or (N * V, S, -1))

    res = {}
    # embedding
    trace: dict = {}
    forward(prog, codes, trace)
    xin = codes.reshape(N * V, d["L"], d["P"])
    pin = np.concatenate([xin, np.zeros_like(xin)], axis=-1)
    fin = HWFormats(**hw).in_f
    h1, f1 = _dense(prog, "embed_hidden", pin, fin)
    a1, fa1 = _act(prog, h1, f1, c["act_fn"])
    o1, fo1 = _dense(prog, "embed_output", a1, fa1)
    r1, fr1 = _dense(prog, "embed_residual", pin, fin)
    fe = max(fo1, fr1)
    esum = _align(o1, fo1, fe) + _align(r1, fr1, fe)
    res["embed_post"] = _cmp("embed_post", trace["embed_post"][0], outs["embed_post"], Fa["f"],
                             _ties(esum, fe, Fa["f"]))
    prev = "forecast_query"
    for b in range(B):
        n, fn = _rmsnorm(prog, f"b{b}_attn_norm0", resid(prev), fres)
        res[f"b{b}_attn_norm0"] = _cmp("", n, outs[f"b{b}_attn_norm0"], fn)
        nk = acodes(f"b{b}_attn_norm0")
        qkv, fq = _dense(prog, f"b{b}_attn0_wQKV", nk, Fa["f"])
        qkv = qkv.reshape(N * V, S, 3, hd, H).transpose(0, 1, 2, 4, 3)
        qr, fr = _rope(prog, qkv[:, :, 0], fq)
        kr, _ = _rope(prog, qkv[:, :, 1], fq)
        o, fo = _attention(prog, qr, kr, qkv[:, :, 2], fr, fq)
        res[f"b{b}_attn0_sdpa"] = _cmp("", o, outs[f"b{b}_attn0_sdpa"], fo)
        g, fgd = _dense(prog, f"b{b}_attn0_gate", nk, Fa["f"])
        gs, fgs = _gate(prog, g, fgd, c["gate_fn"])
        res[f"b{b}_attn0_gate_sig_q"] = _cmp("", gs, outs[f"b{b}_attn0_gate_sig_q"], fgs)
        ok = acodes(f"b{b}_attn0_sdpa")
        gk = np.round(np.asarray(outs[f"b{b}_attn0_gate_sig_q"], np.float64) * 2.0 ** fgs).astype(np.int64)
        wo, fwo = _dense(prog, f"b{b}_attn0_wO", ok * gk.reshape(ok.shape), Fa["f"] + fgs)
        res[f"b{b}_attn0_post"] = _cmp("", requant(wo, fwo, Fa), outs[f"b{b}_attn0_post"], Fa["f"],
                                       _ties(wo, fwo, Fa["f"]))
        n, fn = _rmsnorm(prog, f"b{b}_mlp_norm", resid(f"b{b}_res0"), fres)
        res[f"b{b}_mlp_norm"] = _cmp("", n, outs[f"b{b}_mlp_norm"], fn)
        nk = acodes(f"b{b}_mlp_norm")
        w12, f12 = _dense(prog, f"b{b}_w12", nk, Fa["f"])
        ua, fua = _act(prog, w12[..., M:], f12, c["act_fn"])
        res[f"b{b}_silu"] = _cmp("", ua, outs[f"b{b}_silu"], fua, _ties(np.maximum(w12[..., M:], 0), f12, fua))
        uk = np.round(np.asarray(outs[f"b{b}_silu"], np.float64) * 2.0 ** fua).astype(np.int64).reshape(ua.shape)
        w3, f3 = _dense(prog, f"b{b}_w3", w12[..., :M] * uk, f12 + fua)
        res[f"b{b}_mlp_post"] = _cmp("", requant(w3, f3, Fa), outs[f"b{b}_mlp_post"], Fa["f"], _ties(w3, f3, Fa["f"]))
        prev = f"b{b}_res_mlp"
    n, fn = _rmsnorm(prog, "out_norm", resid(prev), fres)
    res["out_norm"] = _cmp("", n, outs["out_norm"], fn)
    dt = acodes("out_norm")[:, L:, :]
    hh, fh = _dense(prog, "head_hidden", dt, Fa["f"])
    ha, fha = _act(prog, hh, fh, c["act_fn"])
    ho, fho = _dense(prog, "head_output", ha, fha)
    hr, fhr = _dense(prog, "head_residual", dt, Fa["f"])
    f = max(fho, fhr)
    tot = _align(ho, fho, f) + _align(hr, fhr, f)
    res["head_post"] = _cmp("", requant(tot, f, Fa), outs["head_post"], Fa["f"], _ties(tot, f, Fa["f"]))
    return res


def check_quality(prog: Program, model, split: dict, knots, season: int, names=None) -> dict:
    """Test metrics of the integer program and of the Keras QAT model on the same windows."""
    from ... import data as dd
    from ... import teacher as tch

    codes = quantize_input(split["xs"], prog.hw)
    zq = quantiles_from_raw(forward(prog, codes), prog)
    zk = model.predict(split["xs"], batch_size=256, verbose=0)
    out = {}
    for tag, z in (("integer_program", zq), ("keras_qat", zk)):
        pred = tch.from_student_space(z, split["loc"], split["sc"])
        m = dd.summarize(pred, split["y"], split["x"], knots, season, names)
        out[tag] = {k: m[k] for k in ("mase", "wql", "coverage_80")}
    return out
