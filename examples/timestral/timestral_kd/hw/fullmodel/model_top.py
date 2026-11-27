"""model_top / full-model integer golden (phase P3a).

P3a delivers the *integer golden model* of the whole compressed student plus a harness that
checks it against the trained QAT model.  This golden is the reference the RTL ``model_top``
must reproduce bit-exactly in P3b -- getting the op order right (and *measuring* the gap to
the float model here) de-risks the RTL exactly as ``fixedpoint._validate`` de-risked the
per-op blocks.

Arithmetic policy (mirrors the serial export in ``rtl.py``):
  * every dense uses the exact QAT-quantised kernel/bias from ``rtl.layer_specs``, fed the
    layer's own per-input fixed-point format ``kif`` (``[scaled|missing]``, ``A``, ``RELU``,
    ``WO`` -- taken straight from ``rtl.layer_input_formats`` via ``layer_specs``), and
    truncated to ``OUT_FRAC`` fractional bits with ``floor`` (TRN) like the digit-serial RTL;
  * every non-dense op is the already-Verilator-proven ``fixedpoint.*_ref`` (rmsnorm,
    softmax, rope, quantile head), fed and read in *value* space;
  * explicit quantizers (SiLU LUT, hard-sigmoid gate, monotone-quantile head) use the qkeras
    ``floor(x+0.5)`` convention via ``fixedpoint.to_code``.

Because the golden reproduces the RTL's TRN dense and the hardware-chosen QOUT format it will
*not* equal the float32 QAT forward bit-for-bit; ``check`` reports the deviation in QOUT LSBs
(the honest gate).  The bit-exact gate (golden == RTL) lands in P3b.

Run (needs the qkeras env for the model + weights):

    cd experimental/tsfm
    python -m timestral_kd.hw.fullmodel.model_top check --run runs/students/kdd/B_d16_L1_b3_ci_p70 --n 16
"""

from __future__ import annotations

import argparse

import numpy as np

from . import fixedpoint as fp

# --- graph constants (FULLMODEL.md; student B_d16_L1_b3_ci_p70 / kdd) --------------------
G = dict(V=3, P=16, D=16, H=2, M=32, L=16, h=4, S=20, Q=3, hd=8)
OUT_FRAC = 4                 # serial-export out_frac (dense TRN granularity)


# ----------------------------------------------------------------------------- quantisation
def _fmt(name: str) -> dict:
    return fp.fmt(fp.B3[name])


def _quant(x, spec: str):
    F = fp.fmt(spec)
    return fp.to_value(fp.to_code(x, F), F)


def _quant_kif(x, kif):
    """Round each last-axis column to its own (k, i, f) input format (per-column)."""
    k, i, f = (np.asarray(v).astype(int) for v in kif)
    bits = k + i + f
    lo = np.where(k > 0, -(2.0 ** (bits - 1)), 0.0)
    hi = np.where(k > 0, 2.0 ** (bits - 1) - 1, 2.0 ** bits - 1)
    code = np.clip(np.floor(np.asarray(x, np.float64) * 2.0 ** f + 0.5), lo, hi)
    return code * 2.0 ** (-f)


def _dense(x, W, b=None, out_frac: int = OUT_FRAC):
    """Constant-weight dense with TRN to ``out_frac`` (matches the digit-serial RTL)."""
    y = np.asarray(x, np.float64) @ np.asarray(W, np.float64)
    if b is not None:
        y = y + np.asarray(b, np.float64)
    return np.floor(y * 2.0 ** out_frac) / 2.0 ** out_frac


# ----------------------------------------------------------------------------- nonlinear ops
def _silu_act(x):
    """Quantized SiLU (QUnary: lut_in -> silu -> lut_out)."""
    xq = _quant(x, fp.B3["lut_in"])
    return _quant(xq / (1.0 + np.exp(-xq)), fp.B3["lut_out"])


def _hard_sigmoid(x):
    """quantized hard_sigmoid to sig_out (5,0,keep_negative=0)."""
    return _quant(np.clip(0.2 * x + 0.5, 0.0, 1.0), fp.B3["sig_out"])


def _rmsnorm(x, scale, luts):
    fa, fg = _fmt("a"), _fmt("gain")
    X = fp.to_code(x, fa)
    Gc = fp.to_code(scale, fg)
    return fp.to_value(fp.rmsnorm_ref(X, Gc, luts), fa)


def _softmax(scores, mask, luts):
    return fp.to_value(fp.softmax_ref(scores, luts, mask=mask), _fmt("prob"))


def _rope_codes():
    cos_f, sin_f = fp.rope_tables_ref(G["S"], 10000.0, fp.ROPE_N)
    fc = fp.fmt(fp.ROPE_CS)
    return fp.to_code(cos_f, fc), fp.to_code(sin_f, fc)


def _rope_val(x_hs, cos_c, sin_c):
    """x_hs (S, hd) a-format values -> partial rotate-half values (via proven rope_ref)."""
    fa = _fmt("a")
    X = fp.to_code(x_hs, fa)
    Y = np.stack([fp.rope_ref(X[s], cos_c[s], sin_c[s]) for s in range(X.shape[0])])
    return fp.to_value(Y, fa)


# ----------------------------------------------------------------------------- load
def load_model(run: str):
    """cfg, metrics, model, preset, and per-layer dense specs {name: (W, b, kif)}."""
    from .. import dense as rtl
    cfg, met, model, preset = rtl._student(run)
    _, _, specs = rtl.layer_specs(run)
    W = {n: (w, b, kif) for n, w, b, kif in specs}
    return cfg, met, model, preset, W


def _scale(model, name):
    return np.asarray(model.get_layer(name).scale, np.float64)


# ----------------------------------------------------------------------------- golden
def forward_one(x, W, model, luts, knots, width_factor, mask):
    """Integer golden forward for ONE window.

    ``x``: (V, L*P) *scaled* values (the model input, pre-quantisation).  Returns QOUT-format
    integer codes ``(V, h*P, Q)`` -- the exact interface the RTL quantile output exposes.
    """
    V, P, L, h, D, H, hd, M, S, Q = (G[k] for k in ("V", "P", "L", "h", "D", "H", "hd", "M", "S", "Q"))

    def Dm(name, xin):
        w, b, kif = W[name]
        return _dense(_quant_kif(xin, kif), w, b)

    # ---- embed residual block -----------------------------------------------------------
    patch = np.asarray(x, np.float64).reshape(V, L, P)
    pin = np.concatenate([patch, np.zeros((V, L, P))], axis=-1)            # [vals | missing]
    eh = _silu_act(Dm("embed_hidden", pin))
    eo = Dm("embed_output", eh)
    er = Dm("embed_residual", pin)
    xs = eo + er                                                           # (V, L, D)

    # ---- forecast query tokens (learned, appended along the sequence axis) --------------
    tok = np.asarray(model.get_layer("forecast_query").token, np.float64).reshape(1, 1, D)
    xs = np.concatenate([xs, np.broadcast_to(tok, (V, h, D))], axis=1)     # (V, S, D)

    # ---- block 0: time attention + SwiGLU MLP -------------------------------------------
    n = _rmsnorm(xs, _scale(model, "b0_attn_norm0"), luts)
    qkv = Dm("b0_attn0_wQKV", n).reshape(V, S, 3, hd, H).transpose(0, 1, 2, 4, 3)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]                     # (V, S, H, hd)
    g = _hard_sigmoid(Dm("b0_attn0_gate", n))                              # (V, S, D)

    cos_c, sin_c = _rope_codes()
    q = np.stack([np.stack([_rope_val(q[vi, :, hi], cos_c, sin_c) for hi in range(H)], -2)
                  for vi in range(V)])
    k = np.stack([np.stack([_rope_val(k[vi, :, hi], cos_c, sin_c) for hi in range(H)], -2)
                  for vi in range(V)])

    scale = 1.0 / np.sqrt(hd)
    o = np.zeros((V, S, H, hd))
    for vi in range(V):
        for hi in range(H):
            scores = (q[vi, :, hi] @ k[vi, :, hi].T) * scale
            o[vi, :, hi] = _softmax(scores, mask, luts) @ v[vi, :, hi]
    o = o.reshape(V, S, D) * g
    o = Dm("b0_attn0_wO", o)
    xs = xs + o                                                            # attention residual

    n = _rmsnorm(xs, _scale(model, "b0_mlp_norm"), luts)
    w12 = Dm("b0_w12", n)                                                  # (V, S, 2M)
    gate, up = w12[..., :M], w12[..., M:]
    m = Dm("b0_w3", gate * _silu_act(up))
    xs = xs + m                                                            # MLP residual

    # ---- out_norm, decoder tokens, head residual block ----------------------------------
    xs = _rmsnorm(xs, _scale(model, "out_norm"), luts)
    xt = xs[:, L:, :]                                                      # (V, h, D)
    hh = _silu_act(Dm("head_hidden", xt))
    ho = Dm("head_output", hh)
    hr = Dm("head_residual", xt)
    raw = (ho + hr).reshape(V, h * P, Q)                                   # (V, h*P, Q)

    # ---- monotone quantiles via the hardware integer LUT + cumsum -----------------------
    raw_codes = fp.to_code(raw, _fmt("a"))
    return fp.quantiles_hw_ref(raw_codes, knots, width_factor, fp.softplus_lut())


# ----------------------------------------------------------------------------- harness
def scaled_inputs(n: int, seed: int = 0) -> np.ndarray:
    """Random *scaled* windows in the model's input range (asinh of ~N(0,1) context).

    Avoids a dataset dependency: the golden and the float model are both fed these directly,
    which is exactly what the RTL sees at ``model_inp`` (already scale_context'd).
    """
    rng = np.random.default_rng(seed)
    V, L, P = G["V"], G["L"], G["P"]
    raw = rng.normal(0.0, 1.0, (n, V, L, P))
    return np.arcsinh(raw).reshape(n, V, L * P).astype(np.float64)


def check(run: str, n: int = 16, seed: int = 0) -> dict:
    """Compare the integer golden against the trained model's float forward (QOUT LSBs)."""
    from ...model import forecast_mask
    cfg, met, model, preset, W = load_model(run)
    luts = fp.build_luts(D=G["D"])
    knots = tuple(float(k) for k in cfg.knots)
    wf = float(cfg.width_factor)
    mask = forecast_mask(G["L"], G["h"])

    xs = scaled_inputs(n, seed)
    ref = np.asarray(model(xs.astype(np.float32), training=False))         # (n,V,h*P,Q) normalised
    fq = fp.fmt(fp.QOUT)
    worst = 0.0
    for i in range(n):
        gold = fp.to_value(forward_one(xs[i], W, model, luts, knots, wf, mask), fq)
        worst = max(worst, float(np.abs(gold - ref[i]).max()) * 2.0 ** fq["f"])
    print(f"golden vs model: max |diff| = {worst:.2f} QOUT LSB  "
          f"(1 LSB = 2^-{fq['f']} = {2.0 ** -fq['f']:.4f}); n={n}")
    return {"max_lsb": worst, "qout_frac": fq["f"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    s = ap.add_subparsers(dest="cmd", required=True)
    c = s.add_parser("check")
    c.add_argument("--run", required=True)
    c.add_argument("--n", type=int, default=16)
    c.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "check":
        check(a.run, a.n, a.seed)


if __name__ == "__main__":
    main()
