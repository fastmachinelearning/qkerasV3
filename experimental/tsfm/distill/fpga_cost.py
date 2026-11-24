"""First-order GateMate CCGM1A1 cost model for student transformers.

These are planning estimates to rank student configurations, not synthesis results.
Validate a chosen design with yosys + nextpnr (Kolibri flow, export_da4ml.py or an HLS
export).  Every constant below is an explicit, editable assumption.

Two deployment classes
----------------------
A  folded:   weights and activations in BRAM, ``n_mac`` shared multiply-accumulate units,
             (the limit of class B for a large reuse factor, but with weights in RAM)
             element-wise functions as BRAM lookup tables.  Area ~ independent of model
             size; the model must fit the BRAM; latency = MACs / n_mac / f.
B  streamed: every dense layer is one constant-weight shift-add block (CSD digits of the
             quantised weights -> adders), tokens stream through at one token per cycle;
             element-wise functions as logic LUTs, one per lane; attention keys/values are
             buffered in BRAM and scores use real (activation x activation) multipliers.
             ``reuse`` time-multiplexes all per-token units (cf. hls4ml reuse_factor);
             ``min_reuse_to_fit`` is the smallest power of two that fits the CPE budget.
"""

from __future__ import annotations

import math

import numpy as np

from . import student as stu

GATEMATE = {
    "cpe": 20480,                   # CCGM1A1 programmable elements
    "bram_bits": 32 * 40 * 1024,    # 32 x 40 kbit block RAM (datasheet; verify for your board)
    "lut4_to_cpe": 1.83,            # measured in the logicnet-fpga paper (4-LUT ~ 1.83 CPEs)
    "cpe_per_adder_bit": 1.0,       # assumption for multipliers: one CPE per full-adder bit
    "f_clk_hz": 50e6,               # assumption for a pipelined design on the A1
    "util_limit": 0.8,              # keep 20 % for routing / control
}


def _mult_cpe(a_bits: int, b_bits: int, g=GATEMATE) -> float:
    """Array multiplier ~ a*b full-adder cells."""
    return a_bits * b_bits * g["cpe_per_adder_bit"]


def _lut_logic_cpe(in_bits: int, out_bits: int, g=GATEMATE) -> float:
    """Table with ``in_bits`` inputs per output bit as a 4-LUT tree (upper bound)."""
    return out_bits * max(1, 2 ** max(in_bits - 4, 0)) * g["lut4_to_cpe"]


def _csd_digits(v: np.ndarray) -> np.ndarray:
    """Number of non-zero canonical-signed-digit digits of integers ``v``."""
    v = np.abs(v.astype(np.int64))
    n = np.zeros_like(v)
    while np.any(v):
        odd = (v & 1).astype(bool)
        # CSD: if v mod 4 == 3 use -1 and carry
        r = np.where(odd, 2 - (v & 3), 0)
        n += (r != 0)
        v = (v - r) >> 1
    return n


def _exact_frac_bits(k: np.ndarray, max_f: int = 24) -> int:
    """Smallest f with k * 2**f integer (per-channel power-of-two scales make f > w - 2)."""
    for f in range(max_f + 1):
        if np.allclose(k * 2.0 ** f, np.round(k * 2.0 ** f)):
            return f
    return max_f


def _quantized_kernels(qmodel, frac_bits: int | None = None):
    """Integer-scaled quantised kernels of all dense layers (for CSD counting).

    ``frac_bits=None`` (default) scales every kernel exactly.  A fixed ``frac_bits`` rounds
    and silently drops weights below 2**-frac_bits (the sweep-v1 bug: ``auto_po2`` scales
    put most 3-4 bit weights below 0.25, so 5.7k non-zero weights were counted as 0.9k).
    """
    out = []
    for lyr in qmodel.layers:
        q = getattr(lyr, "kernel_quantizer_internal", None)
        if q is None or not hasattr(lyr, "kernel"):
            continue
        k = np.asarray(q(lyr.kernel), dtype=np.float64)
        f = _exact_frac_bits(k) if frac_bits is None else frac_bits
        out.append((lyr.name, np.round(k * 2.0 ** f)))
    return out


# Calibration against yosys synth_gatemate + nextpnr-himbaechel (CCGM1A1) of the 14 dense
# layers of B_d16_L1_b3 (KDD), da4ml 0.6 pipelined Verilog inside a shift-register harness
# (experimental/tsfm/distill/rtl, 2026-11-24): CPE = 1.59 x da4ml adder cost (max err 16 %),
# or 0.56 x CSD digits x (input bits + weight bits + log2 fan-in) without da4ml (max err 23 %).
CPE_PER_DA4ML_COST = 1.59
CPE_PER_DIGIT_BIT = 0.56


def estimate_dense_cpe(qmodel, in_bits: int, w_bits: int, use_da4ml: bool = True) -> dict:
    """Measured-calibrated CPE of all constant-weight dense layers (one instance each)."""
    kernels = _quantized_kernels(qmodel)
    digits = sum(int(_csd_digits(k).sum()) for _, k in kernels)
    nz = sum(int(np.count_nonzero(k)) for _, k in kernels)
    fallback = sum(int(_csd_digits(k).sum()) * (in_bits + w_bits + math.ceil(math.log2(max(k.shape[0], 2))))
                   for _, k in kernels) * CPE_PER_DIGIT_BIT
    res = {"csd_digits": digits, "nonzero_weights": nz, "dense_cpe": float(fallback), "method": "csd"}
    if use_da4ml:
        try:
            import da4ml.trace as tr
        except ImportError:
            return res
        cost = 0.0
        f_in = in_bits - 1 - 3                              # presets use quantized_bits(a, 3)
        for lyr in qmodel.layers:
            q = getattr(lyr, "kernel_quantizer_internal", None)
            if q is None or not hasattr(lyr, "kernel"):
                continue
            k = np.asarray(q(lyr.kernel), dtype=np.float64)
            n = k.shape[0]
            x = tr.FixedVariableArray.from_kif(np.ones(n, bool), np.full(n, 3), np.full(n, max(f_in, 0)))
            cost += float(tr.comb_trace(x, x @ k).cost)
        res.update(dense_cpe=cost * CPE_PER_DA4ML_COST, da4ml_cost=cost, method="da4ml")
    return res


def estimate(cfg: "stu.StudentConfig", V: int, qmodel=None, n_mac: int = 16, reuse: int = 1, g=GATEMATE) -> dict:
    ops = stu.count_ops(cfg, V)
    bits = stu.preset_bits(cfg.preset)
    w, a = bits["w"], bits["a"]
    li, lo = bits.get("lut_in", a), bits.get("lut_out", a)
    D, M, S, T = cfg.d_model, cfg.mlp_hidden, ops["seq_len"], ops["tokens"]
    n_attn = sum(1 if k == "time" else 2 for k in cfg.schedule())
    res = {"weight_bits": ops["params"] * w, "class": cfg.deployment, "assumptions": dict(g)}

    # ---------------- class A: folded
    n_tables = 3 + (cfg.act_fn == "silu") + (cfg.gate_fn == "sigmoid")   # exp, rsqrt, softplus (+ silu, sigmoid)
    table_bits = n_tables * (2 ** li) * lo
    act_bits = 2 * T * max(D, M, cfg.eh, cfg.hh) * a + 2 * T * D * a   # double-buffered activations + K/V
    bram_a = ops["params"] * w + table_bits + act_bits
    mac_cpe = n_mac * (_mult_cpe(max(w, a), a) + (w + a + 8) * g["cpe_per_adder_bit"])
    cpe_a = mac_cpe + 1500                                 # assumption: control/address generation
    cycles_a = (ops["weight_macs"] + ops["act_macs"]) / n_mac + sum(ops["lookups"].values()) / 2
    res["A"] = {"bram_bits": int(bram_a), "bram_util": bram_a / g["bram_bits"], "cpe": int(cpe_a),
                "cpe_util": cpe_a / g["cpe"], "n_mac": n_mac, "latency_us": cycles_a / g["f_clk_hz"] * 1e6,
                "fits": bool(bram_a <= g["util_limit"] * g["bram_bits"] and cpe_a <= g["util_limit"] * g["cpe"])}

    # ---------------- class B: token-streamed constant weights
    if qmodel is not None and stu.PRESETS.get(cfg.preset) is not None:
        dc = estimate_dense_cpe(qmodel, a, w)
        digits, nz, dense_cpe_v = dc["csd_digits"], dc["nonzero_weights"], dc["dense_cpe"]
        dense_method = dc["method"]
    else:                                                  # no model: assume dense 3/4-bit weights
        digits = int(ops["params"] * 1.0)
        nz = ops["params"]
        dense_cpe_v = digits * (a + w + 5) * CPE_PER_DIGIT_BIT
        dense_method = "params"
    dense_cpe = dense_cpe_v
    # element-wise lanes: one table per feature lane of each function site
    act_lanes = cfg.eh + cfg.n_blocks * M + cfg.hh
    lanes = {"exp": n_attn * cfg.num_heads, "rsqrt": n_attn + cfg.n_blocks + 1,
             "softplus": cfg.patch_size * (len(cfg.knots) - 1)}
    cheap_cpe = 0.0
    if cfg.act_fn == "silu":
        lanes["silu"] = act_lanes
    else:                                                  # relu: one AND per output bit
        cheap_cpe += act_lanes * lo * 0.5
    if cfg.gate_fn == "sigmoid":
        lanes["sigmoid"] = n_attn * D
    else:                                                  # hard sigmoid: shift-add + clamp
        cheap_cpe += n_attn * D * (a + 2)
    lut_cpe = sum(n * _lut_logic_cpe(li, lo, g) for n in lanes.values()) + cheap_cpe
    hd = D // cfg.num_heads
    attn_mult = n_attn * 2 * cfg.num_heads * hd            # QK and AV lanes, one key per cycle
    norm_mult = (n_attn + cfg.n_blocks + 1) * 2 * D        # square + scale per lane
    mult_cpe = (attn_mult + norm_mult) * _mult_cpe(a, a, g)
    fixed_cpe = lut_cpe + mult_cpe
    budget = g["util_limit"] * g["cpe"]
    kv_bits = n_attn * 2 * T * D * a

    def _b(r):
        # reuse r: every per-token unit (adder trees, lane tables, lane multipliers) is
        # time-multiplexed r times; 25 % operand-mux overhead and 500 CPEs control for r > 1
        par = (dense_cpe + fixed_cpe) / r
        cpe = par if r == 1 else par * 1.25 + 500
        cycles = T * r + V * S * S * len(cfg.schedule())    # token stream x reuse + per-query key loop
        return cpe, cycles

    cpe_b, cycles_b = _b(reuse)
    min_reuse = None
    for r in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        if _b(r)[0] <= budget:
            min_reuse = r
            break
    res["B"] = {"cpe": int(cpe_b), "cpe_util": cpe_b / g["cpe"], "dense_cpe": int(dense_cpe),
                "lut_cpe": int(lut_cpe), "mult_cpe": int(mult_cpe), "csd_digits": digits, "nonzero_weights": nz,
                "dense_method": dense_method,
                "bram_bits": int(kv_bits), "latency_us": cycles_b / g["f_clk_hz"] * 1e6, "reuse": reuse,
                "min_reuse_to_fit": min_reuse,
                "latency_us_at_min_reuse": (_b(min_reuse)[1] / g["f_clk_hz"] * 1e6) if min_reuse else None,
                "fits": bool(cpe_b <= budget and kv_bits <= g["util_limit"] * g["bram_bits"]),
                "fits_with_reuse": bool(min_reuse is not None and kv_bits <= g["util_limit"] * g["bram_bits"])}
    res["fits"] = res["A"]["fits"] if cfg.deployment == "A" else res["B"]["fits_with_reuse"]
    return res
