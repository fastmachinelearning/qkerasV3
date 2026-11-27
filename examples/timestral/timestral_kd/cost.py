"""First-order GateMate CCGM1A1 cost model for student transformers.

These are planning estimates to rank student configurations, not synthesis results.
Device constants and the calibrated dense-layer estimate live in ``qkeras.distill.hw.gatemate``;
this module adds the Timestral-specific accounting (attention, norms, tables, reuse).
Validate a chosen design with yosys + nextpnr (``timestral_kd.hw.rtl``).  Every constant below is an explicit, editable assumption.

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

from qkeras.distill.hw.gatemate import (  # noqa: F401  (re-exported for plots / callers)
    CPE_PER_DA4ML_COST,
    CPE_PER_DIGIT_BIT,
    GATEMATE,
    csd_digits as _csd_digits,
    estimate_dense_cpe,
    exact_frac_bits as _exact_frac_bits,
    integer_kernels as _quantized_kernels,
    lut_logic_cpe as _lut_logic_cpe,
    mult_cpe as _mult_cpe,
)

from . import student as stu


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
