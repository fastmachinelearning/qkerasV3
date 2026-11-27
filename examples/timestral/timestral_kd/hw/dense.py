"""RTL export and GateMate place & route of a trained class-B student's dense layers.

Stage 1 of the hardware flow: every constant-weight ``QDense`` of a trained student becomes
one da4ml shift-add block (exact QAT-quantised kernel and bias, input format taken from
the quantiser that feeds the layer in the QAT graph), is checked bit-exactly against numpy,
written as pipelined Verilog and implemented on GateMate CCGM1A1 with yosys
``synth_gatemate -luttree -nomx8`` + nextpnr-himbaechel inside the Kolibri shift-register
harness (``kolibri_fpga.targets.harness`` from logicnet-fpga).

Attention scores/softmax, RMSNorm and the quantile head are not part of stage 1.

Run through the pipeline CLI (paths are derived from ``<dataset>/<student>``)::

    python -m timestral_kd export kdd/B_d12_L1_b3_p70_ci
    python -m timestral_kd pnr    kdd/B_d12_L1_b3_p70_ci --jobs 7
    python -m timestral_kd report kdd/B_d12_L1_b3_p70_ci

or directly: ``python -m timestral_kd.hw.dense {export,pnr,collect,deploy} --run/--out ...``.

Needs ``da4ml`` (0.6), yosys >= 0.40 with synth_gatemate (``$YOSYS``), ``yowasp-nextpnr-himbaechel-gatemate``
on ``PATH`` and the Kolibri harness (``--kolibri-src`` or ``$KOLIBRI_FPGA_SRC``).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import numpy as np


# da4ml ships its own libomp; on macOS it crashes (segfault in cmvm) once TensorFlow has run ops in
# the same process.  With OMP_NUM_THREADS=1 set *before* the process starts it is stable, so on
# macOS the export re-runs itself in a child process with that environment.
def _needs_isolation() -> bool:
    return sys.platform == "darwin" and os.environ.get("OMP_NUM_THREADS") != "1"


def _isolated_export(argv: list[str], out: str) -> list[dict]:
    example = str(Path(__file__).resolve().parents[2])
    env = {**os.environ, "OMP_NUM_THREADS": "1",
           "PYTHONPATH": os.pathsep.join(filter(None, [example, os.environ.get("PYTHONPATH")]))}
    subprocess.run([sys.executable, "-m", "timestral_kd.hw.dense", "export", *argv], env=env, check=True)
    return json.load(open(os.path.join(out, "export.json")))["layers"]



# ----------------------------------------------------------------------------- formats
def _parse_q(spec: str) -> tuple[int, int, int]:
    """qkeras quantizer string -> da4ml (k, i, f)."""
    name, args = spec.split("(", 1)
    parts = [p.strip() for p in args.rstrip(")").split(",")]
    bits, integer = int(parts[0]), int(parts[1]) if len(parts) > 1 and "=" not in parts[1] else 0
    signed = "relu" not in name and "keep_negative=0" not in spec.replace(" ", "") \
        and "keep_negative=False" not in spec.replace(" ", "")
    if len(parts) > 3 and "=" not in parts[3]:
        signed = signed and parts[3] not in ("0", "False")
    k = int(signed)
    return k, integer, bits - k - integer


def layer_input_formats(cfg, preset: dict) -> dict:
    """Per-layer input (k, i, f) of the class-B student graph (see timestral_kd.model)."""
    A = _parse_q(preset["a"])
    rb, ri = [int(x) for x in preset["lut_out"].split("(")[1].split(")")[0].split(",")[:2]]
    RELU = (0, ri, rb - ri)
    G = _parse_q(preset["sig_out"])
    return {"A": A, "RELU": RELU, "GATE": G, "WO": (1, A[1] + G[1], A[2] + G[2])}


def _student(run: str):
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    from .. import student as st
    cfg = st.StudentConfig.from_dict(json.load(open(os.path.join(run, "config.json")))["student"])
    met = json.load(open(os.path.join(run, "metrics.json")))
    m = st.build_student(cfg, met["V"], preset=cfg.preset)
    m.load_weights(os.path.join(run, "qat.weights.h5"))
    return cfg, met, m, st.PRESETS[cfg.preset]


# ----------------------------------------------------------------------------- export
def layer_specs(run: str, out_frac: int | None = 4):
    """[(name, W, b, (k, i, f))] for every QDense of a trained student, in graph order.

    ``W``/``b`` are the exact QAT-quantised kernel/bias; ``(k, i, f)`` are per-input arrays of the
    fixed-point format that feeds the layer in the QAT graph (see module docstring).
    """
    import da4ml.trace as tr

    cfg, met, m, preset = _student(run)
    fm = layer_input_formats(cfg, preset)
    P = cfg.patch_size

    def wb(name):
        lyr = m.get_layer(name)
        W = np.asarray(lyr.kernel_quantizer_internal(lyr.kernel), dtype=np.float64)
        b = None
        if getattr(lyr, "use_bias", False) and lyr.bias is not None:
            b = np.asarray(lyr.bias_quantizer_internal(lyr.bias), dtype=np.float64)
        return W, b

    def full(n, t):
        return tuple(np.full(n, v) for v in t)

    def out_kifs_parallel(name, kif):
        W, b = wb(name)
        k, i, f = (np.asarray(v) for v in kif)
        x = tr.FixedVariableArray.from_kif(k.astype(bool), i.astype(int), f.astype(int))
        y = x @ W
        if b is not None:
            y = y + b
        if out_frac is not None:
            y = y.quantize(f=np.full(W.shape[1], out_frac), round_mode="TRN")
        return tr.comb_trace(x, y).out_kifs

    def kif_for(name, n):
        if name in ("embed_hidden", "embed_residual"):        # [scaled values | missing flags]
            A = fm["A"]
            return (np.r_[np.full(P, A[0]), np.zeros(n - P)], np.r_[np.full(P, A[1]), np.ones(n - P)],
                    np.r_[np.full(P, A[2]), np.zeros(n - P)])
        if name in ("embed_output", "head_output"):
            return full(n, fm["RELU"])
        if name.endswith("_wO"):
            return full(n, fm["WO"])
        if name.endswith("_w3"):                              # linear half of w12 x relu(other half)
            blk = name[: -len("_w3")]
            n12 = m.get_layer(f"{blk}_w12").kernel.shape[0]
            k_, i_, f_ = (np.asarray(v)[:n] for v in out_kifs_parallel(f"{blk}_w12", full(n12, fm["A"])))
            R = fm["RELU"]
            return k_, i_ + R[1], f_ + R[2]
        return full(n, fm["A"])                               # RMSNorm outputs

    specs = []
    for lyr in m.layers:
        if type(lyr).__name__ != "QDense":
            continue
        W, b = wb(lyr.name)
        kif = tuple(np.asarray(v).astype(int) for v in kif_for(lyr.name, W.shape[0]))
        specs.append((lyr.name, W, b, kif))
    return cfg, met, specs


def _random_inputs(kif, n_vec, seed=0):
    k, i, f = (np.asarray(v, dtype=float) for v in kif)
    step = 2.0 ** -f
    lo, hi = np.where(k > 0, -(2.0 ** i), 0.0), 2.0 ** i - step
    rng = np.random.default_rng(seed)
    x = np.round(rng.uniform(lo, hi, (n_vec, len(k))) / step) * step
    x[0], x[1] = lo, hi                                       # include both range corners
    return x


def export(run: str, out: str, latency_cutoff: float = 5.0, out_frac: int | None = 4) -> list[dict]:
    """Write one fully parallel, pipelined Verilog block per dense layer; returns the export records."""
    if _needs_isolation():
        return _isolated_export(["--run", run, "--out", out, "--latency-cutoff", str(latency_cutoff)]
                                + (["--out-frac", str(out_frac)] if out_frac is not None else []), out)
    import da4ml.codegen as cg
    import da4ml.trace as tr

    cfg, met, specs = layer_specs(run, out_frac)
    rows = []
    for name, W, b, kif in specs:
        k, i, f = kif
        x = tr.FixedVariableArray.from_kif(k.astype(bool), i, f)
        y = x @ W
        if b is not None:
            y = y + b
        if out_frac is not None:
            y = y.quantize(f=np.full(W.shape[1], out_frac), round_mode="TRN")
        comb = tr.comb_trace(x, y)
        xv = _random_inputs(kif, 500)
        ref = xv @ W + (0.0 if b is None else b)
        if out_frac is not None:
            ref = np.floor(ref * 2 ** out_frac) / 2 ** out_frac
        err = float(np.abs(comb.predict(xv) - ref).max())
        if err != 0.0:
            raise AssertionError(f"{name}: da4ml block is not bit-exact (max err {err})")
        d = os.path.join(out, name)
        pipe = tr.to_pipeline(comb, latency_cutoff=latency_cutoff, verbose=False)
        cg.RTLModel(pipe, name, d, latency_cutoff=latency_cutoff, print_latency=False).write()
        meta = json.load(open(os.path.join(d, "metadata.json")))
        rows.append(dict(layer=name, top=name, mode="parallel", digit_bits=None, cycles_per_token=1,
                         shape=list(W.shape), nonzero=int(np.count_nonzero(W)), bias=b is not None,
                         in_bits=int((k + i + f).sum()), out_bits=int(np.array(comb.out_kifs).sum()),
                         da4ml_cost=float(comb.cost), stages=int(meta["latency"]), latency=int(meta["latency"]) + 1,
                         reg_bits=int(meta["reg_bits"]), max_err=err))
    os.makedirs(out, exist_ok=True)
    json.dump(dict(run=run, student=cfg.name, dataset=met["dataset"], mode="parallel",
                   latency_cutoff=latency_cutoff, out_frac=out_frac, layers=rows),
              open(os.path.join(out, "export.json"), "w"), indent=1)
    return rows


# ----------------------------------------------------------------------------- digit-serial
def _frac_bits(v, max_f: int = 40) -> int:
    v = np.atleast_1d(np.asarray(v, dtype=np.float64))
    for f in range(max_f + 1):
        if np.allclose(v * 2.0 ** f, np.round(v * 2.0 ** f), rtol=0, atol=0):
            return f
    raise ValueError("value needs more than %d fractional bits" % max_f)


def _bits_signed(lo: float, hi: float) -> int:
    """Two's complement width holding every integer in [lo, hi]."""
    m = max(abs(int(np.floor(lo))), abs(int(np.ceil(hi))) + 1, 1)
    return int(np.ceil(np.log2(m))) + 1


def serial_plan(W, b, kif, d: int, out_frac: int | None = 4):
    """Arithmetic plan of the digit-serial layer (all integers, documented in ``serial_verilog``).

    x_j = X_j 2^-f_j with X_j a B_j-bit integer.  Signed inputs are made unsigned by flipping the
    MSB, U_j = X_j + 2^(B_j-1); every U_j is cut into R = ceil(max B / d) unsigned d-bit digits
    (MSB first).  Per cycle one constant-weight block computes P_t = sum_j W'_jo D_jt with
    W' = W 2^-f_j and the accumulator does acc <- acc * 2^d + P_t.  After R cycles
    y_o = acc_o - sum_j W'_jo 2^(B_j-1) [signed j] + b_o, truncated to ``out_frac`` bits.
    """
    k, i, f = (np.asarray(v).astype(int) for v in kif)
    Bj = k + i + f
    R = int(np.ceil(Bj.max() / d))
    Wp = W * 2.0 ** (-f[:, None])
    C = -(Wp * (k[:, None] * 2.0 ** (Bj[:, None] - 1))).sum(0)
    if b is not None:
        C = C + b
    return dict(Bj=Bj, R=R, Wp=Wp, C=C, k=k, f=f, d=d, out_frac=out_frac)


def serial_verilog(name: str, plan: dict, core_kifs, core_widths, core_latency: int, core_has_clk: bool,
                   keep=None) -> str:
    """Verilog of ``<name>_serial`` (start/valid streaming) and ``<name>_pr`` (back-to-back tokens).

    ``keep`` are the original input indices that survive into the da4ml core (rows of ``W`` that are
    not entirely zero).  da4ml drops all-zero input variables from its ports and repacks the rest, so
    the digit bus fed to the core must contain *only* the kept inputs, in ascending original order.
    The external ``model_inp`` still exposes all ``n`` inputs (unchanged interface); dead inputs are
    simply never sampled.
    """
    Bj, R, d, C, k = plan["Bj"], plan["R"], plan["d"], plan["C"], plan["k"]
    n, n_out = len(Bj), len(C)
    keep = np.arange(n) if keep is None else np.asarray(keep).astype(int)
    nk = len(keep)
    OF = plan["out_frac"] if plan["out_frac"] is not None else 0
    ko, io_, fo = (np.asarray(v).astype(int) for v in core_kifs)
    F = int(max([OF, _frac_bits(C)] + [int(x) for x, w in zip(fo, core_widths) if w > 0]))
    RD = R * d
    # accumulator range: |acc| <= sum_j |W'_jo| (2^Bj - 1) (unsigned digits) in units 2^-F, plus C
    absw = np.abs(plan["Wp"])
    acc_max = (absw * (2.0 ** Bj[:, None] - 1)).sum(0) * 2.0 ** F
    ACC = int(max(_bits_signed(-a - abs(c) * 2.0 ** F, a + abs(c) * 2.0 ** F) for a, c in zip(acc_max, C)))
    out_lo = (np.floor((-(absw * (2.0 ** Bj[:, None])).sum(0) - np.abs(C)) * 2.0 ** OF)).min()
    out_hi = (np.ceil(((absw * (2.0 ** Bj[:, None])).sum(0) + np.abs(C)) * 2.0 ** OF)).max()
    OW = _bits_signed(out_lo, out_hi)
    IN_W = int(Bj.sum())
    offs = np.r_[0, np.cumsum(Bj)[:-1]].astype(int)
    coffs = np.r_[0, np.cumsum(core_widths)[:-1]].astype(int)
    CW = int(np.sum(core_widths))
    L = []
    a = L.append
    a(f"// digit-serial constant-weight layer {name}: {nk}/{n} inputs ({n - nk} pruned), {n_out} outputs, "
      f"{d}-bit digits, R={R} cycles/token")
    a(f"module {name}_serial (input clk, input start, input [{IN_W - 1}:0] model_inp,")
    a(f"    output reg [{n_out * OW - 1}:0] model_out, output reg valid);")
    a(f"    reg [{nk * RD - 1}:0] sr;")
    a(f"    reg [{max(1, int(np.ceil(np.log2(R + 1)))) - 1}:0] cnt;")
    a("    reg busy;")
    a(f"    wire [{nk * d - 1}:0] digits;")
    for kk, jj in enumerate(keep):
        a(f"    assign digits[{kk * d + d - 1}:{kk * d}] = sr[{kk * RD + RD - 1}:{kk * RD + RD - d}];")
    a("    always @(posedge clk) begin")
    a("        if (start) begin")
    for kk, jj in enumerate(keep):
        jj = int(jj)
        src = f"model_inp[{offs[jj] + Bj[jj] - 1}:{offs[jj]}]"
        if k[jj]:
            msb = offs[jj] + Bj[jj] - 1
            src = (f"{{~model_inp[{msb}], model_inp[{msb - 1}:{offs[jj]}]}}" if Bj[jj] > 1 else f"~model_inp[{msb}]")
        pad = RD - int(Bj[jj])
        val = f"{{{pad}'d0, {src}}}" if pad > 0 else src
        a(f"            sr[{kk * RD + RD - 1}:{kk * RD}] <= {val};")
    a("            busy <= 1'b1; cnt <= 0;")
    a("        end else if (busy) begin")
    for kk in range(nk):
        a(f"            sr[{kk * RD + RD - 1}:{kk * RD}] <= {{sr[{kk * RD + RD - d - 1}:{kk * RD}], {d}'d0}};" if RD > d
          else f"            sr[{kk * RD + RD - 1}:{kk * RD}] <= {d}'d0;")
    a(f"            cnt <= cnt + 1'b1; if (cnt == {R - 1}) busy <= 1'b0;")
    a("        end")
    a("    end")
    # tags travel alongside the core pipeline
    a(f"    wire t_v = busy; wire t_first = busy && (cnt == 0); wire t_last = busy && (cnt == {R - 1});")
    a(f"    wire [{CW - 1}:0] p;")
    if core_has_clk:
        a(f"    {name}_core core (.clk(clk), .model_inp(digits), .model_out(p));")
    else:
        a(f"    {name}_core core (.model_inp(digits), .model_out(p));")
    if core_latency > 0:
        a(f"    reg [{core_latency - 1}:0] v_sr, f_sr, l_sr;")
        a("    always @(posedge clk) begin")
        if core_latency > 1:
            a(f"        v_sr <= {{v_sr[{core_latency - 2}:0], t_v}}; f_sr <= {{f_sr[{core_latency - 2}:0], t_first}}; "
              f"l_sr <= {{l_sr[{core_latency - 2}:0], t_last}};")
        else:
            a("        v_sr <= t_v; f_sr <= t_first; l_sr <= t_last;")
        a("    end")
        a(f"    wire p_v = v_sr[{core_latency - 1}], p_first = f_sr[{core_latency - 1}], p_last = l_sr[{core_latency - 1}];")
    else:
        a("    wire p_v = t_v, p_first = t_first, p_last = t_last;")
    for o in range(n_out):
        w = int(core_widths[o])
        if w == 0:
            a(f"    wire signed [{ACC - 1}:0] pe{o} = {ACC}'sd0;")
            continue
        fld = f"p[{coffs[o] + w - 1}:{coffs[o]}]"
        ext = f"$signed({fld})" if ko[o] else f"$signed({{1'b0, {fld}}})"
        a(f"    wire signed [{ACC - 1}:0] pe{o}_r = {ext};")
        sh = F - int(fo[o])
        a(f"    wire signed [{ACC - 1}:0] pe{o} = pe{o}_r <<< {sh};" if sh else f"    wire signed [{ACC - 1}:0] pe{o} = pe{o}_r;")
    a("    reg fin;")
    a(f"    reg signed [{ACC - 1}:0] " + ", ".join(f"acc{o}" for o in range(n_out)) + ";")
    a("    always @(posedge clk) begin")
    a("        if (p_v) begin")
    for o in range(n_out):
        a(f"            acc{o} <= (p_first ? {ACC}'sd0 : (acc{o} <<< {d})) + pe{o};")
    a("        end")
    a("        fin <= p_v && p_last;")
    a("        valid <= fin;")
    a("        if (fin) begin")
    for o in range(n_out):
        ci = int(round(C[o] * 2.0 ** F))
        cst = f"{ACC}'sd{ci}" if ci >= 0 else f"-{ACC}'sd{-ci}"
        a(f"            model_out[{o * OW + OW - 1}:{o * OW}] <= (acc{o} + {cst}) >>> {F - OF};")
    a("        end")
    a("    end")
    a("endmodule")
    a("")
    a(f"// place & route top: a new token every R={R} cycles (start strobe from a free-running counter)")
    a(f"module {name}_pr (input clk, input [{IN_W - 1}:0] model_inp, output [{n_out * OW - 1}:0] model_out);")
    a(f"    reg [{max(1, int(np.ceil(np.log2(R + 1)))) - 1}:0] c = 0;")
    a(f"    always @(posedge clk) c <= (c == {R - 1}) ? 0 : c + 1'b1;")
    a("    wire valid;")
    a(f"    {name}_serial s (.clk(clk), .start(c == 0), .model_inp(model_inp), .model_out(model_out), .valid(valid));")
    a("endmodule")
    info = dict(R=R, F=F, ACC=ACC, OW=OW, IN_W=IN_W, OUT_W=n_out * OW, offsets=offs.tolist())
    return "\n".join(L) + "\n", info


def export_serial(run: str, out: str, digit_bits: int = 2, latency_cutoff: float = 5.0, out_frac: int | None = 4,
                  verify: bool = True, n_vec: int = 200) -> list[dict]:
    """Digit-serial (time-multiplexed) variant of every dense layer, Verilator-verified."""
    if _needs_isolation():
        return _isolated_export(["--run", run, "--out", out, "--digit-bits", str(digit_bits),
                                 "--latency-cutoff", str(latency_cutoff)]
                                + (["--out-frac", str(out_frac)] if out_frac is not None else [])
                                + ([] if verify else ["--no-verify"]), out)
    import da4ml.codegen as cg
    import da4ml.trace as tr

    cfg, met, specs = layer_specs(run, out_frac)
    rows = []
    for name, W, b, kif in specs:
        plan = serial_plan(W, b, kif, digit_bits, out_frac)
        n = W.shape[0]
        # da4ml drops all-zero input rows from its ports; build the core over only the kept inputs so
        # the core's packed input order matches the digit bus (else every input after a pruned row shifts).
        keep = np.where(np.any(plan["Wp"] != 0, axis=1))[0]
        Wpk = plan["Wp"][keep]
        x = tr.FixedVariableArray.from_kif(np.zeros(len(keep), bool), np.full(len(keep), digit_bits), np.zeros(len(keep), int))
        comb = tr.comb_trace(x, x @ Wpk, keep_dead_inputs=True)
        ck = np.array(comb.out_kifs)
        widths = ck.sum(0).astype(int)
        d = os.path.join(out, name)
        if latency_cutoff and latency_cutoff > 0:
            core = tr.to_pipeline(comb, latency_cutoff=latency_cutoff, verbose=False)
        else:
            core = comb
        cg.RTLModel(core, f"{name}_core", d, latency_cutoff=latency_cutoff if latency_cutoff else -1,
                    print_latency=False).write()
        meta = json.load(open(os.path.join(d, "metadata.json")))
        core_v = open(os.path.join(d, "src", f"{name}_core.v")).read()
        has_clk = "input clk" in core_v.split(");")[0]
        lat = int(meta["latency"]) + 1 if has_clk else 0
        v, info = serial_verilog(name, plan, ck, widths, lat, has_clk, keep=keep)
        open(os.path.join(d, "src", f"{name}_serial.v"), "w").write(v)
        row = dict(layer=name, top=f"{name}_pr", mode="serial", digit_bits=digit_bits, cycles_per_token=plan["R"],
                   latency=plan["R"] + lat + 2, shape=list(W.shape), nonzero=int(np.count_nonzero(W)),
                   pruned_inputs=int(n - len(keep)),
                   bias=b is not None, in_bits=info["IN_W"], out_bits=info["OUT_W"], acc_bits=info["ACC"],
                   da4ml_cost=float(comb.cost), core_latency=lat, reg_bits=int(meta.get("reg_bits", 0)))
        if verify:
            row["verilator_errors"] = verilate_check(d, name, plan, kif, W, b, info, out_frac, n_vec, keep=keep)
            if row["verilator_errors"] != 0:
                raise AssertionError(f"{name}: digit-serial RTL mismatches numpy ({row['verilator_errors']} vectors)")
        rows.append(row)
    os.makedirs(out, exist_ok=True)
    json.dump(dict(run=run, student=cfg.name, dataset=met["dataset"], mode="serial", digit_bits=digit_bits,
                   latency_cutoff=latency_cutoff, out_frac=out_frac, layers=rows),
              open(os.path.join(out, "export.json"), "w"), indent=1)
    return rows


def _to_hex(fields, widths) -> str:
    v = 0
    off = 0
    for x, w in zip(fields, widths):
        v |= (int(x) & ((1 << int(w)) - 1)) << off
        off += int(w)
    return format(v, "x")


def verilate_check(d: str, name: str, plan, kif, W, b, info, out_frac, n_vec: int = 200,
                   verilator: str | None = None, keep=None) -> int:
    """Stream random on-grid vectors through ``<name>_serial`` in Verilator; returns mismatches.

    ``model_inp`` still carries all inputs (``keep`` is handled inside the generated RTL), so the
    stimulus is packed over every input ``j`` using its full ``Bj`` width.
    """
    verilator = verilator or os.environ.get("VERILATOR", "verilator")
    k, i, f = (np.asarray(v).astype(int) for v in kif)
    xv = _random_inputs(kif, n_vec, seed=1)
    ref = xv @ W + (0.0 if b is None else b)
    OF = out_frac if out_frac is not None else 0
    yint = np.floor(np.round(ref * 2.0 ** OF, 6)).astype(np.int64)
    xint = np.round(xv * 2.0 ** f).astype(np.int64)
    sim = os.path.join(d, "sim_serial")
    os.makedirs(sim, exist_ok=True)
    with open(os.path.join(sim, "x.mem"), "w") as fx, open(os.path.join(sim, "y.mem"), "w") as fy:
        for xr, yr in zip(xint, yint):
            fx.write(_to_hex(xr, plan["Bj"]) + "\n")
            fy.write(_to_hex(yr, [info["OW"]] * len(yr)) + "\n")
    wait = plan["R"] + 40
    tb = f"""`timescale 1ns/1ps
module tb;
    reg clk = 0; always #5 clk = ~clk;
    reg [{info['IN_W'] - 1}:0] xm [0:{n_vec - 1}];
    reg [{info['OUT_W'] - 1}:0] ym [0:{n_vec - 1}];
    reg [{info['IN_W'] - 1}:0] x; reg start = 0; wire [{info['OUT_W'] - 1}:0] y; wire valid;
    integer t, c, err;
    {name}_serial dut (.clk(clk), .start(start), .model_inp(x), .model_out(y), .valid(valid));
    initial begin
        $readmemh("{os.path.abspath(sim)}/x.mem", xm); $readmemh("{os.path.abspath(sim)}/y.mem", ym); err = 0;
        for (t = 0; t < {n_vec}; t = t + 1) begin
            @(negedge clk); x = xm[t]; start = 1; @(negedge clk); start = 0;
            c = 0; while (!valid && c < {wait}) begin @(posedge clk); #1; c = c + 1; end
            if (!valid || y !== ym[t]) begin err = err + 1; if (err < 4) $display("mismatch %0d valid=%0d", t, valid); end
        end
        $display("ERRORS=%0d", err); $finish;
    end
endmodule
"""
    open(os.path.join(sim, "tb.v"), "w").write(tb)
    srcs = [os.path.join(d, "src", f) for f in os.listdir(os.path.join(d, "src")) if f.endswith(".v")
            and not f.endswith("_wrapper.v")] + glob.glob(os.path.join(d, "src", "static", "*.v"))
    r = subprocess.run([verilator, "--binary", "--timing", "-Wno-fatal", "-Wno-lint", "-Wno-style", "-O1",
                        "--top-module", "tb", os.path.join(sim, "tb.v")] + srcs + ["-Mdir", os.path.join(sim, "obj")],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"verilator build failed for {name}:\n{r.stderr[-2000:]}")
    r = subprocess.run([os.path.abspath(os.path.join(sim, "obj", "Vtb"))], capture_output=True, text=True, cwd=sim)
    import re
    m = re.search(r"ERRORS=(\d+)", r.stdout)
    if not m:
        raise RuntimeError(f"simulation of {name} gave no result:\n{r.stdout[-1000:]}{r.stderr[-1000:]}")
    return int(m.group(1))


# ----------------------------------------------------------------------------- place & route
def _harness(kolibri_src: str | None):
    if kolibri_src:
        sys.path.insert(0, os.path.expanduser(kolibri_src))
    from kolibri_fpga.targets import harness
    return harness


def pnr_layer(out: str, layer: str, stages: int, seed: int = 1, timeout: int = 1800, kolibri_src=None,
              top: str | None = None) -> dict:
    h = _harness(kolibri_src)
    src = os.path.join(out, layer, "src")
    srcs = [f for f in glob.glob(src + "/*.v") + glob.glob(src + "/static/*.v") if not f.endswith("_wrapper.v")]
    wd = os.path.join(out, layer, f"pr_s{seed}")
    t = time.time()
    # da4ml registers the input and the output around ``stages`` pipeline stages
    r = h.run(srcs, top or layer, "model_inp", "model_out", "gatemate", wd, clk_port="clk", cycles=stages + 1,
              timeout=timeout, seed=seed)
    if "cells" in r:
        r["cpe_estimate"] = h.gatemate_cpe_estimate(r["cells"])
    r.update(wall_s=round(time.time() - t, 1), seed=seed, layer=layer)
    json.dump(r, open(os.path.join(wd, "result.json"), "w"), indent=1)
    return r


def pnr(out: str, jobs: int = 4, seeds=(1, 2), timeout: int = 1800, kolibri_src=None, layers=None) -> list[dict]:
    """P&R every exported layer; a layer that does not route with the first seed tries the next."""
    exp = json.load(open(os.path.join(out, "export.json")))["layers"]
    todo = [r for r in exp if not layers or r["layer"] in layers]

    def job(r):
        res = None
        for s in seeds:
            res = pnr_layer(out, r["layer"], r.get("latency", r.get("stages", 0) + 1) - 1, s, timeout, kolibri_src,
                            top=r.get("top"))
            if res.get("routed"):
                break
        return res

    with ThreadPoolExecutor(jobs) as ex:
        results = list(ex.map(job, todo))
    for r in results:
        print(r["layer"], "routed" if r.get("routed") else "FAILED", r.get("fmax_mhz"), r.get("cpe_estimate"), flush=True)
    return results


def collect(out: str) -> "pd.DataFrame":
    """Best routed seed per layer -> ``<out>/pnr_layers.csv`` (parallel or digit-serial export)."""
    import pandas as pd
    exp = json.load(open(os.path.join(out, "export.json")))
    rows = []
    for e in exp["layers"]:
        best = None
        for f in glob.glob(os.path.join(out, e["layer"], "pr*", "result.json")):
            r = json.load(open(f))
            r.setdefault("seed", int(Path(f).parent.name.split("_s")[-1]) if "_s" in Path(f).parent.name else None)
            if r.get("routed") and r.get("fmax_mhz") and (best is None or r["fmax_mhz"] > best["fmax_mhz"]):
                best = r
        cyc = e.get("cycles_per_token", 1)
        row = dict(layer=e["layer"], mode=e.get("mode", "parallel"), digit_bits=e.get("digit_bits"),
                   shape="x".join(map(str, e["shape"])), nonzero=e["nonzero"], da4ml_cost=e["da4ml_cost"],
                   in_bits=e["in_bits"], out_bits=e["out_bits"], acc_bits=e.get("acc_bits"),
                   cycles_per_token=cyc, latency_cycles=e.get("latency", e.get("stages", 0) + 1),
                   routed=best is not None)
        if best:
            row.update(fmax_mhz=best["fmax_mhz"], cpe=best["cpe_estimate"], cpe_lt_half=best["util"]["CPE_LT"][0],
                       cpe_ff=best["util"]["CPE_FF"][0], addf=best["cells"].get("CC_ADDF", 0), seed=best.get("seed"),
                       mtokens_per_s=best["fmax_mhz"] / cyc)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "pnr_layers.csv"), index=False)
    return df


def deploy(out: str) -> dict:
    """Deployment summary from ``pnr_layers.csv`` (no toolchain needed).

    Layers run *sequentially* in a forward pass, but da4ml bakes the constant weights into
    the wiring, so a single bitstream must hold every layer at once:

    * ``sum_cpe``   - all layers resident in one image (the true single-bitstream area)
    * ``peak_cpe``  - largest layer; area needed if the chip is reconfigured per layer
                      (one bitstream per layer + external config sequencing)
    * ``throughput``- slowest layer's Mtok/s (chained pipeline)
    """
    import csv

    with open(os.path.join(out, "pnr_layers.csv")) as fh:
        all_rows = list(csv.DictReader(fh))
    rows = [r for r in all_rows if r.get("routed") == "True"]
    if not rows:
        raise ValueError(f"{out}/pnr_layers.csv has no routed layers")
    missing = [r["layer"] for r in all_rows if r.get("routed") != "True"]
    chip = 20480
    sum_cpe = sum(float(r["cpe"]) for r in rows)
    peak = max(rows, key=lambda r: float(r["cpe"]))
    tput = min(rows, key=lambda r: float(r["mtokens_per_s"]))
    fmax = [float(r["fmax_mhz"]) for r in rows]
    summ = dict(
        layers=len(rows),
        sum_cpe=sum_cpe,
        sum_frac=sum_cpe / chip,
        peak_layer=peak["layer"],
        peak_cpe=float(peak["cpe"]),
        peak_frac=float(peak["cpe"]) / chip,
        fmax_min_mhz=min(fmax),
        fmax_max_mhz=max(fmax),
        throughput_mtok_s=float(tput["mtokens_per_s"]),
        throughput_layer=tput["layer"],
        layers_total=len(all_rows),
        not_routed=missing,
    )
    if missing:
        print(f"WARNING: {len(missing)} of {len(all_rows)} layers not routed ({', '.join(missing)}); "
              "the numbers below cover the routed layers only")
    print(f"single bitstream (all layers resident): {sum_cpe:8.0f} CPE = {summ['sum_frac']:.2f}x chip "
          f"{('FITS' if sum_cpe <= chip else 'DOES NOT FIT') if not missing else 'INCOMPLETE'}")
    print(f"reconfigure per layer (peak only):      {summ['peak_cpe']:8.0f} CPE = {summ['peak_frac']:.2f}x chip "
          f"({'FITS' if summ['peak_cpe'] <= chip else 'DOES NOT FIT'}) [{summ['peak_layer']}]")
    print(f"throughput (chained, slowest layer):    {summ['throughput_mtok_s']:.2f} Mtok/s [{summ['throughput_layer']}]")
    print(f"fmax range:                             {summ['fmax_min_mhz']:.1f} - {summ['fmax_max_mhz']:.1f} MHz")
    json.dump(summ, open(os.path.join(out, "deploy.json"), "w"), indent=1)
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--run", required=True); e.add_argument("--out", required=True)
    e.add_argument("--latency-cutoff", type=float, default=5.0); e.add_argument("--out-frac", type=int, default=4)
    e.add_argument("--digit-bits", type=int, default=None,
                   help="digit-serial (time-multiplexed) mode with d-bit input digits; default fully parallel")
    e.add_argument("--no-verify", action="store_true", help="skip the Verilator check of the serial RTL")
    p = sub.add_parser("pnr"); p.add_argument("--out", required=True); p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3]); p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--kolibri-src", default=os.environ.get("KOLIBRI_FPGA_SRC")); p.add_argument("--layers", nargs="*")
    c = sub.add_parser("collect"); c.add_argument("--out", required=True)
    d = sub.add_parser("deploy"); d.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "export":
        if a.digit_bits:
            rows = export_serial(a.run, a.out, a.digit_bits, a.latency_cutoff, a.out_frac, verify=not a.no_verify)
        else:
            rows = export(a.run, a.out, a.latency_cutoff, a.out_frac)
        if not _needs_isolation():          # otherwise the child process already printed the table
            for r in rows:
                print(r["layer"], r["shape"], r["mode"], "R", r["cycles_per_token"], "cost", r["da4ml_cost"],
                      "verilator errors", r.get("verilator_errors", "-"))
    elif a.cmd == "pnr":
        pnr(a.out, a.jobs, tuple(a.seeds), a.timeout, a.kolibri_src, a.layers)
    elif a.cmd == "deploy":
        deploy(a.out)
    else:
        print(collect(a.out).to_string(index=False))


if __name__ == "__main__":
    main()
