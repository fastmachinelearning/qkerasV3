"""Verilog generators for the non-dense blocks (phase P1), Verilator bit-exact vs P0 refs.

Each generator emits a self-contained combinational module with its LUTs baked in as
``initial``-initialised ROMs, so there are no external memory files to keep in sync. The
integer datapaths are transcribed one-to-one from the P0 references in ``fixedpoint``:
``rmsnorm_ref``, ``softmax_ref``, ``rope_ref`` and ``quantiles_hw_ref``. Because all of
them round with ``floor(x+0.5)`` on integer operands (or clip a pre-rounded QOUT value),
the Verilator check must show **zero** mismatches (not "<= 1 LSB").

    python -m timestral_kd.hw.blocks   # build + Verilator-check rmsnorm, softmax, rope, quantile_head
"""

from __future__ import annotations

import os
import re
import subprocess

import numpy as np

from . import fixedpoint as fp

W = dict(a=7, g=5, score=7, prob=6, exp=7, inv=7)


# ----------------------------------------------------------------------------- helpers
def _rom(name: str, data: np.ndarray, width: int) -> str:
    lines = [f"  reg [{width - 1}:0] {name} [0:{len(data) - 1}];", "  initial begin"]
    for i, v in enumerate(data):
        lines.append(f"    {name}[{i}] = {width}'h{int(v) & ((1 << width) - 1):x};")
    lines.append("  end")
    return "\n".join(lines)


def _pack(codes, widths) -> int:
    v = off = 0
    for c, w in zip(codes, widths):
        v |= (int(c) & ((1 << w) - 1)) << off
        off += w
    return v


# ----------------------------------------------------------------------------- RMSNorm
def rmsnorm_verilog(luts: dict, D: int = 16) -> str:
    fa, fms, frs, fg = fp.fmt(fp.B3["a"]), fp.fmt(fp.B3["ms"]), fp.fmt(fp.B3["rsqrt"]), fp.fmt(fp.B3["gain"])
    assert fms["f"] - 2 * fa["f"] == -2 and D == 16, "shift/widths below assume B3, D=16"
    sh = 2 * fa["f"] - fms["f"] + int(np.log2(D))   # = 6: mean = round(sum(x^2)*2^(fms-2fa)/D)
    assert sh == 6 and frs["f"] + fg["f"] == 6
    xw, gw = W["a"], W["g"]
    L = []
    a = L.append
    a(f"module rmsnorm (input [{D * xw - 1}:0] x, input [{D * gw - 1}:0] g, output [{D * xw - 1}:0] y);")
    a(_rom("rsqrt_rom", luts["RSQRT"], 9))
    for i in range(D):
        a(f"  wire signed [{xw - 1}:0] x{i} = x[{i * xw + xw - 1}:{i * xw}];")
        a(f"  wire signed [{gw - 1}:0] g{i} = g[{i * gw + gw - 1}:{i * gw}];")
    for i in range(D):
        a(f"  wire [13:0] q{i} = x{i} * x{i};")
    a("  wire [16:0] ss = " + " + ".join(f"q{i}" for i in range(D)) + ";")
    a(f"  wire [17:0] ms_raw = (ss + 18'd{1 << (sh - 1)}) >> {sh};")
    a("  wire [8:0] MS = (ms_raw > 18'd511) ? 9'd511 : ms_raw[8:0];")
    a("  wire [8:0] R = rsqrt_rom[MS];")
    for i in range(D):
        a(f"  wire signed [11:0] pg{i} = x{i} * g{i};")
        a(f"  wire signed [22:0] nm{i} = pg{i} * $signed({{1'b0, R}});")
        a(f"  wire signed [22:0] cr{i} = (nm{i} + 23'sd{1 << (frs['f'] + fg['f'] - 1)}) >>> {frs['f'] + fg['f']};")
        a(f"  wire signed [{xw - 1}:0] y{i} = (cr{i} > 23'sd{fa['hi']}) ? {xw}'sd{fa['hi']} : "
          f"(cr{i} < -23'sd{-fa['lo']}) ? -{xw}'sd{-fa['lo']} : cr{i}[{xw - 1}:0];")
        a(f"  assign y[{i * xw + xw - 1}:{i * xw}] = y{i};")
    a("endmodule")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- softmax
def softmax_verilog(luts: dict, S: int = 20) -> str:
    fs, fe, fi, fp_ = fp.fmt(fp.B3["score"]), fp.fmt(fp.B3["exp"]), fp.fmt(fp.B3["inv"]), fp.fmt(fp.B3["prob"])
    sw, ew = W["score"], W["exp"]
    lo = fs["lo"]                                   # -64
    inv_n = int(S * fe["hi"] + 1)                    # sumE index range
    sh = fe["f"] + fi["f"] - fp_["f"]               # = 7
    L = []
    a = L.append
    a(f"module softmax (input [{S * sw - 1}:0] s, input [{S - 1}:0] attend, output [{S * W['prob'] - 1}:0] p);")
    a(_rom("exp_rom", luts["EXP"][: (fs["hi"] - lo + 1)], ew))
    a(_rom("inv_rom", luts["INV"][:inv_n], W["inv"]))
    for i in range(S):
        a(f"  wire signed [{sw - 1}:0] s{i} = s[{i * sw + sw - 1}:{i * sw}];")
    # max over attend
    a(f"  reg signed [{sw - 1}:0] maxv;")
    a("  always @* begin")
    a(f"    maxv = -{sw}'sd{-lo};")
    for i in range(S):
        a(f"    if (attend[{i}] && s{i} > maxv) maxv = s{i};")
    a("  end")
    # exp + sum
    a(f"  reg [{ew - 1}:0] " + ", ".join(f"e{i}" for i in range(S)) + ";")
    a(f"  reg [{int(np.ceil(np.log2(inv_n + 1)))}:0] sumE;")
    a("  always @* begin")
    a("    sumE = 0;")
    for i in range(S):
        a(f"    if (attend[{i}]) begin")
        a(f"      if ((s{i} - maxv) < -8'sd{-lo}) e{i} = exp_rom[0];")
        a(f"      else e{i} = exp_rom[(s{i} - maxv) + 8'sd{-lo}];")
        a(f"      sumE = sumE + e{i};")
        a("    end else e{i} = 0;".format(i=i))
    a("  end")
    a(f"  wire [{W['inv'] - 1}:0] invv = (sumE == 0) ? 0 : inv_rom[sumE - 1];")
    for i in range(S):
        a(f"  wire [13:0] pr{i} = e{i} * invv + 14'd{1 << (sh - 1)};")
        a(f"  wire [5:0] p{i} = (pr{i} >> {sh}) > 6'd{fp_['hi']} ? 6'd{fp_['hi']} : pr{i}[{sh + 5}:{sh}];")
        a(f"  assign p[{i * W['prob'] + W['prob'] - 1}:{i * W['prob']}] = p{i};")
    a("endmodule")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- RoPE
def rope_verilog(n: int = fp.ROPE_N, hd: int = fp.HEAD_DIM, S: int = 20) -> str:
    """Partial rotate-half, transcribed from fixedpoint.rope_ref.

    dim r and r+hd/2 are mixed for r<n; other dims pass through. cos/sin tables baked in,
    indexed by ``pos``. One rounding (floor(x+0.5) == (x + 2^(sh-1)) >>> sh) per output.
    """
    fa, fc = fp.fmt(fp.B3["a"]), fp.fmt(fp.ROPE_CS)
    xw, cw, sh = fa["bits"], fc["bits"], fc["f"]
    half = hd // 2
    cos_f, sin_f = fp.rope_tables_ref(S)
    cos_c, sin_c = fp.to_code(cos_f, fc), fp.to_code(sin_f, fc)
    pw = xw + cw            # product width (7+14=21)
    aw = pw + 1             # accumulated diff/sum width
    posw = max(1, int(np.ceil(np.log2(S))))
    rot = {j for j in range(n)} | {j + half for j in range(n)}
    L = []
    a = L.append
    a(f"module rope (input [{posw - 1}:0] pos, input [{hd * xw - 1}:0] x, output [{hd * xw - 1}:0] y);")
    for j in range(n):
        a(_rom(f"cos{j}", cos_c[:, j], cw))
        a(_rom(f"sin{j}", sin_c[:, j], cw))
    for i in range(hd):
        a(f"  wire signed [{xw - 1}:0] x{i} = x[{i * xw + xw - 1}:{i * xw}];")
    for j in range(n):
        a(f"  wire signed [{cw - 1}:0] c{j} = cos{j}[pos];")
        a(f"  wire signed [{cw - 1}:0] s{j} = sin{j}[pos];")
        a(f"  wire signed [{pw - 1}:0] pa{j} = x{j} * c{j};")
        a(f"  wire signed [{pw - 1}:0] pb{j} = x{j + half} * s{j};")
        a(f"  wire signed [{aw - 1}:0] d{j} = pa{j} - pb{j};")
        a(f"  wire signed [{pw - 1}:0] pe{j} = x{j + half} * c{j};")
        a(f"  wire signed [{pw - 1}:0] pf{j} = x{j} * s{j};")
        a(f"  wire signed [{aw - 1}:0] g{j} = pe{j} + pf{j};")
        a(f"  wire signed [{aw - 1}:0] rr{j} = (d{j} + {aw}'sd{1 << (sh - 1)}) >>> {sh};")
        a(f"  wire signed [{aw - 1}:0] rs{j} = (g{j} + {aw}'sd{1 << (sh - 1)}) >>> {sh};")
        a(f"  wire signed [{xw - 1}:0] y{j} = (rr{j} > {aw}'sd{fa['hi']}) ? {xw}'sh{fa['hi'] & ((1 << xw) - 1):x} : "
          f"(rr{j} < -{aw}'sd{-fa['lo']}) ? {xw}'sh{fa['lo'] & ((1 << xw) - 1):x} : rr{j}[{xw - 1}:0];")
        a(f"  wire signed [{xw - 1}:0] y{j + half} = (rs{j} > {aw}'sd{fa['hi']}) ? {xw}'sh{fa['hi'] & ((1 << xw) - 1):x} : "
          f"(rs{j} < -{aw}'sd{-fa['lo']}) ? {xw}'sh{fa['lo'] & ((1 << xw) - 1):x} : rs{j}[{xw - 1}:0];")
    for i in range(hd):
        if i in rot:
            continue
        a(f"  wire signed [{xw - 1}:0] y{i} = x{i};")
    for i in range(hd):
        a(f"  assign y[{i * xw + xw - 1}:{i * xw}] = y{i};")
    a("endmodule")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- quantile head
def quantile_head_verilog(knots=(0.1, 0.5, 0.9), width_factor: float = 3.0, G: int = 8) -> str:
    """median ∓ cumsum(width·softplus(raw)), transcribed from fixedpoint.quantiles_hw_ref.

    softplus ROM holds softplus·2^(f+G) (18-bit); width stored as round(w·2^G). Each knot
    increment is (sp·woff + 2^(2G-1)) >> 2G in QOUT units, rounding once. The median is
    clipped first (as in MonotoneQuantiles), then increments accumulate, then final clip.
    """
    fa, fq = fp.fmt(fp.B3["a"]), fp.fmt(fp.QOUT)
    Q = len(knots)
    xw, ow = fa["bits"], fq["bits"]
    splut = fp.softplus_lut(G=G)
    splutw = int(splut.max()).bit_length()
    woff = fp.qround(fp.quantile_increment_widths(knots, width_factor) * 2.0 ** G).astype(int)
    ww = int(woff.max()).bit_length()
    prw = splutw + ww       # full product width (no truncation before the shift)
    m = list(knots).index(0.5)
    hi_pat, lo_pat = fq["hi"] & ((1 << ow) - 1), fq["lo"] & ((1 << ow) - 1)
    accw = 16
    sh = 2 * G
    L = []
    a = L.append
    a(f"module quantile_head (input [{Q * xw - 1}:0] raw, output [{Q * ow - 1}:0] q);")
    a(_rom("splut", splut, splutw))
    for i in range(Q):
        a(f"  wire signed [{xw - 1}:0] r{i} = raw[{i * xw + xw - 1}:{i * xw}];")
        a(f"  wire [6:0] ix{i} = r{i} ^ 7'b1000000;")     # + (-fa.lo)=64 -> ROM index 0..127
    mew = max(ow, xw + (fq["f"] - fa["f"]) + 1)     # wide enough for <<shift and the ow-bit slice
    a(f"  wire signed [{mew - 1}:0] med = r{m} <<< {fq['f'] - fa['f']};")
    a(f"  wire signed [{ow - 1}:0] qm = (med > {mew}'sd{fq['hi']}) ? {ow}'sh{hi_pat:x} : "
      f"(med < -{mew}'sd{-fq['lo']}) ? {ow}'sh{lo_pat:x} : med[{ow - 1}:0];")
    prev = "qm"
    for i in range(m + 1, Q):
        a(f"  wire [{prw - 1}:0] pr{i} = splut[ix{i}] * {ww}'d{woff[i]};")
        a(f"  wire signed [{accw - 1}:0] d{i} = (pr{i} + {prw}'d{1 << (sh - 1)}) >> {sh};")
        a(f"  wire signed [{accw - 1}:0] acc{i} = {prev} + d{i};")
        a(f"  wire signed [{ow - 1}:0] q{i} = (acc{i} > {accw}'sd{fq['hi']}) ? {ow}'sh{hi_pat:x} : "
          f"(acc{i} < -{accw}'sd{-fq['lo']}) ? {ow}'sh{lo_pat:x} : acc{i}[{ow - 1}:0];")
        prev = f"acc{i}"
    prev = "qm"
    for i in range(m - 1, -1, -1):
        a(f"  wire [{prw - 1}:0] pr{i} = splut[ix{i}] * {ww}'d{woff[i]};")
        a(f"  wire signed [{accw - 1}:0] d{i} = (pr{i} + {prw}'d{1 << (sh - 1)}) >> {sh};")
        a(f"  wire signed [{accw - 1}:0] acc{i} = {prev} - d{i};")
        a(f"  wire signed [{ow - 1}:0] q{i} = (acc{i} > {accw}'sd{fq['hi']}) ? {ow}'sh{hi_pat:x} : "
          f"(acc{i} < -{accw}'sd{-fq['lo']}) ? {ow}'sh{lo_pat:x} : acc{i}[{ow - 1}:0];")
        prev = f"acc{i}"
    a(f"  wire signed [{ow - 1}:0] q{m} = qm;")
    for i in range(Q):
        a(f"  assign q[{i * ow + ow - 1}:{i * ow}] = q{i};")
    a("endmodule")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- Verilator
def _to_hex(v: int, width: int) -> str:
    return format(v & ((1 << width) - 1), "x")


def _run_verilator(workdir: str, top: str, tb: str, srcs: list[str]) -> int:
    os.makedirs(workdir, exist_ok=True)
    tb_path = os.path.join(workdir, "tb.v")
    open(tb_path, "w").write(tb)
    verilator = os.environ.get("VERILATOR", "verilator")
    r = subprocess.run([verilator, "--binary", "--timing", "-Wno-fatal", "-Wno-lint", "-Wno-style", "-O1",
                        "--top-module", "tb", tb_path] + srcs + ["-Mdir", os.path.join(workdir, "obj")],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"verilator build failed:\n{r.stderr[-3000:]}")
    r = subprocess.run([os.path.abspath(os.path.join(workdir, "obj", "Vtb"))], capture_output=True, text=True,
                       cwd=workdir)
    m = re.search(r"ERRORS=(\d+)", r.stdout)
    if not m:
        raise RuntimeError(f"no result:\n{r.stdout[-1500:]}{r.stderr[-1500:]}")
    return int(m.group(1))


def check_rmsnorm(workdir: str, n_vec: int = 300, seed: int = 0) -> int:
    luts = fp.build_luts(16)
    src = os.path.join(workdir, "rmsnorm.v")
    os.makedirs(workdir, exist_ok=True)
    open(src, "w").write(rmsnorm_verilog(luts, 16))
    fa, fg = fp.fmt(fp.B3["a"]), fp.fmt(fp.B3["gain"])
    rng = np.random.default_rng(seed)
    X = rng.integers(fa["lo"], fa["hi"] + 1, (n_vec, 16))
    G = rng.integers(fg["lo"], fg["hi"] + 1, (n_vec, 16))
    X[0] = fa["hi"]; X[1] = fa["lo"]; X[2] = 0                     # corners
    with open(os.path.join(workdir, "x.mem"), "w") as fx, open(os.path.join(workdir, "g.mem"), "w") as fg_, \
            open(os.path.join(workdir, "y.mem"), "w") as fy:
        for x, g in zip(X, G):
            y = fp.rmsnorm_ref(x, g, luts)
            fx.write(_to_hex(_pack(x, [7] * 16), 112) + "\n")
            fg_.write(_to_hex(_pack(g, [5] * 16), 80) + "\n")
            fy.write(_to_hex(_pack(y, [7] * 16), 112) + "\n")
    tb = f"""`timescale 1ns/1ps
module tb;
  reg [111:0] xm [0:{n_vec - 1}]; reg [79:0] gm [0:{n_vec - 1}]; reg [111:0] ym [0:{n_vec - 1}];
  reg [111:0] x; reg [79:0] g; wire [111:0] y; integer t, err;
  rmsnorm dut (.x(x), .g(g), .y(y));
  initial begin
    $readmemh("{os.path.abspath(workdir)}/x.mem", xm);
    $readmemh("{os.path.abspath(workdir)}/g.mem", gm);
    $readmemh("{os.path.abspath(workdir)}/y.mem", ym);
    err = 0;
    for (t = 0; t < {n_vec}; t = t + 1) begin
      x = xm[t]; g = gm[t]; #1;
      if (y !== ym[t]) begin err = err + 1; if (err < 4) $display("mismatch %0d y=%h exp=%h", t, y, ym[t]); end
    end
    $display("ERRORS=%0d", err); $finish;
  end
endmodule
"""
    return _run_verilator(workdir, "rmsnorm", tb, [src])


def check_softmax(workdir: str, n_vec: int = 300, seed: int = 1) -> int:
    luts = fp.build_luts(16)
    src = os.path.join(workdir, "softmax.v")
    os.makedirs(workdir, exist_ok=True)
    open(src, "w").write(softmax_verilog(luts, 20))
    fs = fp.fmt(fp.B3["score"])
    rng = np.random.default_rng(seed)
    SC = rng.integers(fs["lo"], fs["hi"] + 1, (n_vec, 20))
    AT = (rng.random((n_vec, 20)) < 0.85).astype(int)
    AT[:, :] = np.maximum(AT, 0)
    AT[np.arange(n_vec), rng.integers(0, 20, n_vec)] = 1       # always >=1 attend
    SC[0] = fs["hi"]; SC[1] = fs["lo"]
    with open(os.path.join(workdir, "s.mem"), "w") as fs_, open(os.path.join(workdir, "at.mem"), "w") as fa_, \
            open(os.path.join(workdir, "p.mem"), "w") as fpp:
        for s, at in zip(SC, AT):
            p = fp.softmax_ref(fp.to_value(s, fs), luts, mask=at.astype(bool))
            fs_.write(_to_hex(_pack(s, [7] * 20), 140) + "\n")
            fa_.write(_to_hex(_pack(at, [1] * 20), 20) + "\n")
            fpp.write(_to_hex(_pack(p, [6] * 20), 120) + "\n")
    tb = f"""`timescale 1ns/1ps
module tb;
  reg [139:0] sm [0:{n_vec - 1}]; reg [19:0] am [0:{n_vec - 1}]; reg [119:0] pm [0:{n_vec - 1}];
  reg [139:0] s; reg [19:0] at; wire [119:0] p; integer t, err;
  softmax dut (.s(s), .attend(at), .p(p));
  initial begin
    $readmemh("{os.path.abspath(workdir)}/s.mem", sm);
    $readmemh("{os.path.abspath(workdir)}/at.mem", am);
    $readmemh("{os.path.abspath(workdir)}/p.mem", pm);
    err = 0;
    for (t = 0; t < {n_vec}; t = t + 1) begin
      s = sm[t]; at = am[t]; #1;
      if (p !== pm[t]) begin err = err + 1; if (err < 4) $display("mismatch %0d p=%h exp=%h", t, p, pm[t]); end
    end
    $display("ERRORS=%0d", err); $finish;
  end
endmodule
"""
    return _run_verilator(workdir, "softmax", tb, [src])


def check_rope(workdir: str, n_vec: int = 300, seed: int = 2, S: int = 20) -> int:
    n, hd = fp.ROPE_N, fp.HEAD_DIM
    src = os.path.join(workdir, "rope.v")
    os.makedirs(workdir, exist_ok=True)
    open(src, "w").write(rope_verilog(n, hd, S))
    fa, fc = fp.fmt(fp.B3["a"]), fp.fmt(fp.ROPE_CS)
    cos_f, sin_f = fp.rope_tables_ref(S)
    cos_c, sin_c = fp.to_code(cos_f, fc), fp.to_code(sin_f, fc)
    posw = max(1, int(np.ceil(np.log2(S))))
    rng = np.random.default_rng(seed)
    P = rng.integers(0, S, n_vec)
    X = rng.integers(fa["lo"], fa["hi"] + 1, (n_vec, hd))
    X[0] = fa["hi"]; X[1] = fa["lo"]; X[2] = 0
    with open(os.path.join(workdir, "p.mem"), "w") as fpp, open(os.path.join(workdir, "x.mem"), "w") as fx, \
            open(os.path.join(workdir, "y.mem"), "w") as fy:
        for p, x in zip(P, X):
            y = fp.rope_ref(x, cos_c[p], sin_c[p])
            fpp.write(_to_hex(int(p), posw) + "\n")
            fx.write(_to_hex(_pack(x, [fa["bits"]] * hd), hd * fa["bits"]) + "\n")
            fy.write(_to_hex(_pack(y, [fa["bits"]] * hd), hd * fa["bits"]) + "\n")
    xw, ow = hd * fa["bits"], hd * fa["bits"]
    tb = f"""`timescale 1ns/1ps
module tb;
  reg [{posw - 1}:0] pm [0:{n_vec - 1}]; reg [{xw - 1}:0] xm [0:{n_vec - 1}]; reg [{ow - 1}:0] ym [0:{n_vec - 1}];
  reg [{posw - 1}:0] pos; reg [{xw - 1}:0] x; wire [{ow - 1}:0] y; integer t, err;
  rope dut (.pos(pos), .x(x), .y(y));
  initial begin
    $readmemh("{os.path.abspath(workdir)}/p.mem", pm);
    $readmemh("{os.path.abspath(workdir)}/x.mem", xm);
    $readmemh("{os.path.abspath(workdir)}/y.mem", ym);
    err = 0;
    for (t = 0; t < {n_vec}; t = t + 1) begin
      pos = pm[t]; x = xm[t]; #1;
      if (y !== ym[t]) begin err = err + 1; if (err < 4) $display("mismatch %0d y=%h exp=%h", t, y, ym[t]); end
    end
    $display("ERRORS=%0d", err); $finish;
  end
endmodule
"""
    return _run_verilator(workdir, "rope", tb, [src])


def check_quantile_head(workdir: str, n_vec: int = 300, seed: int = 3, knots=(0.1, 0.5, 0.9),
                        width_factor: float = 3.0, G: int = 8) -> int:
    Q = len(knots)
    src = os.path.join(workdir, "quantile_head.v")
    os.makedirs(workdir, exist_ok=True)
    open(src, "w").write(quantile_head_verilog(knots, width_factor, G))
    fa, fq = fp.fmt(fp.B3["a"]), fp.fmt(fp.QOUT)
    splut = fp.softplus_lut(G=G)
    rng = np.random.default_rng(seed)
    R = rng.integers(fa["lo"], fa["hi"] + 1, (n_vec, Q))
    R[0] = fa["hi"]; R[1] = fa["lo"]; R[2] = 0
    with open(os.path.join(workdir, "r.mem"), "w") as fr, open(os.path.join(workdir, "q.mem"), "w") as fq_:
        for r in R:
            q = fp.quantiles_hw_ref(r, knots, width_factor, splut, G=G)
            fr.write(_to_hex(_pack(r, [fa["bits"]] * Q), Q * fa["bits"]) + "\n")
            fq_.write(_to_hex(_pack(q, [fq["bits"]] * Q), Q * fq["bits"]) + "\n")
    iw, ow = Q * fa["bits"], Q * fq["bits"]
    tb = f"""`timescale 1ns/1ps
module tb;
  reg [{iw - 1}:0] rm [0:{n_vec - 1}]; reg [{ow - 1}:0] qm [0:{n_vec - 1}];
  reg [{iw - 1}:0] raw; wire [{ow - 1}:0] q; integer t, err;
  quantile_head dut (.raw(raw), .q(q));
  initial begin
    $readmemh("{os.path.abspath(workdir)}/r.mem", rm);
    $readmemh("{os.path.abspath(workdir)}/q.mem", qm);
    err = 0;
    for (t = 0; t < {n_vec}; t = t + 1) begin
      raw = rm[t]; #1;
      if (q !== qm[t]) begin err = err + 1; if (err < 4) $display("mismatch %0d q=%h exp=%h", t, q, qm[t]); end
    end
    $display("ERRORS=%0d", err); $finish;
  end
endmodule
"""
    return _run_verilator(workdir, "quantile_head", tb, [src])


def check_all(workdir: str = None, n_vec: int = 300) -> dict:
    workdir = workdir or os.path.join(os.environ.get("TIMESTRAL_KD_BUILD", os.path.join("runs", "hw")), "blocks")
    res = dict(rmsnorm=check_rmsnorm(os.path.join(workdir, "rmsnorm"), n_vec),
               softmax=check_softmax(os.path.join(workdir, "softmax"), n_vec),
               rope=check_rope(os.path.join(workdir, "rope"), n_vec),
               quantile_head=check_quantile_head(os.path.join(workdir, "quantile_head"), n_vec))
    print("P1 Verilator bit-exact check:", res)
    assert all(v == 0 for v in res.values()), "P1 RTL is not bit-exact vs references"
    print("P1 rmsnorm + softmax + rope + quantile_head RTL bit-exact vs fixedpoint references.")
    return res


if __name__ == "__main__":
    check_all()
