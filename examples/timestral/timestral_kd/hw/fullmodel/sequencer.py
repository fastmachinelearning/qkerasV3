"""Glue datapath + shared-MAC sequencer + BRAM weight loading (phase P2).

P1 proved the four non-dense blocks (rmsnorm, softmax, rope, quantile_head) bit-exact.
P2 supplies the *reusable* remainder of the datapath so P3 can wire a full ``model_top``:

  * glue      - relu, hard_sigmoid, residual add, token select (the "trivial glue" of the
                block list).  Fixed-point formats taken from ``fixedpoint.B3``.
  * mac_seq   - the shared-MAC sequencer: one multiplier + accumulator sweeps a dense
                layer's (out, in) grid, weights resident in a BRAM (``initial``-baked ROM,
                which is how a block RAM is inferred and how ``rdl``/yosys expect it).
  * bram       - a parameterised ROM loader used by mac_seq (address -> weight word).

Every integer datapath is transcribed from a numpy reference in this file, and the
Verilator testbenches compare against those references, so "matches the model" is proven on
the arithmetic.  Rounding is ``floor(x+0.5)`` / arithmetic shift, matching qkeras.

    python -m timestral_kd.hw.fullmodel.sequencer        # build + Verilator-check glue + mac_seq
"""

from __future__ import annotations

import math
import os
import re
import subprocess

import numpy as np

from ... import paths
from . import fixedpoint as fp

A = fp.fmt(fp.B3["a"])          # signed 7b, f=3, [-64, 63]
LU = fp.fmt(fp.B3["lut_out"])   # signed 7b, f=3 (relu output format)
SG = fp.fmt(fp.B3["sig_out"])   # unsigned 5b, f=5, [0, 31]


# ----------------------------------------------------------------------------- helpers
def _hex(v: int, width: int) -> str:
    return format(int(v) & ((1 << width) - 1), "x")


def _pack(codes, widths) -> int:
    v = off = 0
    for c, w in zip(codes, widths):
        v |= (int(c) & ((1 << w) - 1)) << off
        off += w
    return v


def _rom(name: str, data, width: int) -> str:
    lines = [f"  reg [{width - 1}:0] {name} [0:{len(data) - 1}];", "  initial begin"]
    for i, v in enumerate(data):
        lines.append(f"    {name}[{i}] = {width}'h{int(v) & ((1 << width) - 1):x};")
    lines.append("  end")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- references
def relu_ref(x: int) -> int:
    return int(np.clip(max(0, x), LU["lo"], LU["hi"]))


def hard_sigmoid_ref(x: int) -> int:
    """keras 3 hard_sigmoid = clip(x/6 + 0.5, 0, 1) = relu6(x+3)/6, to sig_out format."""
    t = np.clip(x + 3 * 2 ** A["f"], 0, 6 * 2 ** A["f"])          # units 2^-fa, [0, 6]
    val = t / (6.0 * 2 ** A["f"])                                 # in [0, 1]
    return int(np.clip(fp.qround(val * 2 ** SG["f"]), SG["lo"], SG["hi"]))


def resadd_ref(a: int, b: int) -> int:
    return int(np.clip(a + b, A["lo"], A["hi"]))


def select_ref(v, sel: int):
    return int(v[sel])


def mac_ref(x, w) -> int:
    """Integer MAC of a dense row: y = arithmetic_shift(sum x_i w_i, SH), sat to OW signed."""
    acc = int(np.dot(np.asarray(x, np.int64), np.asarray(w, np.int64)))
    return acc


# ----------------------------------------------------------------------------- glue RTL
def relu_verilog() -> str:
    return f"""module glue_relu (input signed [{A['bits'] - 1}:0] x, output [{LU['bits'] - 1}:0] y);
  // relu: negatives -> 0, else pass (max code {LU['hi']} already inside the signed width)
  assign y = x[{A['bits'] - 1}] ? {LU['bits']}'sd0 : x[{LU['bits'] - 1}:0];
endmodule
"""


def hard_sigmoid_verilog() -> str:
    # round(2t/3) = floor((t*43691 + 32768) >> 16); verified exhaustively vs hard_sigmoid_ref.
    return f"""module glue_hard_sigmoid (input signed [{A['bits'] - 1}:0] x, output [{SG['bits'] - 1}:0] y);
  wire signed [7:0] t0 = x + 8'sd{3 * 2 ** A['f']};
  wire [7:0] t = (t0 < 0) ? 8'd0 : (t0 > 8'sd{6 * 2 ** A['f']}) ? 8'd{6 * 2 ** A['f']} : t0[7:0];
  wire [29:0] m = t * 22'd43691 + 30'd32768;
  wire [13:0] c = m >> 16;
  assign y = (c > 14'd{SG['hi']}) ? {SG['bits']}'d{SG['hi']} : c[{SG['bits'] - 1}:0];
endmodule
"""


def resadd_verilog() -> str:
    return f"""module glue_resadd (input signed [{A['bits'] - 1}:0] a, input signed [{A['bits'] - 1}:0] b,
    output [{A['bits'] - 1}:0] y);
  wire signed [{A['bits']}:0] s = a + b;
  assign y = (s > {A['bits'] + 1}'sd{A['hi']}) ? {A['bits']}'sd{A['hi']} :
             (s < -{A['bits'] + 1}'sd{-A['lo']}) ? -{A['bits']}'sd{-A['lo']} : s[{A['bits'] - 1}:0];
endmodule
"""


def select_verilog(n: int, w: int) -> str:
    sw = max(1, int(math.ceil(math.log2(n))))
    return f"""module glue_select (input [{n * w - 1}:0] v, input [{sw - 1}:0] sel, output [{w - 1}:0] y);
  assign y = v[sel * {w} +: {w}];
endmodule
"""


# ----------------------------------------------------------------------------- mac_seq RTL
def mac_sequencer_verilog(I: int, O: int, XW: int, WW: int, OW: int, SH: int, W: np.ndarray) -> str:
    """Shared-MAC dense: one signed multiplier + accumulator sweeps an O x I grid.

    Weights live in a baked BRAM ``rom`` (O*I words of WW bits); the sequencer streams
    ``x[i] * rom[o*I + i]`` into ``acc`` and emits ``saturate(acc >>> SH)`` per output.
    Signed inputs ``x`` are held stable by the host for the whole token.
    """
    assert W.shape == (O, I), "weight shape"
    ACC = XW + WW + int(math.ceil(math.log2(I))) + 2
    L = []
    a = L.append
    a(f"// shared-MAC sequencer: {O} outputs x {I} inputs, {ACC}-bit accumulator, SH={SH}")
    a(f"module mac_seq (input clk, input rst, input start, input [{I * XW - 1}:0] x,")
    a(f"    output reg [{O * OW - 1}:0] y, output reg valid, output reg busy);")
    a("  localparam IDLE = 2'd0, MAC = 2'd1, EMIT = 2'd2;")
    a(_rom("rom", W.reshape(-1), WW))
    a("  reg [1:0] st;")
    a(f"  reg [{max(1, int(math.ceil(math.log2(I + 1)))) - 1}:0] ii;")
    a(f"  reg [{max(1, int(math.ceil(math.log2(O + 1)))) - 1}:0] oi;")
    a(f"  reg signed [{ACC - 1}:0] acc;")
    a(f"  wire signed [{XW - 1}:0] xi = x[ii * {XW} +: {XW}];")
    a(f"  wire signed [{WW - 1}:0] wi = rom[oi * {I} + ii];")
    a(f"  wire signed [{ACC - 1}:0] prod = xi * wi;")
    a(f"  wire signed [{ACC - 1}:0] shv = acc >>> {SH};")
    lo, hi = -(1 << (OW - 1)), (1 << (OW - 1)) - 1
    a(f"  wire signed [{OW - 1}:0] sat = (shv > {ACC}'sd{hi}) ? {OW}'sd{hi} : "
      f"(shv < -{ACC}'sd{-lo}) ? -{OW}'sd{-lo} : shv[{OW - 1}:0];")
    a("  always @(posedge clk) begin")
    a("    if (rst) begin st <= IDLE; ii <= 0; oi <= 0; acc <= 0; busy <= 0; valid <= 0; y <= 0; end")
    a("    else case (st)")
    a("      IDLE: begin valid <= 1'b0; busy <= 1'b0;")
    a("        if (start) begin st <= MAC; ii <= 0; oi <= 0; acc <= 0; busy <= 1'b1; end")
    a("      end")
    a("      MAC: begin")
    a("        acc <= acc + prod;")
    a(f"        if (ii == {I - 1}) st <= EMIT; else ii <= ii + 1'b1;")
    a("      end")
    a("      EMIT: begin")
    a(f"        y[oi * {OW} +: {OW}] <= sat; acc <= 0; ii <= 0;")
    a(f"        if (oi == {O - 1}) begin st <= IDLE; valid <= 1'b1; busy <= 1'b0; end")
    a("        else begin oi <= oi + 1'b1; st <= MAC; end")
    a("      end")
    a("      default: st <= IDLE;")
    a("    endcase")
    a("  end")
    a("endmodule")
    info = dict(I=I, O=O, XW=XW, WW=WW, OW=OW, SH=SH, ACC=ACC,
                in_w=I * XW, out_w=O * OW, cycles=O * (I + 1) + 1)
    return "\n".join(L) + "\n", info


# ----------------------------------------------------------------------------- Verilator
def _run(workdir: str, tb: str, srcs) -> int:
    os.makedirs(workdir, exist_ok=True)
    tb_path = os.path.join(workdir, "tb.v")
    open(tb_path, "w").write(tb)
    verilator = os.environ.get("VERILATOR", "verilator")
    r = subprocess.run([verilator, "--binary", "--timing", "-Wno-fatal", "-Wno-lint", "-Wno-style", "-O1",
                        "--top-module", "tb", tb_path] + list(srcs) + ["-Mdir", os.path.join(workdir, "obj")],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"verilator build failed:\n{r.stderr[-3000:]}")
    r = subprocess.run([os.path.abspath(os.path.join(workdir, "obj", "Vtb"))], capture_output=True,
                       text=True, cwd=workdir)
    m = re.search(r"ERRORS=(\d+)", r.stdout)
    if not m:
        raise RuntimeError(f"no result:\n{r.stdout[-1500:]}{r.stderr[-1500:]}")
    return int(m.group(1))


def check_glue(workdir: str, n_vec: int = 400, seed: int = 0) -> dict:
    os.makedirs(workdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    # --- relu
    X = rng.integers(A["lo"], A["hi"] + 1, n_vec); X[0], X[1] = A["lo"], A["hi"]
    open(os.path.join(workdir, "relu.v"), "w").write(relu_verilog())
    with open(os.path.join(workdir, "rx.mem"), "w") as fx, open(os.path.join(workdir, "ry.mem"), "w") as fy:
        for x in X:
            fx.write(_hex(x, A["bits"]) + "\n"); fy.write(_hex(relu_ref(int(x)), LU["bits"]) + "\n")
    tb = f"""`timescale 1ns/1ps
module tb; reg [{A['bits']-1}:0] xm [0:{n_vec-1}]; reg [{LU['bits']-1}:0] ym [0:{n_vec-1}];
  reg signed [{A['bits']-1}:0] x; wire [{LU['bits']-1}:0] y; integer t, err; glue_relu dut(.x(x),.y(y));
  initial begin $readmemh("{os.path.abspath(workdir)}/rx.mem",xm); $readmemh("{os.path.abspath(workdir)}/ry.mem",ym);
    err=0; for(t=0;t<{n_vec};t=t+1) begin x=xm[t]; #1; if(y!==ym[t]) err=err+1; end
    $display("ERRORS=%0d",err); $finish; end endmodule
"""
    r_relu = _run(os.path.join(workdir, "relu"), tb, [os.path.join(workdir, "relu.v")])
    # --- hard_sigmoid
    open(os.path.join(workdir, "hs.v"), "w").write(hard_sigmoid_verilog())
    with open(os.path.join(workdir, "hx.mem"), "w") as fx, open(os.path.join(workdir, "hy.mem"), "w") as fy:
        for x in X:
            fx.write(_hex(x, A["bits"]) + "\n"); fy.write(_hex(hard_sigmoid_ref(int(x)), SG["bits"]) + "\n")
    tb = f"""`timescale 1ns/1ps
module tb; reg [{A['bits']-1}:0] xm [0:{n_vec-1}]; reg [{SG['bits']-1}:0] ym [0:{n_vec-1}];
  reg signed [{A['bits']-1}:0] x; wire [{SG['bits']-1}:0] y; integer t, err; glue_hard_sigmoid dut(.x(x),.y(y));
  initial begin $readmemh("{os.path.abspath(workdir)}/hx.mem",xm); $readmemh("{os.path.abspath(workdir)}/hy.mem",ym);
    err=0; for(t=0;t<{n_vec};t=t+1) begin x=xm[t]; #1; if(y!==ym[t]) err=err+1; end
    $display("ERRORS=%0d",err); $finish; end endmodule
"""
    r_hs = _run(os.path.join(workdir, "hs"), tb, [os.path.join(workdir, "hs.v")])
    # --- residual add
    open(os.path.join(workdir, "ra.v"), "w").write(resadd_verilog())
    Xa = rng.integers(A["lo"], A["hi"] + 1, n_vec); Xb = rng.integers(A["lo"], A["hi"] + 1, n_vec)
    with open(os.path.join(workdir, "ra.mem"), "w") as fa, open(os.path.join(workdir, "rb.mem"), "w") as fb, \
            open(os.path.join(workdir, "ry2.mem"), "w") as fy:
        for a_, b_ in zip(Xa, Xb):
            fa.write(_hex(a_, A["bits"]) + "\n"); fb.write(_hex(b_, A["bits"]) + "\n")
            fy.write(_hex(resadd_ref(int(a_), int(b_)), A["bits"]) + "\n")
    tb = f"""`timescale 1ns/1ps
module tb; reg [{A['bits']-1}:0] am [0:{n_vec-1}]; reg [{A['bits']-1}:0] bm [0:{n_vec-1}];
  reg [{A['bits']-1}:0] ym [0:{n_vec-1}]; reg signed [{A['bits']-1}:0] a,b; wire [{A['bits']-1}:0] y;
  integer t, err; glue_resadd dut(.a(a),.b(b),.y(y));
  initial begin $readmemh("{os.path.abspath(workdir)}/ra.mem",am); $readmemh("{os.path.abspath(workdir)}/rb.mem",bm);
    $readmemh("{os.path.abspath(workdir)}/ry2.mem",ym); err=0;
    for(t=0;t<{n_vec};t=t+1) begin a=am[t]; b=bm[t]; #1; if(y!==ym[t]) err=err+1; end
    $display("ERRORS=%0d",err); $finish; end endmodule
"""
    r_ra = _run(os.path.join(workdir, "ra"), tb, [os.path.join(workdir, "ra.v")])
    # --- token select (N=8 words of A-width)
    n, w = 8, A["bits"]
    open(os.path.join(workdir, "sel.v"), "w").write(select_verilog(n, w))
    V = rng.integers(A["lo"], A["hi"] + 1, (n_vec, n)); S = rng.integers(0, n, n_vec)
    with open(os.path.join(workdir, "sv.mem"), "w") as fv, open(os.path.join(workdir, "ss.mem"), "w") as fs_, \
            open(os.path.join(workdir, "sy.mem"), "w") as fy:
        for v, s in zip(V, S):
            fv.write(_hex(_pack(v, [w] * n), n * w) + "\n"); fs_.write(_hex(int(s), 3) + "\n")
            fy.write(_hex(select_ref(v, int(s)), w) + "\n")
    tb = f"""`timescale 1ns/1ps
module tb; reg [{n*w-1}:0] vm [0:{n_vec-1}]; reg [2:0] sm [0:{n_vec-1}]; reg [{w-1}:0] ym [0:{n_vec-1}];
  reg [{n*w-1}:0] v; reg [2:0] sel; wire [{w-1}:0] y; integer t, err; glue_select dut(.v(v),.sel(sel),.y(y));
  initial begin $readmemh("{os.path.abspath(workdir)}/sv.mem",vm); $readmemh("{os.path.abspath(workdir)}/ss.mem",sm);
    $readmemh("{os.path.abspath(workdir)}/sy.mem",ym); err=0;
    for(t=0;t<{n_vec};t=t+1) begin v=vm[t]; sel=sm[t]; #1; if(y!==ym[t]) err=err+1; end
    $display("ERRORS=%0d",err); $finish; end endmodule
"""
    r_sel = _run(os.path.join(workdir, "sel"), tb, [os.path.join(workdir, "sel.v")])
    return dict(relu=r_relu, hard_sigmoid=r_hs, resadd=r_ra, select=r_sel)


def check_mac(workdir: str, I: int = 8, O: int = 4, XW: int = 7, WW: int = 6, OW: int = 10, SH: int = 4,
              n_vec: int = 200, seed: int = 1) -> int:
    rng = np.random.default_rng(seed)
    W = rng.integers(-(1 << (WW - 1)), 1 << (WW - 1), (O, I))
    v, info = mac_sequencer_verilog(I, O, XW, WW, OW, SH, W)
    os.makedirs(workdir, exist_ok=True)
    open(os.path.join(workdir, "mac_seq.v"), "w").write(v)
    lo_x, hi_x = -(1 << (XW - 1)), (1 << (XW - 1)) - 1
    X = rng.integers(lo_x, hi_x + 1, (n_vec, I)); X[0], X[1] = lo_x, hi_x
    lo_o, hi_o = -(1 << (OW - 1)), (1 << (OW - 1)) - 1
    with open(os.path.join(workdir, "mx.mem"), "w") as fx, open(os.path.join(workdir, "my.mem"), "w") as fy:
        for x in X:
            y = [int(np.clip(mac_ref(x, W[o]) >> SH, lo_o, hi_o)) for o in range(O)]
            fx.write(_hex(_pack(x, [XW] * I), I * XW) + "\n")
            fy.write(_hex(_pack(y, [OW] * O), O * OW) + "\n")
    wait = info["cycles"] + 40
    tb = f"""`timescale 1ns/1ps
module tb;
  reg clk=0; always #5 clk=~clk;
  reg [{info['in_w']-1}:0] xm [0:{n_vec-1}]; reg [{info['out_w']-1}:0] ym [0:{n_vec-1}];
  reg [{info['in_w']-1}:0] x; reg start=0, rst=1; wire [{info['out_w']-1}:0] y; wire valid, busy;
  integer t, c, err; mac_seq dut(.clk(clk),.rst(rst),.start(start),.x(x),.y(y),.valid(valid),.busy(busy));
  initial begin
    $readmemh("{os.path.abspath(workdir)}/mx.mem",xm); $readmemh("{os.path.abspath(workdir)}/my.mem",ym); err=0;
    @(negedge clk); @(negedge clk); rst=0;
    for(t=0;t<{n_vec};t=t+1) begin
      @(negedge clk); x=xm[t]; start=1; @(negedge clk); start=0;
      c=0; while(!valid && c<{wait}) begin @(posedge clk); #1; c=c+1; end
      if(!valid || y!==ym[t]) begin err=err+1; if(err<4) $display("mismatch %0d y=%h exp=%h",t,y,ym[t]); end
    end
    $display("ERRORS=%0d",err); $finish;
  end
endmodule
"""
    return _run(workdir, tb, [os.path.join(workdir, "mac_seq.v")])


def check_all(workdir: str | None = None) -> dict:
    workdir = workdir or str(paths.RUNS / "hw" / "seq")
    res = dict(check_glue(workdir))
    res["mac_seq"] = check_mac(workdir)
    print("P2 Verilator bit-exact check:", res)
    assert all(v == 0 for v in res.values()), "P2 RTL is not bit-exact vs references"
    print("P2 glue + shared-MAC sequencer + BRAM weight loading bit-exact vs references.")
    return res


if __name__ == "__main__":
    check_all()