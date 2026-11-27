"""Verilator check of the generated engine against the instruction-level simulator (bit-exact).

``verify(img, x_codes, workdir)`` writes the engine, runs ``x_codes (N, L*P)`` (one variate per
run) through it in Verilator and returns mismatches against ``isa.simulate`` plus the measured
clock cycles per run.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np

from . import engine_rtl
from .isa import Image, simulate

TB = r"""`timescale 1ns/1ps
module tb;
    reg clk = 0, rst = 1, start = 0, host_we = 0;
    reg [{RAM_AW}-1:0] host_addr = 0;
    reg [31:0] host_wdata = 0;
    wire [31:0] host_rdata;
    wire busy, done;
    timestral_engine dut(.clk(clk), .rst(rst), .start(start), .busy(busy), .done(done), .host_we(host_we),
                         .host_addr(host_addr), .host_wdata(host_wdata), .host_rdata(host_rdata));
    always #5 clk = ~clk;
    reg [31:0] xin [0:{NIN}-1];
    integer n, i, fo, cyc;
    initial begin
        $readmemh("xin.hex", xin);
        fo = $fopen("out.txt", "w");
        repeat (3) @(posedge clk);
        rst = 0;
        for (n = 0; n < {N}; n = n + 1) begin
            for (i = 0; i < {IN_LEN}; i = i + 1) begin
                @(negedge clk); host_we = 1; host_addr = {XIN_BASE} + i; host_wdata = xin[n * {IN_LEN} + i];
            end
            @(negedge clk); host_we = 0; start = 1;
            @(negedge clk); start = 0;
            cyc = 0;
            while (!done) begin @(negedge clk); cyc = cyc + 1; end
            for (i = 0; i < {OUT_LEN}; i = i + 1) begin
                @(negedge clk); host_addr = {OUT_BASE} + i;
                @(negedge clk); $fwrite(fo, "%0d ", $signed(host_rdata));
            end
            $fwrite(fo, "\n");
            $fwrite(fo, "#cycles %0d\n", cyc);
        end
        $fclose(fo);
        $finish;
    end
endmodule
"""


def verify(img: Image, x_codes: np.ndarray, workdir: str, verilator: str | None = None, keep: bool = True) -> dict:
    verilator = verilator or os.environ.get("VERILATOR", "verilator")
    os.makedirs(workdir, exist_ok=True)
    p = engine_rtl.write(img, workdir)
    N = x_codes.shape[0]
    with open(os.path.join(workdir, "xin.hex"), "w") as f:
        for v in np.asarray(x_codes, np.int64).reshape(-1):
            f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")
    with open(os.path.join(workdir, "tb.v"), "w") as f:
        f.write(TB.format(N=N, NIN=N * img.in_len, **p))
    obj = os.path.join(workdir, "obj")
    r = subprocess.run([verilator, "--binary", "--timing", "-Wno-fatal", "-Wno-lint", "-Wno-style", "-O2",
                        "--top-module", "tb", "-Mdir", obj, "tb.v", "engine.v"], cwd=workdir,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("verilator build failed:\n" + r.stderr[-3000:])
    r = subprocess.run([os.path.join(obj, "Vtb")], cwd=workdir, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        raise RuntimeError("simulation failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    rows, cycles = [], []
    for line in open(os.path.join(workdir, "out.txt")):
        line = line.strip()
        if line.startswith("#cycles"):
            cycles.append(int(line.split()[1]))
        elif line:
            rows.append([int(t) for t in line.split()])
    got = np.asarray(rows, np.int64)
    ref = simulate(img, np.asarray(x_codes, np.int64))
    res = dict(n_runs=N, mismatches=int(np.sum(got != ref)), n=int(ref.size),
               cycles_min=int(min(cycles)), cycles_max=int(max(cycles)), params=p)
    if not keep:
        shutil.rmtree(obj, ignore_errors=True)
    return res
