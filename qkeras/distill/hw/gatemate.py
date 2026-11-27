"""Cologne Chip GateMate (CCGM1A1) device constants and cost primitives.

Planning estimates for ranking student configurations, not synthesis results.  Every constant
is an explicit, editable assumption; ``DEVICES`` can hold other parts (e.g. a CCGM1A2).

Constant-weight dense layers (every weight baked into shift-add logic) are costed from the
*quantised* kernels: with da4ml installed, ``CPE_PER_DA4ML_COST`` x da4ml adder cost;
otherwise ``CPE_PER_DIGIT_BIT`` x CSD digits x (input bits + weight bits + log2 fan-in).
Both are calibrated against yosys ``synth_gatemate`` + nextpnr-himbaechel of the 14 dense
layers of the Timestral student ``B_d16_L1_b3`` (da4ml 0.6, max error 16 % / 23 %).
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile

import numpy as np

from ..weights import quantized_kernels

GATEMATE = {
    "cpe": 20480,                   # CCGM1A1 programmable elements
    "bram_bits": 32 * 40 * 1024,    # 32 x 40 kbit block RAM (datasheet; verify for your board)
    "lut4_to_cpe": 1.83,            # measured in the logicnet-fpga paper (4-LUT ~ 1.83 CPEs)
    "cpe_per_adder_bit": 1.0,       # assumption for multipliers: one CPE per full-adder bit
    "f_clk_hz": 50e6,               # assumption for a pipelined design on the A1
    "util_limit": 0.8,              # keep 20 % for routing / control
}
DEVICES = {"CCGM1A1": GATEMATE}

CPE_PER_DA4ML_COST = 1.59
CPE_PER_DIGIT_BIT = 0.56


def mult_cpe(a_bits: int, b_bits: int, g=GATEMATE) -> float:
    """Array multiplier ~ a*b full-adder cells."""
    return a_bits * b_bits * g["cpe_per_adder_bit"]


def lut_logic_cpe(in_bits: int, out_bits: int, g=GATEMATE) -> float:
    """Table with ``in_bits`` inputs per output bit as a 4-LUT tree (upper bound)."""
    return out_bits * max(1, 2 ** max(in_bits - 4, 0)) * g["lut4_to_cpe"]


def csd_digits(v: np.ndarray) -> np.ndarray:
    """Number of non-zero canonical-signed-digit digits of integers ``v``."""
    v = np.abs(v.astype(np.int64))
    n = np.zeros_like(v)
    while np.any(v):
        odd = (v & 1).astype(bool)
        r = np.where(odd, 2 - (v & 3), 0)      # v mod 4 == 3 -> digit -1 and carry
        n += (r != 0)
        v = (v - r) >> 1
    return n


def exact_frac_bits(k: np.ndarray, max_f: int = 24) -> int:
    """Smallest f with k * 2**f integer (per-channel power-of-two scales make f > w - 2)."""
    for f in range(max_f + 1):
        if np.allclose(k * 2.0 ** f, np.round(k * 2.0 ** f)):
            return f
    return max_f


def integer_kernels(qmodel, frac_bits: int | None = None):
    """Integer-scaled quantised kernels ``[(name, int array)]`` of all quantised dense layers.

    ``frac_bits=None`` scales every kernel exactly.  A fixed ``frac_bits`` rounds and silently
    drops weights below ``2**-frac_bits`` (``auto_po2`` scales put most 3-4 bit weights below
    0.25, so a fixed 1-bit fraction under-counts non-zero weights by about 6x).
    """
    out = []
    for name, k in quantized_kernels(qmodel):
        f = exact_frac_bits(k) if frac_bits is None else frac_bits
        out.append((name, np.round(k * 2.0 ** f)))
    return out


def estimate_dense_cpe(qmodel, in_bits: int, w_bits: int, use_da4ml: bool = True, in_int_bits: int = 3) -> dict:
    """Calibrated CPE of all constant-weight dense layers (one instance each).

    ``in_bits`` / ``in_int_bits``: signed input format ``quantized_bits(in_bits, in_int_bits)``.
    """
    kernels = integer_kernels(qmodel)
    digits = sum(int(csd_digits(k).sum()) for _, k in kernels)
    nz = sum(int(np.count_nonzero(k)) for _, k in kernels)
    fallback = sum(int(csd_digits(k).sum()) * (in_bits + w_bits + math.ceil(math.log2(max(k.shape[0], 2))))
                   for _, k in kernels) * CPE_PER_DIGIT_BIT
    res = {"csd_digits": digits, "nonzero_weights": nz, "dense_cpe": float(fallback), "method": "csd"}
    if use_da4ml:
        try:
            import da4ml.trace  # noqa: F401
        except ImportError:
            return res
        cost = da4ml_cost([k for _, k in quantized_kernels(qmodel)], in_int_bits, max(in_bits - 1 - in_int_bits, 0))
        res.update(dense_cpe=cost * CPE_PER_DA4ML_COST, da4ml_cost=cost, method="da4ml")
    return res


_DA4ML_CHILD = """
import sys, numpy as np, da4ml.trace as tr
z = np.load(sys.argv[1]); i, f = int(z["i"]), int(z["f"]); cost = 0.0
for n in sorted(k for k in z.files if k.startswith("k")):
    k = z[n]; m = k.shape[0]
    x = tr.FixedVariableArray.from_kif(np.ones(m, bool), np.full(m, i), np.full(m, f))
    cost += float(tr.comb_trace(x, x @ k).cost)
print(repr(cost))
"""


def da4ml_cost(kernels, in_int_bits: int, in_frac_bits: int) -> float:
    """Summed da4ml adder cost of constant-weight ``x @ k`` for each (float) kernel ``k``.

    da4ml ships its own libomp, which crashes on macOS once TensorFlow has run ops in the same
    process unless ``OMP_NUM_THREADS=1`` was set at process start; in that case the trace runs in
    a child process with that environment (same result).
    """
    if sys.platform == "darwin" and os.environ.get("OMP_NUM_THREADS") != "1":
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "k.npz")
            np.savez(path, i=in_int_bits, f=in_frac_bits, **{f"k{j:04d}": k for j, k in enumerate(kernels)})
            r = subprocess.run([sys.executable, "-c", _DA4ML_CHILD, path], capture_output=True, text=True,
                               env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
        return float(r.stdout.strip().splitlines()[-1])
    import da4ml.trace as tr

    cost = 0.0
    for k in kernels:
        n = k.shape[0]
        x = tr.FixedVariableArray.from_kif(np.ones(n, bool), np.full(n, in_int_bits), np.full(n, in_frac_bits))
        cost += float(tr.comb_trace(x, x @ k).cost)
    return cost
