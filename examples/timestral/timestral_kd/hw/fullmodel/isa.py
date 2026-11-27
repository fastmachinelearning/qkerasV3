"""Instruction set, compiler and instruction-level simulator of the Timestral GateMate engine.

The engine is a small sequential tensor processor.  Every instruction is one loop nest

    for r in range(R):                       # rows
      for c in range(C or c <= r):           # columns (``tri_c``: causal, c <= r)
        acc = init[r, c]                     # 0, -inf (MAX) or a value from RAM / CROM (optionally negated)
        for k in range(K or k <= r):         # reduction (``tri_k``: causal, k <= r)
          acc = acc + A[r, c, k] * B[r, c, k]        # op MAC
          acc = max(acc, A[r, c, k])                 # op MAX
        out[r, c] = epilogue(acc)

with operand addresses ``base + r*sr + c*sc + k*sk`` in one of four memories:

* ``RAM``   scratchpad (signed words); the input window is written to ``XIN`` by the host and the
            ``head_post`` codes are read from ``OUT``,
* ``WROM``  dense kernels (small signed integers),
* ``CROM``  constants: biases (pre-aligned), gains, RoPE tables, the forecast-query token, immediates,
* ``LROM``  lookup tables (rsqrt, exp, reciprocal, activation tables), only used by the epilogue.

Epilogue (in this order): ``div`` (round-half-even division by a constant) *or* ``shift``
(right shift with ``rne`` / ``rhu`` rounding, left shift when negative), ``relu``, ``clip``,
``lut`` (``LROM[base + value - offset]``).

``compile_program(prog)`` lowers ``program.forward`` for ONE variate (the student is
channel-independent, so the host starts the engine once per variate).  ``simulate`` executes the
instructions on numpy arrays; ``check`` asserts that it equals ``program.forward`` bit-exactly and
returns the value ranges that size the hardware words.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .program import HWFormats, Program, rhu_shift, rne_div, rne_shift

RAM, WROM, CROM, IMM = 0, 1, 2, 3       # IMM = CROM with all strides 0 (kept separate for readability)
SPACES = {RAM: "RAM", WROM: "WROM", CROM: "CROM", IMM: "CROM"}
MAC, MAX = 0, 1
RNE, RHU = 0, 1


@dataclass
class Opnd:
    space: int = RAM
    base: int = 0
    sr: int = 0
    sc: int = 0
    sk: int = 0

    def addr(self, r, c, k):
        return self.base + r * self.sr + c * self.sc + k * self.sk


@dataclass
class Instr:
    note: str
    R: int
    C: int
    K: int
    a: Opnd
    b: Opnd
    out: Opnd
    op: int = MAC
    tri_c: bool = False
    tri_k: bool = False
    init: Opnd | None = None          # None: 0 (MAC) / -inf (MAX)
    init_neg: bool = False
    shift: int = 0
    rnd: int = RNE
    div: int = 1
    relu: bool = False
    clip: tuple | None = None
    lut: tuple | None = None          # (LROM base, offset)


@dataclass
class Image:
    instrs: list
    wrom: np.ndarray
    crom: np.ndarray
    lrom: np.ndarray
    ram_words: int
    xin: int                          # RAM base of the input window (L*P codes)
    out: int                          # RAM base of the head_post codes (h * P*Q)
    in_len: int
    out_len: int
    luts: dict = field(default_factory=dict)      # name -> (base, offset)
    bufs: dict = field(default_factory=dict)      # name -> (base, rows, cols)

    def to_dict(self):
        return dict(instrs=[asdict(i) for i in self.instrs], ram_words=self.ram_words, xin=self.xin,
                    out=self.out, in_len=self.in_len, out_len=self.out_len, luts=self.luts, bufs=self.bufs)


# ----------------------------------------------------------------------------- compiler
class _Mem:
    def __init__(self):
        self.data: list[int] = []
        self.index: dict = {}

    def put(self, key, values) -> int:
        if key in self.index:
            return self.index[key]
        base = len(self.data)
        self.data.extend(int(v) for v in np.asarray(values, np.int64).reshape(-1))
        self.index[key] = base
        return base


def compile_program(prog: Program) -> Image:
    d, hw, c = prog.dims, prog.hw, prog.cfg
    P, L, h, S, D, H, hd, M, n = (d[k] for k in ("P", "L", "h", "S", "D", "H", "hd", "M", "n"))
    Q, EH, HH = d["Q"], d["EH"], d["HH"]
    Fa, Fms, Frs, Fg = prog.F("a"), prog.F("ms"), prog.F("rsqrt"), prog.F("gain")
    Fs, Fe, Fi, Fp, Fsig = prog.F("score"), prog.F("exp"), prog.F("inv"), prog.F("prob"), prog.F("sig_out")
    fin = HWFormats(**hw).in_f
    fres = hw["tok_f"]
    wrom, crom, lrom = _Mem(), _Mem(), _Mem()
    ins: list[Instr] = []
    bufs: dict = {}
    top = [0]

    def buf(name, rows, cols):
        bufs[name] = (top[0], rows, cols)
        top[0] += rows * cols
        return bufs[name][0]

    def imm(v):
        return Opnd(IMM, crom.put(("imm", int(v)), [v]))

    luts = {}
    for name, tab in prog.luts.items():
        luts[name] = lrom.put(("lut", name), tab)

    def lut(name, offset):
        return (luts[name], int(offset))

    def clipF(F):
        return (int(F["lo"]), int(F["hi"]))

    relu_bits = None
    if c["act_fn"] == "relu":
        bits, integer = (int(v) for v in prog.q["lut_out"].split("(")[1].split(")")[0].split(",")[:2])
        relu_bits = (bits, bits - integer)

    def dense(note, name, xbase, xcols, rows, fx, out, init=None, krows=None):
        """RAM rows (rows x K at xbase, row stride xcols) @ W -> out (rows x O); exact; returns f.

        ``krows``: use only the first rows of W (the embedding's missing-flag inputs are zero)."""
        dd = prog.dense[name]
        W = dd["W"] if krows is None else dd["W"][:krows]
        K, O = W.shape
        fy = fx + dd["fw"]
        wb = wrom.put(("W", name, W.shape[0]), W)
        if init is None and dd["b"] is not None:
            init = Opnd(CROM, crom.put(("b", name, fy), dd["b"] << (fy - dd["fb"])), 0, 1, 0)
        ins.append(Instr(note, rows, O, K, Opnd(RAM, xbase, xcols, 0, 1), Opnd(WROM, wb, 0, 1, O),
                         Opnd(RAM, out, O, 1, 0), init=init))
        return fy

    def act(note, src, rows, cols, sstride, fx, out):
        """activation (relu / table) of a RAM block -> out codes; returns f."""
        if relu_bits:
            bits, fo = relu_bits
            ins.append(Instr(note, rows, cols, 1, Opnd(RAM, src, sstride, 1, 0), imm(1), Opnd(RAM, out, cols, 1, 0),
                             relu=True, shift=fx - fo, clip=(0, (1 << bits) - 1)))
            return fo
        Flin, Flo = prog.F("lut_in"), prog.F("lut_out")
        ins.append(Instr(note, rows, cols, 1, Opnd(RAM, src, sstride, 1, 0), imm(1), Opnd(RAM, out, cols, 1, 0),
                         shift=fx - Flin["f"], clip=clipF(Flin), lut=lut("act", Flin["lo"])))
        return Flo["f"]

    def requant_add(note, a_base, fa_, b_base, fb_, rows, cols, out, F):
        """out = quantizer F (a + b) with a, b at their own f (exact alignment)."""
        f = max(fa_, fb_)
        tmp = buf(note + "_al", rows, cols)
        ins.append(Instr(note + " align", rows, cols, 1, Opnd(RAM, a_base, cols, 1, 0), imm(1 << (f - fa_)),
                         Opnd(RAM, tmp, cols, 1, 0)))
        ins.append(Instr(note, rows, cols, 1, Opnd(RAM, b_base, cols, 1, 0), imm(1 << (f - fb_)),
                         Opnd(RAM, out, cols, 1, 0), init=Opnd(RAM, tmp, cols, 1, 0), shift=f - F["f"],
                         clip=clipF(F)))

    def rmsnorm(note, name, res, out):
        fm = min(fres, hw["ms_f"])
        xt = buf(note + "_xm", S, D)
        r = buf(note + "_r", S, 1)
        rg = buf(note + "_rg", S, D)
        ins.append(Instr(note + " x->ms_f", S, D, 1, Opnd(RAM, res, D, 1, 0), imm(1), Opnd(RAM, xt, D, 1, 0),
                         shift=fres - fm, rnd=RHU))
        dv = D << (2 * fm - Fms["f"])
        rnd = dict(shift=dv.bit_length() - 1) if dv & (dv - 1) == 0 else dict(div=dv)   # power of two: shift
        ins.append(Instr(note + " ms,rsqrt", S, 1, D, Opnd(RAM, xt, D, 0, 1), Opnd(RAM, xt, D, 0, 1),
                         Opnd(RAM, r, 1, 0, 0), clip=clipF(Fms), lut=lut("rsqrt", Fms["lo"]), **rnd))
        gb = crom.put(("gain", name), prog.gains[name])
        ins.append(Instr(note + " r*g", S, D, 1, Opnd(RAM, r, 1, 0, 0), Opnd(CROM, gb, 0, 1, 0),
                         Opnd(RAM, rg, D, 1, 0)))
        ins.append(Instr(note, S, D, 1, Opnd(RAM, res, D, 1, 0), Opnd(RAM, rg, D, 1, 0), Opnd(RAM, out, D, 1, 0),
                         shift=fres + Frs["f"] + Fg["f"] - Fa["f"], clip=clipF(Fa)))
        return Fa["f"]

    # ---------------- input and embedding
    xin = buf("xin", L, P)
    h1 = buf("embed_hidden", L, EH)
    f1 = dense("embed_hidden", "embed_hidden", xin, P, L, fin, h1, krows=P)   # missing-flag half is zero
    a1 = buf("embed_act", L, EH)
    fa1 = act("embed_act", h1, L, EH, EH, f1, a1)
    o1 = buf("embed_output", L, D)
    fo1 = dense("embed_output", "embed_output", a1, EH, L, fa1, o1)
    r1 = buf("embed_residual", L, D)
    fr1 = dense("embed_residual", "embed_residual", xin, P, L, fin, r1, krows=P)
    e = buf("embed_post", L, D)
    requant_add("embed_post", o1, fo1, r1, fr1, L, D, e, Fa)

    # ---------------- residual stream (tok_f), forecast-query tokens
    res = buf("res", S, D)
    ins.append(Instr("res <- embed", L, D, 1, Opnd(RAM, e, D, 1, 0), imm(1 << (fres - Fa["f"])),
                     Opnd(RAM, res, D, 1, 0)))
    tb = crom.put(("token",), prog.token)
    ins.append(Instr("res <- token", h, D, 0, Opnd(), Opnd(), Opnd(RAM, res + L * D, D, 1, 0),
                     init=Opnd(CROM, tb, 0, 1, 0)))

    cosb = crom.put(("cos",), prog.cos)
    sinb = crom.put(("sin",), prog.sin)
    nsin = crom.put(("nsin",), -prog.sin)
    # rope tables per (s, j, k): segment 1 [cos, -sin], segment 3 [cos, +sin]
    t1 = crom.put(("rope1",), np.stack([prog.cos, -prog.sin], -1))
    t3 = crom.put(("rope3",), np.stack([prog.cos, prog.sin], -1))
    del cosb, sinb, nsin

    for b in range(d["B"]):
        nb = buf(f"b{b}_attn_norm0", S, D)
        rmsnorm(f"b{b}_attn_norm0", f"b{b}_attn_norm0", res, nb)
        qkv = buf(f"b{b}_qkv", S, 3 * D)
        fq = dense(f"b{b}_wQKV", f"b{b}_attn0_wQKV", nb, D, S, Fa["f"], qkv)
        # qkv column of (t, h, j) = (t*hd + j)*H + h
        qr = buf(f"b{b}_qr", S, D)       # rotated q, layout (s, h*hd + j)
        kr = buf(f"b{b}_kr", S, D)
        half = hd // 2
        sh = fq + hw["rope_f"] - hw["qrot_f"]
        for t, dst in ((0, qr), (1, kr)):
            for hh in range(H):
                col = lambda j: (t * hd + j) * H + hh           # noqa: E731
                ob = dst + hh * hd
                # rotated low part: y_j = x_j cos_j - x_{j+half} sin_j
                ins.append(Instr(f"b{b} rope{t}{hh} lo", S, n, 2, Opnd(RAM, qkv + col(0), 3 * D, H, half * H),
                                 Opnd(CROM, t1, 2 * n, 2, 1), Opnd(RAM, ob, D, 1, 0), shift=sh, rnd=RHU))
                if half > n:
                    ins.append(Instr(f"b{b} rope{t}{hh} pass lo", S, half - n, 1,
                                     Opnd(RAM, qkv + col(n), 3 * D, H, 0), imm(1 << hw["rope_f"]),
                                     Opnd(RAM, ob + n, D, 1, 0), shift=sh, rnd=RHU))
                # rotated high part: y_{j+half} = x_{j+half} cos_j + x_j sin_j
                ins.append(Instr(f"b{b} rope{t}{hh} hi", S, n, 2, Opnd(RAM, qkv + col(half), 3 * D, H, -half * H),
                                 Opnd(CROM, t3, 2 * n, 2, 1), Opnd(RAM, ob + half, D, 1, 0), shift=sh, rnd=RHU))
                if hd > half + n:
                    ins.append(Instr(f"b{b} rope{t}{hh} pass hi", S, hd - half - n, 1,
                                     Opnd(RAM, qkv + col(half + n), 3 * D, H, 0), imm(1 << hw["rope_f"]),
                                     Opnd(RAM, ob + half + n, D, 1, 0), shift=sh, rnd=RHU))
        fr = hw["qrot_f"]
        att = buf(f"b{b}_attn", S, D)    # layout (s, h*hd + j)
        sc = buf(f"b{b}_sc", S, S)
        sc2 = buf(f"b{b}_sc2", S, S)
        mx = buf(f"b{b}_max", S, 1)
        ee = buf(f"b{b}_e", S, S)
        inv = buf(f"b{b}_inv", S, 1)
        pp = buf(f"b{b}_p", S, S)
        for hh in range(H):
            ins.append(Instr(f"b{b} qk h{hh}", S, S, hd, Opnd(RAM, qr + hh * hd, D, 0, 1),
                             Opnd(RAM, kr + hh * hd, 0, D, 1),
                             Opnd(RAM, sc, S, 1, 0), tri_c=True, shift=2 * fr - hw["score_f"], rnd=RHU))
            ins.append(Instr(f"b{b} /sqrt(hd) h{hh}", S, S, 1, Opnd(RAM, sc, S, 1, 0), imm(prog.invsqrt),
                             Opnd(RAM, sc2, S, 1, 0), tri_c=True, shift=hw["invsqrt_f"], rnd=RHU))
            ins.append(Instr(f"b{b} max h{hh}", S, 1, S, Opnd(RAM, sc2, S, 0, 1), Opnd(), Opnd(RAM, mx, 1, 0, 0),
                             op=MAX, tri_k=True))
            ins.append(Instr(f"b{b} exp h{hh}", S, S, 1, Opnd(RAM, sc2, S, 1, 0), imm(1), Opnd(RAM, ee, S, 1, 0),
                             tri_c=True, init=Opnd(RAM, mx, 1, 0, 0), init_neg=True,
                             shift=hw["score_f"] - Fs["f"], clip=clipF(Fs), lut=lut("exp", Fs["lo"])))
            ins.append(Instr(f"b{b} sum,inv h{hh}", S, 1, S, Opnd(RAM, ee, S, 0, 1), imm(1), Opnd(RAM, inv, 1, 0, 0),
                             tri_k=True, lut=lut("inv", 1)))
            ins.append(Instr(f"b{b} prob h{hh}", S, S, 1, Opnd(RAM, ee, S, 1, 0), Opnd(RAM, inv, 1, 0, 0),
                             Opnd(RAM, pp, S, 1, 0), tri_c=True, shift=Fe["f"] + Fi["f"] - Fp["f"], clip=clipF(Fp)))
            # o[q, h, j] = sum_k p[q, k] v[k, h, j];  v column (2*hd + j)*H + h
            ins.append(Instr(f"b{b} pv h{hh}", S, hd, S, Opnd(RAM, pp, S, 0, 1),
                             Opnd(RAM, qkv + 2 * hd * H + hh, 0, H, 3 * D), Opnd(RAM, att + hh * hd, D, 1, 0),
                             tri_k=True, shift=Fp["f"] + fq - Fa["f"], clip=clipF(Fa)))
        gd = buf(f"b{b}_gate", S, D)
        fgd = dense(f"b{b}_gate", f"b{b}_attn0_gate", nb, D, S, Fa["f"], gd)
        gs = buf(f"b{b}_gate_q", S, D)
        if c["gate_fn"] == "hard_sigmoid":
            # (x * 2^fs + 3 * 2^(fx+fs)) / (6 * 2^fx), round half even, clip
            ins.append(Instr(f"b{b} hard_sigmoid", S, D, 1, Opnd(RAM, gd, D, 1, 0), imm(1 << Fsig["f"]),
                             Opnd(RAM, gs, D, 1, 0), init=imm(3 << (fgd + Fsig["f"])), div=6 << fgd,
                             clip=clipF(Fsig)))
        else:
            Flin = prog.F("lut_in")
            ins.append(Instr(f"b{b} sigmoid", S, D, 1, Opnd(RAM, gd, D, 1, 0), imm(1), Opnd(RAM, gs, D, 1, 0),
                             shift=fgd - Flin["f"], clip=clipF(Flin), lut=lut("gate", Flin["lo"])))
        go = buf(f"b{b}_gated", S, D)
        ins.append(Instr(f"b{b} gated", S, D, 1, Opnd(RAM, att, D, 1, 0), Opnd(RAM, gs, D, 1, 0),
                         Opnd(RAM, go, D, 1, 0)))
        wo = buf(f"b{b}_wO", S, D)
        fwo = dense(f"b{b}_wO", f"b{b}_attn0_wO", go, D, S, Fa["f"] + Fsig["f"], wo)
        ins.append(Instr(f"b{b} attn post + res", S, D, 1, Opnd(RAM, wo, D, 1, 0), imm(1), Opnd(RAM, wo, D, 1, 0),
                         shift=fwo - Fa["f"], clip=clipF(Fa)))
        ins.append(Instr(f"b{b} res += attn", S, D, 1, Opnd(RAM, wo, D, 1, 0), imm(1 << (fres - Fa["f"])),
                         Opnd(RAM, res, D, 1, 0), init=Opnd(RAM, res, D, 1, 0)))

        nm = buf(f"b{b}_mlp_norm", S, D)
        rmsnorm(f"b{b}_mlp_norm", f"b{b}_mlp_norm", res, nm)
        w12 = buf(f"b{b}_w12", S, 2 * M)
        f12 = dense(f"b{b}_w12", f"b{b}_w12", nm, D, S, Fa["f"], w12)
        ua = buf(f"b{b}_up_act", S, M)
        fua = act(f"b{b}_up_act", w12 + M, S, M, 2 * M, f12, ua)
        sw = buf(f"b{b}_swiglu", S, M)
        ins.append(Instr(f"b{b} swiglu", S, M, 1, Opnd(RAM, w12, 2 * M, 1, 0), Opnd(RAM, ua, M, 1, 0),
                         Opnd(RAM, sw, M, 1, 0)))
        w3 = buf(f"b{b}_w3", S, D)
        f3 = dense(f"b{b}_w3", f"b{b}_w3", sw, M, S, f12 + fua, w3)
        ins.append(Instr(f"b{b} mlp post", S, D, 1, Opnd(RAM, w3, D, 1, 0), imm(1), Opnd(RAM, w3, D, 1, 0),
                         shift=f3 - Fa["f"], clip=clipF(Fa)))
        ins.append(Instr(f"b{b} res += mlp", S, D, 1, Opnd(RAM, w3, D, 1, 0), imm(1 << (fres - Fa["f"])),
                         Opnd(RAM, res, D, 1, 0), init=Opnd(RAM, res, D, 1, 0)))

    no = buf("out_norm", S, D)
    rmsnorm("out_norm", "out_norm", res, no)
    dt = no + L * D
    hh_ = buf("head_hidden", h, HH)
    fh = dense("head_hidden", "head_hidden", dt, D, h, Fa["f"], hh_)
    ha = buf("head_act", h, HH)
    fha = act("head_act", hh_, h, HH, HH, fh, ha)
    ho = buf("head_output", h, P * Q)
    fho = dense("head_output", "head_output", ha, HH, h, fha, ho)
    hr = buf("head_residual", h, P * Q)
    fhr = dense("head_residual", "head_residual", dt, D, h, Fa["f"], hr)
    out = buf("out", h, P * Q)
    requant_add("head_post", ho, fho, hr, fhr, h, P * Q, out, Fa)

    ram_words, bufs = _pack(ins, bufs, keep=("xin", "out"))
    return Image(ins, np.asarray(wrom.data, np.int64), np.asarray(crom.data, np.int64),
                 np.asarray(lrom.data, np.int64), ram_words, bufs["xin"][0], bufs["out"][0], L * P, h * P * Q,
                 luts=luts, bufs=bufs)


def _pack(ins: list, bufs: dict, keep=()) -> tuple[int, dict]:
    """Re-place RAM buffers so that buffers with disjoint lifetimes share addresses.

    Lifetime = first to last instruction touching the buffer (``keep``: whole program).  Greedy
    first-fit in order of first use; operands are rebased in place.
    """
    spans = {n: (b, b + r * c) for n, (b, r, c) in bufs.items()}

    def owner(addr):
        for n, (lo, hi) in spans.items():
            if lo <= addr < hi:
                return n
        raise ValueError(addr)

    life = {n: [len(ins), -1] for n in bufs}
    for t, i in enumerate(ins):
        for o in (i.a, i.b, i.out, i.init):
            if o is not None and o.space == RAM and (o is i.out or o is i.init or i.K > 0):
                n = owner(o.base)
                life[n] = [min(life[n][0], t), max(life[n][1], t)]
    for n in keep:
        life[n] = [-1, len(ins)]
    placed: list = []                                   # (lo, hi, first, last)
    new = {}
    for n in sorted(bufs, key=lambda n: (life[n][0], -(spans[n][1] - spans[n][0]))):
        size = spans[n][1] - spans[n][0]
        if life[n][1] < 0:
            continue
        cand = 0
        while True:
            clash = [p for p in placed if p[0] < cand + size and cand < p[1] and
                     not (p[3] < life[n][0] or life[n][1] < p[2])]
            if not clash:
                break
            cand = max(p[1] for p in clash)
        placed.append((cand, cand + size, life[n][0], life[n][1]))
        new[n] = cand
    for i in ins:
        for o in (i.a, i.b, i.out, i.init):
            if o is not None and o.space == RAM and (o is i.out or o is i.init or i.K > 0):
                n = owner(o.base)
                o.base = new[n] + (o.base - spans[n][0])
    top = max(p[1] for p in placed)
    return top, {n: (new[n], bufs[n][1], bufs[n][2]) for n in new}


# ----------------------------------------------------------------------------- simulator
def _epilogue(i: Instr, acc, lrom):
    if i.div != 1:
        v = rne_div(acc, i.div)
    elif i.shift > 0:
        v = rne_shift(acc, i.shift) if i.rnd == RNE else rhu_shift(acc, i.shift)
    else:
        v = np.asarray(acc, np.int64) << (-i.shift)
    if i.relu:
        v = np.maximum(v, 0)
    if i.clip is not None:
        v = np.clip(v, i.clip[0], i.clip[1])
    if i.lut is not None:
        v = lrom[i.lut[0] + v - i.lut[1]]
    return v


def simulate(img: Image, x_codes: np.ndarray, ranges: dict | None = None) -> np.ndarray:
    """Run the instructions for a batch of single-variate windows ``(N, L*P)`` -> ``(N, h*P*Q)``.

    Vectorised over the batch; loops follow the instruction semantics exactly.  ``ranges``
    collects per-instruction min/max of operands, accumulators and outputs.
    """
    N = x_codes.shape[0]
    ram = np.zeros((N, img.ram_words), np.int64)
    ram[:, img.xin:img.xin + img.in_len] = x_codes
    mem = {WROM: img.wrom, CROM: img.crom, IMM: img.crom}
    for idx, i in enumerate(img.instrs):
        rr, cc = np.meshgrid(np.arange(i.R), np.arange(i.C), indexing="ij")
        valid = (cc <= rr) if i.tri_c else np.ones_like(rr, bool)
        if i.init is None:
            acc = np.full((N, i.R, i.C), np.iinfo(np.int64).min // 4 if i.op == MAX else 0, np.int64)
        else:
            a0 = i.init.addr(rr, cc, 0)
            acc = (ram[:, a0] if i.init.space == RAM else np.broadcast_to(mem[i.init.space][a0], (N, i.R, i.C))).copy()
            if i.init_neg:
                acc = -acc
        for k in range(i.K):
            kmask = valid & ((k <= rr) if i.tri_k else True)
            if not kmask.any():
                continue
            aa = i.a.addr(rr, cc, k)
            av = ram[:, np.where(kmask, aa, 0)] if i.a.space == RAM else mem[i.a.space][np.where(kmask, aa, 0)]
            if i.op == MAX:
                acc = np.where(kmask, np.maximum(acc, av), acc)
                upd = av
            else:
                ba = i.b.addr(rr, cc, k)
                bv = ram[:, np.where(kmask, ba, 0)] if i.b.space == RAM else mem[i.b.space][np.where(kmask, ba, 0)]
                upd = av * bv
                acc = np.where(kmask, acc + upd, acc)
            if ranges is not None:
                rg = ranges.setdefault(idx, dict(a=[0, 0], b=[0, 0], acc=[0, 0], out=[0, 0]))
                avm = np.asarray(av)[..., kmask] if np.ndim(av) == 3 else np.asarray(av)[kmask]
                rg["a"] = [min(rg["a"][0], int(avm.min())), max(rg["a"][1], int(avm.max()))]
                if i.op == MAC:
                    bvm = np.asarray(bv)[..., kmask] if np.ndim(bv) == 3 else np.asarray(bv)[kmask]
                    rg["b"] = [min(rg["b"][0], int(bvm.min())), max(rg["b"][1], int(bvm.max()))]
                accm = acc[..., kmask]
                rg["acc"] = [min(rg["acc"][0], int(accm.min())), max(rg["acc"][1], int(accm.max()))]
        v = _epilogue(i, acc, img.lrom)
        oa = i.out.addr(rr, cc, 0)
        ram[:, oa[valid]] = v[:, valid]
        if ranges is not None:
            rg = ranges.setdefault(idx, dict(a=[0, 0], b=[0, 0], acc=[0, 0], out=[0, 0]))
            if i.K == 0:
                rg["acc"] = [min(rg["acc"][0], int(acc[:, valid].min())), max(rg["acc"][1], int(acc[:, valid].max()))]
            rg["out"] = [min(rg["out"][0], int(v[:, valid].min())), max(rg["out"][1], int(v[:, valid].max()))]
    return ram[:, img.out:img.out + img.out_len]


def check(prog: Program, x_codes: np.ndarray) -> dict:
    """Simulator == ``program.forward`` for every variate of ``x_codes (N, V, ctx)``; returns ranges."""
    from .program import forward

    img = compile_program(prog)
    N, V, T = x_codes.shape
    ref = forward(prog, x_codes).reshape(N * V, -1)
    ranges: dict = {}
    got = simulate(img, x_codes.reshape(N * V, T), ranges)
    mism = int(np.sum(got != ref))
    bits = lambda lo, hi: int(max(abs(lo), abs(hi) + 1)).bit_length() + 1   # noqa: E731  signed width
    w = {k: max(bits(*rg[k]) for rg in ranges.values()) for k in ("a", "b", "acc", "out")}
    ram_w = max(bits(*rg["out"]) for idx, rg in ranges.items() if img.instrs[idx].out.space == RAM)
    return dict(mismatches=mism, n=int(ref.size), instrs=len(img.instrs), ram_words=img.ram_words,
                wrom=len(img.wrom), crom=len(img.crom), lrom=len(img.lrom), widths=w, ram_width=ram_w,
                wrom_bits=bits(int(img.wrom.min()), int(img.wrom.max())),
                crom_bits=bits(int(img.crom.min()), int(img.crom.max())),
                lrom_bits=bits(int(img.lrom.min()), int(img.lrom.max())),
                macs=int(sum(i.R * max(i.C, 1) * max(i.K, 1) for i in img.instrs)))
