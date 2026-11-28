"""Run forecasts on the GateMate board (firmware of ``timestral_kd deploy``) from the host.

    python -m timestral_kd board kdd/B_d16_L1_b3_ci_p70 --port /dev/ttyACM0 --windows 20
    python -m timestral_kd board kdd/B_d16_L1_b3_ci_p70 --offline --windows 20      # no board: simulator

Encoding (host): causal per-patch scaling + asinh of the raw context (as the model), 12-bit input
codes, one engine run per variate -> ``run <hex>``.  Decoding: the board returns the ``head_post``
codes; the host turns them into monotone quantiles and de-normalises them to raw units.  The
offline path pushes the same hex strings through the instruction-level simulator, so it is the
reference the board must reproduce exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from . import paths


class Codec:
    def __init__(self, codec_npz: str | Path):
        from .hw.fullmodel import program as pg

        self.prog = pg.Program.load(str(codec_npz))
        from .hw.fullmodel.program import HWFormats

        self.in_bits = HWFormats(**self.prog.hw).in_bits
        self.out_bits = int(self.prog.F("a")["bits"])
        d = self.prog.dims
        self.h, self.P, self.Q = d["h"], d["P"], d["Q"]

    def encode(self, context: np.ndarray) -> tuple[str, np.ndarray, np.ndarray]:
        """Raw context ``(V, ctx_steps)`` -> (hex line, loc, scale)."""
        from . import teacher as tch
        from .hw.fullmodel import program as pg

        xs, loc, sc = tch.student_frame(np.asarray(context, np.float32)[None], self.prog.cfg["patch_size"])
        codes = pg.quantize_input(xs, self.prog.hw)[0]                            # (V, ctx)
        digits = (self.in_bits + 3) // 4
        mask = (1 << self.in_bits) - 1
        hx = "".join(f"{int(v) & mask:0{digits}x}" for v in codes.reshape(-1))
        return hx, loc[0], sc[0]

    def codes(self, out_hex: str) -> np.ndarray:
        """``out`` hex of the board -> signed engine output codes ``(V, h*P, Q)``."""
        digits = (self.out_bits + 3) // 4
        vals = [int(out_hex[i:i + digits], 16) & ((1 << self.out_bits) - 1) for i in range(0, len(out_hex), digits)]
        vals = np.array([v - (1 << self.out_bits) if v >> (self.out_bits - 1) else v for v in vals], np.int64)
        return vals.reshape(-1, self.h * self.P, self.Q)

    def decode(self, out_hex: str, loc, scale) -> np.ndarray:
        """``out`` hex of the board -> raw-space quantiles ``(V, h*P, Q)``."""
        from . import teacher as tch
        from .hw.fullmodel import program as pg

        raw = self.codes(out_hex)
        z = pg.quantiles_from_raw(raw, self.prog)
        return tch.from_student_space(z[None], np.asarray(loc)[None], np.asarray(scale)[None])[0]

    def offline(self, hx: str) -> str:
        """The board's reply for ``run <hx>`` from the instruction-level simulator."""
        from .hw.fullmodel import isa

        digits = (self.in_bits + 3) // 4
        words = [int(hx[i:i + digits], 16) for i in range(0, len(hx), digits)]
        words = np.array([w - (1 << self.in_bits) if w >> (self.in_bits - 1) else w for w in words], np.int64)
        img = getattr(self, "_img", None) or isa.compile_program(self.prog)
        self._img = img
        out = isa.simulate(img, words.reshape(-1, img.in_len)).reshape(-1)
        od = (self.out_bits + 3) // 4
        return "".join(f"{int(v) & ((1 << (4 * od)) - 1):0{od}x}" for v in out)


class Board:
    def __init__(self, port: str, baud: int = 115200, timeout: float = 30.0):
        import serial

        self.s = serial.Serial(port, baud, timeout=timeout)
        self.cmd("echo 0")

    def cmd(self, line: str) -> str:
        self.s.write((line + "\n").encode())
        self.s.flush()
        while True:
            r = self.s.readline().decode(errors="replace").strip()
            if not r:
                raise TimeoutError(f"no reply to {line[:20]}...")
            if r.startswith(("out ", "err", "info", "selftest", "bench")) or line.startswith("echo"):
                return r

    def run(self, hx: str) -> tuple[str, int]:
        r = self.cmd("run " + hx)
        if r.startswith("err"):
            raise RuntimeError(r)
        parts = r.split()
        return parts[1], int(parts[3])


def run_windows(codec: Codec, contexts: np.ndarray, board: Board | None = None, check: bool = True) -> dict:
    """Run raw contexts ``(N, V, ctx)`` through the board (``board=None``: the simulator).

    Returns ``pred`` (N, V, h*P, Q) raw-space quantiles, ``cycles`` (N,) engine cycles reported by the
    firmware, and with ``check`` ``mismatches`` (N,): output codes in which the board differs from the
    instruction-level simulator (0 everywhere = bit-exact hardware).
    """
    preds, cycles, bad = [], [], []
    for x in contexts:
        hx, loc, sc = codec.encode(x)
        ref = codec.offline(hx) if (board is None or check) else None
        out_hex, cyc = (ref, 0) if board is None else board.run(hx)
        preds.append(codec.decode(out_hex, loc, sc))
        cycles.append(cyc)
        if ref is not None:
            bad.append(int(np.sum(codec.codes(out_hex) != codec.codes(ref))))
    res = dict(pred=np.stack(preds), cycles=np.array(cycles))
    if bad:
        res["mismatches"] = np.array(bad)
    return res


def plot_forecasts(x, y, pred, knots, names=None, tail: int = 128, labels=None, title: str | None = None,
                   xlabel: str = "steps from forecast start", path=None):
    """Grid of forecasts: rows = windows, columns = variates.

    ``x`` (N, V, ctx) raw context, ``y`` (N, V, H) true future, ``pred`` (N, V, H, Q) quantiles at
    ``knots``.  Draws the last ``tail`` context steps, the truth, the median and the band between
    the outermost knots (e.g. P10-P90).  Saves to ``path`` if given; returns the figure.
    """
    import matplotlib

    if path is not None and matplotlib.get_backend().lower() not in ("agg", "pdf", "svg"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    knots = np.asarray(knots, float)
    lo, hi, md = 0, len(knots) - 1, int(np.argmin(np.abs(knots - 0.5)))
    N, V, H = pred.shape[:3]
    tail = min(tail, x.shape[-1])
    names = list(names) if names is not None else [f"variate {v}" for v in range(V)]
    labels = list(labels) if labels is not None else [f"window {i}" for i in range(N)]
    ctx_t, fut_t = np.arange(-tail, 0), np.arange(H)
    c_obs, c_fc = "0.15", "#1f6fb4"
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "xtick.direction": "out", "ytick.direction": "out"})
    fig, axes = plt.subplots(N, V, figsize=(2.4 * V + 0.6, 1.45 * N + 0.7), squeeze=False, sharex=True)
    for i in range(N):
        for v in range(V):
            ax = axes[i, v]
            ax.plot(ctx_t, x[i, v, -tail:], color=c_obs, lw=0.8)
            ax.plot(fut_t, y[i, v], color=c_obs, lw=0.8, ls="--")
            ax.fill_between(fut_t, pred[i, v, :, lo], pred[i, v, :, hi], color=c_fc, alpha=0.25, lw=0)
            ax.plot(fut_t, pred[i, v, :, md], color=c_fc, lw=1.1)
            ax.axvline(0, color="0.6", lw=0.6, ls=":")
            ax.margins(x=0.01, y=0.06)
            if i == 0:
                ax.set_title(names[v], loc="left")
            if v == 0:
                ax.set_ylabel(labels[i])
            if i == N - 1:
                ax.set_xlabel(xlabel)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [Line2D([], [], color=c_obs, lw=0.8, label="observed context"),
               Line2D([], [], color=c_obs, lw=0.8, ls="--", label="observed future"),
               Line2D([], [], color=c_fc, lw=1.1, label=f"forecast median (P{100 * knots[md]:.0f})"),
               Patch(color=c_fc, alpha=0.25, lw=0, label=f"P{100 * knots[lo]:.0f}-P{100 * knots[hi]:.0f} band")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=8)
    fig.tight_layout(rect=(0, 0.06 if N > 2 else 0.1, 1, 0.97 if title else 1))
    if path is not None:
        fig.savefig(path, dpi=200)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="<dataset>/<student> or a student directory (its deploy directory holds codec.npz)")
    ap.add_argument("--port", default=None)
    ap.add_argument("--offline", action="store_true", help="instruction-level simulator instead of a board")
    ap.add_argument("--windows", type=int, default=10, help="number of test windows")
    ap.add_argument("--start", type=int, default=0, help="first test window")
    ap.add_argument("--stride", type=int, default=1, help="step between windows (e.g. 24 = one per day)")
    ap.add_argument("--no-check", action="store_true", help="skip the board-vs-simulator comparison")
    ap.add_argument("--plot", type=Path, default=None, help="save a forecast grid (PNG/PDF) of these windows")
    ap.add_argument("--plot-max", type=int, default=6, help="windows drawn in the plot")
    ap.add_argument("--tail", type=int, default=128, help="context steps drawn before the forecast")
    ap.add_argument("--deploy", type=Path, default=None, help="default runs/deploy/<run>")
    a = ap.parse_args(argv)
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    from . import data as dd
    from . import student as stu
    from . import train
    from .deploy import run_name

    dep = a.deploy or paths.RUNS / "deploy" / run_name(a.run)
    codec = Codec(dep / "codec.npz")
    cfg = stu.StudentConfig.from_dict(codec.prog.cfg)
    run_dir = paths.student_dir(a.run)
    data = Path(json.loads((run_dir / "config.json").read_text())["data"])
    b, info = train.load_bundle(data)
    te = train.prepare_split(b, "test", cfg)
    idx = np.arange(a.start, len(te["x"]), max(1, a.stride))[:a.windows]
    if len(idx) == 0:
        raise SystemExit(f"no test windows from --start {a.start} (test split has {len(te['x'])})")
    board = None if a.offline else Board(a.port)
    t0 = time.time()
    r = run_windows(codec, te["x"][idx], board, check=not a.no_check)
    pred, y, x = r["pred"], te["y"][idx], te["x"][idx]
    m = dd.summarize(pred, y, x, cfg.knots, info["season"], info.get("variates"))
    res = dict(windows=len(idx), first=int(idx[0]), stride=a.stride, source="simulator" if board is None else a.port,
               mase=m["mase"], wql=m["wql"], coverage_80=m["coverage_80"],
               engine_cycles=int(np.mean(r["cycles"])) if board else None, seconds=round(time.time() - t0, 1))
    if board is not None and "mismatches" in r:
        res["mismatched_codes"] = int(r["mismatches"].sum())
        res["bit_exact_windows"] = f"{int((r['mismatches'] == 0).sum())}/{len(idx)}"
    if a.plot is not None:
        k = min(a.plot_max, len(idx))
        src = "simulator" if board is None else "GateMate"
        unit = "hours" if info.get("dataset") == "kdd" else "steps"
        plot_forecasts(x[:k], y[:k], pred[:k], cfg.knots, names=info.get("variates"), tail=a.tail,
                       labels=[f"window {i}" for i in idx[:k]], path=a.plot, xlabel=f"{unit} from forecast start",
                       title=f"{run_name(a.run)} on the {src}: MASE {m['mase']:.3f}, "
                             f"80% coverage {m['coverage_80']:.2f} over {len(idx)} test windows")
        res["plot"] = str(a.plot)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
