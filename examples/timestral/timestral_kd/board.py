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

    def decode(self, out_hex: str, loc, scale) -> np.ndarray:
        """``out`` hex of the board -> raw-space quantiles ``(V, h*P, Q)``."""
        from . import teacher as tch
        from .hw.fullmodel import program as pg

        digits = (self.out_bits + 3) // 4
        vals = [int(out_hex[i:i + digits], 16) & ((1 << self.out_bits) - 1) for i in range(0, len(out_hex), digits)]
        vals = np.array([v - (1 << self.out_bits) if v >> (self.out_bits - 1) else v for v in vals], np.int64)
        raw = vals.reshape(-1, self.h * self.P, self.Q)
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="<dataset>/<student> or a student directory (its deploy directory holds codec.npz)")
    ap.add_argument("--port", default=None)
    ap.add_argument("--offline", action="store_true", help="instruction-level simulator instead of a board")
    ap.add_argument("--windows", type=int, default=10)
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
    n = min(a.windows, len(te["x"]))
    board = None if a.offline else Board(a.port)
    preds, cycles, t0 = [], [], time.time()
    for i in range(n):
        hx, loc, sc = codec.encode(te["x"][i])
        if board is None:
            out_hex, cyc = codec.offline(hx), 0
        else:
            out_hex, cyc = board.run(hx)
        preds.append(codec.decode(out_hex, loc, sc))
        cycles.append(cyc)
    pred = np.stack(preds)
    m = dd.summarize(pred, te["y"][:n], te["x"][:n], cfg.knots, info["season"], info.get("variates"))
    res = dict(windows=n, source="simulator" if board is None else a.port, mase=m["mase"], wql=m["wql"],
               coverage_80=m["coverage_80"], engine_cycles=int(np.mean(cycles)) if board else None,
               seconds=round(time.time() - t0, 1))
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
