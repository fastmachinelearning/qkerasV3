"""Deploy a trained student to the GateMate as a Kolibri Studio LiteX project.

    python -m timestral_kd deploy kdd/B_d16_L1_b3_ci_p70                 # -> runs/deploy/kdd/B_d16_L1_b3_ci_p70
    python -m timestral_kd deploy kdd/B_d16_L1_b3_ci_p70 --verify 6      # + Verilator check of the self-test vectors
    python -m timestral_kd deploy kdd/B_d16_L1_b3_ci_p70 --studio-url studio.example.org --build

The whole student runs in hardware: ``hw/fullmodel`` lowers it to an integer program, compiles
it for the sequential tensor engine and writes the engine's Verilog.  The project directory is
what the Kolibri Studio LiteX IDE expects:

    nn.json               the engine ("interface": "memory"; one run per variate)
    rtl/engine.v          generated engine; rtl/{irom,wrom,crom,lrom}.mem its memories
    firmware/selftest.h   reference vectors (input codes and expected engine outputs)
    codec.npz             what the host needs to encode windows and decode forecasts (board.py)
    README.md             build / program / run steps

With the Kolibri Studio template available (``kolibri_litex`` installed, ``--template`` or a sibling
``kolibri-studio`` checkout) the full IDE project (``soc.py``, firmware) is written too;
``--studio-url`` creates a project in a running Studio and uploads the files instead.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from . import paths

SOC_PY = '''#!/usr/bin/env python3
"""LiteX SoC for the Olimex GateMate A1-EVB with the Timestral engine in nn.json (``python soc.py --build``).

Written by ``timestral_kd deploy``: the engine SoC routes at ~15.3 MHz, so the default clock is
{mhz:g} MHz, and nextpnr seed {seed} (seed 1 can leave a wire overused).  Command-line options win.
"""

import sys

from litex.soc.integration import builder as _builder

from kolibri_litex.soc import KolibriSoC, build_main

NEXTPNR_SEED = {seed}
SYS_CLK_FREQ = "{clk:g}"

_build = _builder.Builder.build


def _seeded_build(self, *args, **kwargs):
    kwargs.setdefault("seed", NEXTPNR_SEED)
    return _build(self, *args, **kwargs)


_builder.Builder.build = _seeded_build


class SoC(KolibriSoC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


if __name__ == "__main__":
    if not any(a.startswith("--sys-clk-freq") for a in sys.argv):
        sys.argv += ["--sys-clk-freq", SYS_CLK_FREQ]
    build_main(SoC)
'''

README = """# Timestral student `{name}` on the GateMate (Kolibri Studio LiteX project)

The whole student (embedding, attention, MLP, quantile head) runs in `rtl/engine.v`, a sequential
tensor engine ({n_instr} instructions, {cycles} cycles per variate, about {ms:.0f} ms per forecast of
{V} variates at {mhz:g} MHz).  The VexRiscv firmware streams one variate at a time through it.

* Kolibri Studio: open this project in the LiteX IDE, **Build gateware**, **Build firmware**,
  **Program FPGA**, **Upload firmware**, then type `selftest` in the terminal (expects `selftest ok`).
* Before the board: `python -m kolibri_litex.sim .` simulates the whole SoC (Verilator) and runs the
  same `selftest` (expects `selftest ok` after a ~20 min Verilator build, ~30 s run).
* Command line (LiteX 2026.08, OSS CAD Suite, RISC-V gcc): `python soc.py --build`,
  `make -C firmware`, `openFPGALoader -c dirtyJtag build/gateware/*.bit`,
  `litex_term --kernel firmware/firmware.bin /dev/ttyACM0`.
* Forecasts from the host: `python -m timestral_kd board {run} --port /dev/ttyACM0`
  (encodes windows with `codec.npz`, sends `run <hex>`, decodes the quantiles).

Firmware commands: `info`, `run <hex>` ({V} x {in_len} words, 3 hex digits each), `bench <n>`,
`selftest`, `echo 0|1`, `reboot`.
"""


def _template(given=None) -> Path | None:
    """Kolibri Studio LiteX project template: kolibri_litex package, ``given``, or a sibling checkout."""
    try:
        from importlib import resources

        t = Path(str(resources.files("kolibri_litex") / "templates" / "project"))
        if t.is_dir():
            return t
    except ImportError:
        pass
    for t in ([Path(given)] if given else []) + [
            paths.REPO.parent / "kolibri-studio" / "packages" / "kolibri-litex" / "src" / "kolibri_litex"
            / "templates" / "project"]:
        if (t / "soc.py").exists():
            return t
    return None


def build(run: str, out: Path, n_selftest: int = 2, verify: int = 0, hw=None, template=None,
          seed: int = 2, sys_clk: float = 14e6) -> dict:
    """Write the project; returns a summary dict."""
    from . import student as stu
    from . import train
    from .hw.fullmodel import engine_rtl, isa
    from .hw.fullmodel import program as pg

    run_dir = paths.student_dir(run)
    prog, model = pg.extract(str(run_dir), hw)
    img = isa.compile_program(prog)
    cfg = stu.StudentConfig.from_dict(prog.cfg)
    out = Path(out)
    (out / "rtl").mkdir(parents=True, exist_ok=True)
    (out / "firmware").mkdir(parents=True, exist_ok=True)
    p = engine_rtl.write(img, str(out / "rtl"))
    (out / "rtl" / "engine.json").unlink(missing_ok=True)

    # self-test vectors from the test split (falls back to random scaled windows)
    meta = json.loads((run_dir / "config.json").read_text())
    data = Path(meta.get("data", ""))
    if not data.exists():
        data = paths.bundle(run_name(run).split("/")[0])
    if data.exists():
        b, _ = train.load_bundle(data)
        xs = train.prepare_split(b, "test", cfg)["xs"][:n_selftest]
    else:
        xs = np.arcsinh(np.random.default_rng(0).normal(size=(n_selftest, 1, cfg.ctx_steps)))
    codes = pg.quantize_input(xs, prog.hw)                         # (N, V, ctx)
    N, V, T = codes.shape
    ref = isa.simulate(img, codes.reshape(N * V, T)).reshape(N, V * img.out_len)
    with open(out / "firmware" / "selftest.h", "w") as f:
        f.write(f"/* Generated by timestral_kd.deploy: {N} test windows of {run}. */\n#include <stdint.h>\n")
        f.write(f"#define SELFTEST_N {N}\n")
        f.write(f"static const int16_t selftest_in[{N}][{V * T}] = {{\n")
        for row in codes.reshape(N, -1):
            f.write("  {" + ",".join(str(int(v)) for v in row) + "},\n")
        f.write("};\n")
        f.write(f"static const int16_t selftest_out[{N}][{V * img.out_len}] = {{\n")
        for row in ref:
            f.write("  {" + ",".join(str(int(v)) for v in row) + "},\n")
        f.write("};\n")

    from .hw.fullmodel.program import HWFormats
    in_bits = HWFormats(**prog.hw).in_bits
    out_bits = int(prog.F("a")["bits"])
    nn = dict(top="timestral_engine", sources=["rtl/engine.v"],
              memfiles=[f"rtl/{m}.mem" for m in ("irom", "wrom", "crom", "lrom")],
              in_port="host_wdata", in_bits=V * img.in_len * in_bits, out_port="host_rdata",
              out_bits=V * img.out_len * out_bits, clk_port="clk", cycles=0, family="rtl", classes=[],
              interface="memory",
              # the engine needs ~25 of the 64 block-RAM halves: cache-less CPU so the SoC fits next to it
              soc=dict(cpu_variant="minimal"),
              mem=dict(rst_port="rst", start_port="start", busy_port="busy", done_port="done", we_port="host_we",
                       addr_port="host_addr", wdata_port="host_wdata", rdata_port="host_rdata",
                       addr_bits=p["RAM_AW"], in_base=img.xin, in_len=img.in_len, in_word_bits=in_bits,
                       out_base=img.out, out_len=img.out_len, out_word_bits=out_bits, runs=V))
    (out / "nn.json").write_text(json.dumps(nn, indent=1) + "\n")
    prog.save(str(out / "codec.npz"))

    summary = dict(run=run, out=str(out), instructions=len(img.instrs), ram_words=img.ram_words,
                   variates=V, in_len=img.in_len, out_len=img.out_len, selftest=N)
    if verify:
        from .hw.fullmodel import engine_verify
        r = engine_verify.verify(img, codes.reshape(N * V, T)[:verify], str(out / "verify"), keep=False)
        summary.update(verify_runs=r["n_runs"], verify_mismatches=r["mismatches"], cycles=r["cycles_max"])
    cyc = summary.get("cycles", 404_000)
    (out / "README.md").write_text(README.format(name=cfg.name, n_instr=len(img.instrs), cycles=cyc,
                                                 ms=V * cyc / (sys_clk / 1e3), mhz=sys_clk / 1e6, V=V, run=run,
                                                 in_len=img.in_len))
    (out / "soc.py").write_text(SOC_PY.format(seed=seed, clk=sys_clk, mhz=sys_clk / 1e6))
    summary.update(nextpnr_seed=seed, sys_clk_hz=sys_clk)
    tmpl = _template(template)
    summary["litex_project"] = tmpl is not None
    if tmpl is not None:                   # full IDE project: soc.py + firmware template (ours win)
        for f in tmpl.rglob("*"):
            dst = out / f.relative_to(tmpl)
            if f.is_file() and "__pycache__" not in f.parts and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(f.read_bytes())
        summary["template"] = str(tmpl)
    return summary


def run_name(run: str) -> str:
    """``kdd/B_d16`` or a student directory path -> ``kdd/B_d16`` (names output folders and projects)."""
    p = Path(run)
    if not p.is_dir():
        return run
    p = p.resolve()
    students = (paths.RUNS / "students").resolve()
    return str(p.relative_to(students)) if p.is_relative_to(students) else f"{p.parent.name}/{p.name}"


def studio_url(url: str) -> str:
    url = url.strip().rstrip("/")
    return url if "://" in url else "https://" + url


def upload(out: Path, url: str, name: str) -> dict:
    """Create a Kolibri Studio IDE project (Studio's LiteX template) and upload the deploy files."""
    import requests

    out, url = Path(out), studio_url(url)
    meta = requests.post(f"{url}/api/projects", json={"name": name[:80]}, timeout=60)
    meta.raise_for_status()
    pid = meta.json()["id"]
    files = [out / "nn.json", out / "soc.py", out / "firmware" / "selftest.h", out / "README.md"]
    files += sorted((out / "rtl").glob("*"))
    for f in files:
        rel = f.relative_to(out).as_posix()
        r = requests.put(f"{url}/api/projects/{pid}/files/{rel}", json={"text": f.read_text()}, timeout=120)
        if not r.ok:
            raise RuntimeError(f"upload of {rel} failed: {r.status_code} {r.text[:200]}")
    return dict(project=pid, files=len(files), ide=f"{url}/ide.html#project={pid}")


_LOG_KEYS = ("Max frequency", "RAM_HALF", "CPE_LT", "ERROR", "rror:", "build finished", "build failed",
             "firmware.bin:")


def studio_build(out: Path, url: str, pid: str, kinds=("gateware", "firmware"), timeout: float = 7200,
                 poll: float = 10.0) -> dict:
    """Build in the Studio (its native yosys / nextpnr / gcc), echo the key log lines, fetch the outputs."""
    import requests

    url, res = studio_url(url), {}
    for kind in kinds:
        r = requests.post(f"{url}/api/projects/{pid}/build", json={"kind": kind}, timeout=60)
        if not r.ok:
            raise RuntimeError(f"{kind} build not started: {r.status_code} {r.text[:200]}")
        off, t0, st = 0, time.time(), {}
        print(f"[studio] {kind} build started", flush=True)
        while time.time() - t0 < timeout:
            time.sleep(poll)
            b = requests.get(f"{url}/api/projects/{pid}/build", params={"offset": off}, timeout=60).json()
            off, st = b["offset"], b["status"]
            for line in b["log"].splitlines():
                if any(k in line for k in _LOG_KEYS):
                    print(f"[studio] {line.strip()}", flush=True)
            if st.get("state") in ("finished", "failed", "cancelled"):
                break
        res[kind] = dict(state=st.get("state"), seconds=round(time.time() - t0), error=st.get("error"))
        if st.get("state") != "finished":
            res[kind]["log"] = f"{url}/api/projects/{pid}/build"
            return res
    dst = Path(out) / "studio"
    dst.mkdir(parents=True, exist_ok=True)
    arts = requests.get(f"{url}/api/projects/{pid}", timeout=60).json().get("artifacts", {})
    for key, a in arts.items():
        r = requests.get(f"{url}/api/projects/{pid}/artifacts/{key}", timeout=300)
        if r.ok:
            (dst / a["name"]).write_bytes(r.content)
            res.setdefault("downloaded", []).append(str(dst / a["name"]))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="<dataset>/<student> or a student directory")
    ap.add_argument("--out", type=Path, default=None, help="default runs/deploy/<dataset>/<student>")
    ap.add_argument("--selftest", type=int, default=2, help="test windows baked into the firmware")
    ap.add_argument("--verify", type=int, default=0, help="Verilator-check this many engine runs")
    ap.add_argument("--studio-url", default=None, help="also create the project in a running Kolibri Studio")
    ap.add_argument("--build", action="store_true",
                    help="with --studio-url: build gateware + firmware in the Studio and download the outputs")
    ap.add_argument("--template", default=None, help="Kolibri Studio LiteX project template directory")
    ap.add_argument("--seed", type=int, default=2, help="nextpnr seed written into the project soc.py")
    ap.add_argument("--sys-clk-mhz", type=float, default=14.0, help="default SoC clock of the project soc.py")
    a = ap.parse_args(argv)
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    name = run_name(a.run)
    out = a.out or paths.RUNS / "deploy" / name
    s = build(a.run, out, a.selftest, a.verify, template=a.template, seed=a.seed,
              sys_clk=a.sys_clk_mhz * 1e6)
    if a.studio_url:
        s["studio"] = upload(out, a.studio_url, f"timestral {name}")
        print(f"[studio] project {s['studio']['project']}: {s['studio']['ide']}", flush=True)
        if a.build:
            s["studio"]["build"] = studio_build(out, a.studio_url, s["studio"]["project"])
    print(json.dumps(s, indent=1, default=str))
    return s


if __name__ == "__main__":
    main()
