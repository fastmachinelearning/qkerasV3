"""Pipeline CLI: Timestral-5M teacher -> quantised student -> GateMate place & route.

    python -m timestral_kd data   kdd                                  # windows + teacher labels
    python -m timestral_kd train  B_d12_L1_b3_p70_ci --dataset kdd     # float distill -> QAT -> prune
    python -m timestral_kd export kdd/B_d12_L1_b3_p70_ci               # da4ml Verilog per dense layer
    python -m timestral_kd pnr    kdd/B_d12_L1_b3_p70_ci --jobs 7      # yosys + nextpnr on CCGM1A1
    python -m timestral_kd report kdd/B_d12_L1_b3_p70_ci               # per-layer table + fit summary

    python -m timestral_kd all    B_d12_L1_b3_p70_ci --dataset kdd     # all of the above
    python -m timestral_kd sweep  prune_v1 --dataset kdd --jobs 4      # many students in parallel
    python -m timestral_kd plots  kdd                                  # Pareto / ablation figures
    python -m timestral_kd verify [--rtl]                              # bit-exact fixed-point (+ Verilator) checks

    python -m timestral_kd deploy kdd/B_d16_L1_b3_ci_p70 [--verify 6]  # whole student -> GateMate engine,
                                                                       # Kolibri Studio LiteX project
    python -m timestral_kd board  kdd/B_d16_L1_b3_ci_p70 --port /dev/ttyACM0   # forecasts on the board

Configs are looked up by name in ``configs/students`` / ``configs/sweeps`` (or given as a path).
Runs are named ``<dataset>/<student>``; all outputs go under ``runs/`` (see ``timestral_kd.paths``).
Options not listed here are passed on to the underlying module (``python -m timestral_kd.train -h`` ...).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import paths


# ----------------------------------------------------------------------------- helpers
def find_config(name: str, kind: str) -> Path:
    """``name`` as a path, or ``configs/<kind>/<name>[.json]``."""
    for p in (Path(name), paths.CONFIGS / kind / name, paths.CONFIGS / kind / f"{name}.json"):
        if p.is_file():
            return p
    choices = sorted(q.stem for q in (paths.CONFIGS / kind).glob("*.json"))
    raise SystemExit(f"config {name!r} not found; available in configs/{kind}: {', '.join(choices)}")


def need_bundle(dataset: str) -> Path:
    b = paths.bundle(dataset)
    if not b.exists():
        raise SystemExit(f"{b} missing - run: python -m timestral_kd data {dataset}")
    return b


def kolibri_src(given: str | None) -> str | None:
    if given:
        return given
    if os.environ.get("KOLIBRI_FPGA_SRC"):
        return os.environ["KOLIBRI_FPGA_SRC"]
    guess = paths.REPO.parent / "logicnet-fpga" / "src"        # sibling checkout of the Kolibri backend
    return str(guess) if guess.is_dir() else None


# ----------------------------------------------------------------------------- argv builders
def data_argv(a, extra) -> list[str]:
    known = {"kdd", "synthetic"}
    ds = a.dataset if a.dataset in known else "csv"
    argv = ["--dataset", ds]
    if ds == "csv":
        argv += ["--name", a.dataset]
    return argv + extra


def train_argv(a, extra) -> tuple[list[str], Path]:
    cfg = find_config(a.config, "students")
    name = json.loads(cfg.read_text())["student"]["name"]
    out = paths.student_dir(f"{a.dataset}/{name}")
    return ["--config", str(cfg), "--data", str(need_bundle(a.dataset)), "--out", str(out)] + extra, out


def sweep_argv(a, extra) -> list[str]:
    sw = find_config(a.sweep, "sweeps")
    return ["--sweep", str(sw), "--data", str(need_bundle(a.dataset)),
            "--out", str(paths.RUNS / "students" / a.dataset)] + extra


def export_argv(a, extra) -> list[str]:
    run = paths.student_dir(a.run)
    if not (run / "qat.weights.h5").exists():
        raise SystemExit(f"{run}/qat.weights.h5 missing - train the student first")
    argv = ["export", "--run", str(run), "--out", str(paths.rtl_dir(a.run, a.digit_bits))]
    return argv + (["--digit-bits", str(a.digit_bits)] if a.digit_bits else []) + extra


def pnr_argv(a, extra) -> list[str]:
    out = paths.rtl_dir(a.run, a.digit_bits)
    if not (out / "export.json").exists():
        raise SystemExit(f"{out}/export.json missing - run: python -m timestral_kd export {a.run}")
    src = kolibri_src(a.kolibri_src)
    return ["pnr", "--out", str(out), "--jobs", str(a.jobs)] + (["--kolibri-src", src] if src else []) + extra


# ----------------------------------------------------------------------------- commands
def cmd_data(a, extra):
    from . import prepare
    prepare.main(data_argv(a, extra))


def cmd_train(a, extra):
    from . import train
    argv, out = train_argv(a, extra)
    train.main(argv)
    print(f"student: {out}")


def cmd_sweep(a, extra):
    from . import sweep
    sweep.main(sweep_argv(a, extra))


def cmd_export(a, extra):
    from .hw import dense
    dense.main(export_argv(a, extra))


def cmd_pnr(a, extra):
    from .hw import dense
    dense.main(pnr_argv(a, extra))


def cmd_report(a, extra):
    from .hw import dense
    out = str(paths.rtl_dir(a.run, a.digit_bits))
    print(dense.collect(out).to_string(index=False))
    dense.deploy(out)


def cmd_plots(a, extra):
    from . import plots
    plots.main(["--sweeps", *[str(paths.RUNS / "students" / d) for d in a.datasets],
                "--out", str(paths.RUNS / "figures")] + extra)


def cmd_verify(a, extra):
    from .hw.fullmodel import fixedpoint
    fixedpoint._validate()
    if a.rtl:
        from .hw.fullmodel import blocks, sequencer
        blocks.check_all()
        sequencer.check_all()


def cmd_deploy(a, extra):
    from . import deploy
    deploy.main([a.run] + extra)


def cmd_board(a, extra):
    from . import board
    board.main([a.run] + extra)


def cmd_all(a, extra):
    if not paths.bundle(a.dataset).exists():
        cmd_data(argparse.Namespace(dataset=a.dataset), [])
    argv, out = train_argv(a, [])
    from . import train
    train.main(argv)
    run = f"{a.dataset}/{out.name}"
    ns = argparse.Namespace(run=run, digit_bits=a.digit_bits, jobs=a.jobs, kolibri_src=a.kolibri_src)
    cmd_export(ns, [])
    cmd_pnr(ns, [])
    cmd_report(ns, [])


# ----------------------------------------------------------------------------- parser
def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m timestral_kd", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("data", help="build runs/data/<dataset>.npz (windows + teacher labels)")
    p.add_argument("dataset", help="kdd | synthetic | <name> for a process CSV (then pass --csv FILE --columns ...)")
    p.set_defaults(fn=cmd_data)

    p = sub.add_parser("train", help="train one student -> runs/students/<dataset>/<name>")
    p.add_argument("config", help="name in configs/students or a path")
    p.add_argument("--dataset", required=True)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("sweep", help="train every student of a sweep file in parallel")
    p.add_argument("sweep", help="name in configs/sweeps or a path")
    p.add_argument("--dataset", required=True)
    p.set_defaults(fn=cmd_sweep)

    for name, fn, hlp in (("export", cmd_export, "da4ml Verilog of every dense layer -> runs/rtl/<run>"),
                          ("pnr", cmd_pnr, "GateMate place & route of every exported layer"),
                          ("report", cmd_report, "per-layer P&R table + single-bitstream fit summary")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("run", help="<dataset>/<student>, e.g. kdd/B_d12_L1_b3_p70_ci")
        p.add_argument("--digit-bits", type=int, default=None, help="digit-serial export (default fully parallel)")
        if name == "pnr":
            p.add_argument("--jobs", type=int, default=4)
            p.add_argument("--kolibri-src", default=None,
                           help="logicnet-fpga/src (default $KOLIBRI_FPGA_SRC or sibling)")
        p.set_defaults(fn=fn)

    p = sub.add_parser("all", help="data (if missing) -> train -> export -> pnr -> report")
    p.add_argument("config")
    p.add_argument("--dataset", required=True)
    p.add_argument("--digit-bits", type=int, default=None)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--kolibri-src", default=None)
    p.set_defaults(fn=cmd_all)

    p = sub.add_parser("plots", help="figures of the sweeps in runs/students/<dataset> -> runs/figures")
    p.add_argument("datasets", nargs="+")
    p.set_defaults(fn=cmd_plots)

    p = sub.add_parser("deploy", help="whole student as one GateMate engine -> Kolibri Studio LiteX project")
    p.add_argument("run", help="<dataset>/<student>")
    p.set_defaults(fn=cmd_deploy)

    p = sub.add_parser("board", help="forecasts on the board (or --offline through the engine simulator)")
    p.add_argument("run", help="<dataset>/<student>")
    p.set_defaults(fn=cmd_board)

    p = sub.add_parser("verify", help="bit-exact fixed-point references vs qkeras (--rtl: + Verilator checks)")
    p.add_argument("--rtl", action="store_true")
    p.set_defaults(fn=cmd_verify)
    return ap


def main(argv=None):
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    a, extra = parser().parse_known_args(argv)
    a.fn(a, extra)


if __name__ == "__main__":
    main(sys.argv[1:])
