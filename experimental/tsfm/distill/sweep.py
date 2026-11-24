"""Run a list of student configs (in parallel subprocesses) and collect the results.

Example::

    python -m distill.sweep --sweep distill/configs/sweep_v1.json --data runs/data/kdd.npz \
        --out runs/students/kdd --jobs 4

The sweep file is ``{"defaults": {...}, "runs": [{"student": {...}, "train": {...}, "alpha": ...}, ...]}``.
Results: ``<out>/<name>/metrics.json`` per run and ``<out>/student_sweep.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _merge(a, b):
    out = json.loads(json.dumps(a))
    for k, v in b.items():
        out[k] = _merge(out.get(k, {}), v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def flatten(m: dict) -> dict:
    row = {k: m.get(k) for k in ("name", "dataset", "V", "alpha", "variate_every", "deployment", "preset", "n_train")}
    ops = m["ops"]
    row.update(params=ops["params"], weight_macs=ops["weight_macs"], act_macs=ops["act_macs"])
    for tag in ("teacher", "float", "qat_ptq", "qat"):
        if tag in m:
            row[f"{tag}_mase"] = m[tag]["mase"]
            row[f"{tag}_wql"] = m[tag]["wql"]
            row[f"{tag}_cov80"] = m[tag]["coverage_80"]
    row["snaive_mase"] = m["seasonal_naive_mase"]
    c = m["cost"]
    row.update(weight_bits=c["weight_bits"], A_bram_util=c["A"]["bram_util"], A_cpe=c["A"]["cpe"],
               A_latency_us=c["A"]["latency_us"], A_fits=c["A"]["fits"], B_cpe_r1=c["B"]["cpe"],
               B_min_reuse=c["B"]["min_reuse_to_fit"], B_latency_us=c["B"]["latency_us_at_min_reuse"],
               B_csd_digits=c["B"]["csd_digits"], B_fits=c["B"]["fits_with_reuse"], fits=c["fits"])
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2, help="TF threads per job")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args(argv)

    sw = json.loads(args.sweep.read_text())
    runs = [_merge(sw.get("defaults", {}), r) for r in sw["runs"]]
    if args.only:
        runs = [r for r in runs if r["student"]["name"] in args.only]
    args.out.mkdir(parents=True, exist_ok=True)

    def job(r):
        name = r["student"]["name"]
        d = args.out / name
        if (d / "metrics.json").exists():
            return name, "cached"
        d.mkdir(parents=True, exist_ok=True)
        cfgp = d / "run_config.json"
        cfgp.write_text(json.dumps({"student": r["student"], "train": r.get("train", {})}, indent=1))
        cmd = [sys.executable, "-m", "distill.train", "--config", str(cfgp), "--data", str(args.data), "--out", str(d)]
        if "alpha" in r:
            cmd += ["--alpha", str(r["alpha"])]
        env = {**os.environ, "OMP_NUM_THREADS": str(args.threads), "TF_NUM_INTRAOP_THREADS": str(args.threads),
               "TF_NUM_INTEROP_THREADS": "1", "TF_CPP_MIN_LOG_LEVEL": "3"}
        with open(d / "train.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        return name, f"rc={rc}"

    if not args.collect_only:
        with ThreadPoolExecutor(args.jobs) as ex:
            for name, st in ex.map(job, runs):
                print(f"{name}: {st}", flush=True)

    rows = []
    for r in runs:
        mp = args.out / r["student"]["name"] / "metrics.json"
        if mp.exists():
            rows.append(flatten(json.loads(mp.read_text())))
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with open(args.out / "student_sweep.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.out / 'student_sweep.csv'}")


if __name__ == "__main__":
    main()
