"""Build dataset bundles (windows + teacher soft labels + teacher scores).

Example::

    python -m timestral_kd.prepare --dataset kdd --kdd-parquet data/monash_kdd_cup_2018.parquet \
        --ckpt ts/Timestral-5M --out runs/data
    python -m timestral_kd.prepare --dataset synthetic --out runs/data
    python -m timestral_kd.prepare --dataset csv --csv process_data.csv --columns NOx Sauerstoff ... --out runs/data

A bundle ``<out>/<dataset>.npz`` holds ``{split}_x`` (N, V, teacher_ctx), ``{split}_y`` (N, V, H)
and ``{split}_tq`` (N, V, H, Qs) teacher quantiles (raw space) for split in train/val/test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import data as dd
from . import teacher as tch

DATASETS = {
    # name: (season, default stride train/eval, variate names)
    "synthetic": (1, 48, None),
    "kdd": (24, 8, list(dd.KDD_DEFAULT)),
    "csv": (1, 16, None),
}


def build_series(args):
    if args.dataset == "synthetic":
        s = dd.synthetic_process(args.n_series, args.length, n_drivers=2, n_targets=3, seed=args.seed)
        return s, ["driver0", "driver1", "target0", "target1", "target2"]
    if args.dataset == "kdd":
        return dd.load_kdd_multivariate(args.kdd_parquet), list(dd.KDD_DEFAULT)
    s = dd.load_process_csv(args.csv, args.columns)
    return s, args.columns or [f"v{i}" for i in range(s[0].shape[0])]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/data"))
    ap.add_argument("--ckpt", type=Path, default=Path("ts/Timestral-5M"))
    ap.add_argument("--kdd-parquet", type=Path, default=Path("data/monash_kdd_cup_2018.parquet"))
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--columns", nargs="*", default=None)
    ap.add_argument("--teacher-ctx", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--stride", type=int, default=None, help="train stride (eval uses 2x)")
    ap.add_argument("--max-train", type=int, default=8000)
    ap.add_argument("--n-series", type=int, default=80)
    ap.add_argument("--length", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args(argv)

    season, stride, _ = DATASETS[args.dataset]
    stride = args.stride or stride
    series, names = build_series(args)
    parts = dd.chronological_split(series, (0.7, 0.1, 0.2))
    bundle, info = {}, {"dataset": args.dataset, "variates": names, "season": season,
                        "teacher_ctx": args.teacher_ctx, "horizon": args.horizon,
                        "knots": list(tch.DEFAULT_STUDENT_KNOTS)}
    for split, p, st, cap in [("train", parts[0], stride, args.max_train), ("val", parts[1], stride, 2000),
                              ("test", parts[2], 2 * stride, 4000)]:
        w = dd.make_windows(p, args.teacher_ctx, args.horizon, st, max_windows=cap, seed=args.seed)
        t0 = time.time()
        tq = tch.cached_teacher_labels(args.out / "cache", {"dataset": args.dataset, "split": split, "seed": args.seed},
                                       args.ckpt, w.x, args.horizon, batch=args.batch)
        info[f"{split}_windows"] = len(w)
        info[f"{split}_teacher_seconds"] = round(time.time() - t0, 1)
        bundle.update({f"{split}_x": w.x, f"{split}_y": w.y, f"{split}_tq": tq.astype(np.float16)})
        print(f"{split}: {len(w)} windows, teacher {info[f'{split}_teacher_seconds']} s", flush=True)

    x, y, q = bundle["test_x"], bundle["test_y"], bundle["test_tq"].astype(np.float32)
    knots = tch.DEFAULT_STUDENT_KNOTS
    info["teacher_test"] = dd.summarize(q, y, x, knots, season, names)
    sn = dd.seasonal_naive(x, args.horizon, season)
    info["seasonal_naive_test_mase"] = float(np.mean(dd.mase(sn, y, x, season)))
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / f"{args.dataset}.npz", **bundle)
    (args.out / f"{args.dataset}_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("teacher_test", "seasonal_naive_test_mase")}, indent=1))


if __name__ == "__main__":
    main()
