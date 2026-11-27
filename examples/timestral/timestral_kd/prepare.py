"""Build dataset bundles (windows + teacher soft labels + teacher scores).

Usually run as ``python -m timestral_kd data <dataset>``; directly::

    python -m timestral_kd.prepare --dataset kdd          # downloads the KDD parquet on first use
    python -m timestral_kd.prepare --dataset synthetic
    python -m timestral_kd.prepare --dataset csv --csv process.csv --columns NOx O2 ... --name process

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
from . import paths
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
    ap.add_argument("--out", type=Path, default=None, help="default: runs/data")
    ap.add_argument("--ckpt", type=Path, default=None, help="default: <repo>/ts/Timestral-5M or $TIMESTRAL_CKPT")
    ap.add_argument("--kdd-parquet", type=Path, default=None, help="default: downloaded to runs/data/raw")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--name", default=None, help="bundle name (default: the dataset name)")
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
    args.out = args.out or paths.data_dir()
    args.ckpt = args.ckpt or paths.CKPT
    if args.dataset == "kdd" and args.kdd_parquet is None:
        args.kdd_parquet = paths.kdd_parquet()
    if args.dataset == "csv" and args.csv is None:
        ap.error("--csv is required for --dataset csv")

    season, stride, _ = DATASETS[args.dataset]
    stride = args.stride or stride
    series, names = build_series(args)
    parts = dd.chronological_split(series, (0.7, 0.1, 0.2))
    bundle, info = {}, {"dataset": args.name or args.dataset, "variates": names, "season": season,
                        "teacher_ctx": args.teacher_ctx, "horizon": args.horizon,
                        "knots": list(tch.DEFAULT_STUDENT_KNOTS)}
    for split, p, st, cap in [("train", parts[0], stride, args.max_train), ("val", parts[1], stride, 2000),
                              ("test", parts[2], 2 * stride, 4000)]:
        w = dd.make_windows(p, args.teacher_ctx, args.horizon, st, max_windows=cap, seed=args.seed)
        if len(w) == 0:
            raise SystemExit(f"{split} split has no window of {args.teacher_ctx} + {args.horizon} steps; "
                             "use longer series (--length) or a smaller --teacher-ctx / --horizon")
        t0 = time.time()
        tq = tch.cached_teacher_labels(args.out / "cache", {"dataset": args.name or args.dataset, "split": split, "seed": args.seed},
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
    np.savez_compressed(args.out / f"{args.name or args.dataset}.npz", **bundle)
    (args.out / f"{args.name or args.dataset}_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("teacher_test", "seasonal_naive_test_mase")}, indent=1))


if __name__ == "__main__":
    main()
