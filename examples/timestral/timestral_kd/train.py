"""Train one student: float distillation, then quantisation-aware fine-tuning.

Example::

    python -m timestral_kd.train --config configs/students/smoke.json --data runs/data/kdd.npz \
        --out runs/students/kdd/smoke

Outputs in ``--out``: config.json, float.weights.h5, qat.weights.h5, metrics.json (test
metrics for float/QAT student, teacher and seasonal naive on the same windows, sizes and
the GateMate cost estimate), history.json.

Pruning (class-B students): add ``"prune": {"sparsity": 0.7, "ramp_frac": 0.5}`` to the
``train`` block (or ``--sparsity 0.7``).  The QAT phase then runs a magnitude-pruning ramp
(``qkeras.distill.trainer.prune_qat``: PolynomialDecay 0 -> sparsity over ``ramp_frac`` of the
QAT epochs, no early stopping) followed by fine-tuning with frozen masks; the saved ``qat.weights.h5`` is a plain
QAT model (masks baked in), so RTL export and cost estimation are unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import numpy as np

from qkeras.distill.losses import QuantileDistillLoss, gt_pinball_metric
from qkeras.distill.presets import is_quantized
from qkeras.distill.trainer import distill

from . import cost as fpga_cost
from . import data as dd
from . import student as stu
from . import teacher as tch

TEACHER_KNOTS = tch.DEFAULT_STUDENT_KNOTS   # knots stored in the bundle
CLIP = 10.0


def load_bundle(path: Path):
    b = np.load(path)
    info = json.loads(Path(str(path).replace(".npz", "_info.json")).read_text())
    return b, info


def prepare_split(b, split: str, cfg: stu.StudentConfig, max_n: int | None = None):
    """Student inputs and packed targets for one split."""
    x_full, y, tq = b[f"{split}_x"], b[f"{split}_y"], b[f"{split}_tq"].astype(np.float32)
    if max_n is not None and len(x_full) > max_n:
        idx = np.random.default_rng(0).choice(len(x_full), max_n, replace=False)
        x_full, y, tq = x_full[idx], y[idx], tq[idx]
    x = x_full[..., -cfg.ctx_steps:]
    y = y[..., : cfg.hor_steps]
    tq = tch.select_knots(tq[:, :, : cfg.hor_steps], TEACHER_KNOTS, cfg.knots)
    xs, loc, sc = tch.student_frame(x, cfg.patch_size)
    yn = np.clip(tch.to_student_space(y, loc, sc), -CLIP, CLIP)
    tqn = np.clip(tch.to_student_space(tq, loc, sc), -CLIP, CLIP)
    packed = np.concatenate([yn[..., None], tqn], axis=-1).astype(np.float32)
    return dict(xs=xs, packed=packed, x=x, y=y, tq=tq, loc=loc, sc=sc)


def predict_raw(model, d, batch=256):
    z = model.predict(d["xs"], batch_size=batch, verbose=0)
    return tch.from_student_space(z, d["loc"], d["sc"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="bundle from distill.prepare")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=None, help="override: 1 = no distillation")
    ap.add_argument("--variate-every", type=int, default=None, help="override (0 = channel-independent)")
    ap.add_argument("--float-epochs", type=int, default=None)
    ap.add_argument("--qat-epochs", type=int, default=None)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sparsity", type=float, default=None, help="override: magnitude-pruning target in QAT")
    args = ap.parse_args(argv)

    conf = json.loads(args.config.read_text())
    cfg = stu.StudentConfig.from_dict(conf["student"])
    tcfg = {"batch_size": 64, "float_epochs": 40, "qat_epochs": 15, "lr": 2e-3, "qat_lr": 3e-4, "alpha": 0.5,
            **conf.get("train", {})}
    if args.alpha is not None:
        tcfg["alpha"] = args.alpha
    if args.variate_every is not None:
        cfg.variate_every = args.variate_every
    if args.float_epochs is not None:
        tcfg["float_epochs"] = args.float_epochs
    if args.qat_epochs is not None:
        tcfg["qat_epochs"] = args.qat_epochs
    if args.sparsity is not None:
        tcfg["prune"] = {**tcfg.get("prune", {}), "sparsity": args.sparsity}
    if tcfg.get("prune") and not tcfg["prune"].get("sparsity"):
        tcfg.pop("prune")
    tcfg["seed"] = args.seed
    keras.utils.set_random_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    b, info = load_bundle(args.data)
    season = info["season"]
    tr = prepare_split(b, "train", cfg, args.max_train or tcfg.get("max_train"))
    va = prepare_split(b, "val", cfg)
    te = prepare_split(b, "test", cfg)
    V = tr["xs"].shape[1]
    names = info.get("variates")
    (args.out / "config.json").write_text(json.dumps({"student": cfg.to_dict(), "train": tcfg,
                                                      "data": str(args.data)}, indent=1))

    metrics = {"name": cfg.name, "dataset": info["dataset"], "V": V, "alpha": tcfg["alpha"],
               "variate_every": cfg.variate_every, "deployment": cfg.deployment, "preset": cfg.preset,
               "n_train": len(tr["xs"]), "n_test": len(te["xs"]), "seed": args.seed}
    metrics["teacher"] = dd.summarize(te["tq"], te["y"], te["x"], cfg.knots, season, names)
    sn = dd.seasonal_naive(te["x"], cfg.hor_steps, season)
    metrics["seasonal_naive_mase"] = float(np.mean(dd.mase(sn, te["y"], te["x"], season)))
    metrics["ops"] = stu.count_ops(cfg, V)

    def evaluate(stage, model):
        return dd.summarize(predict_raw(model, te), te["y"], te["x"], cfg.knots, season, names)

    res = distill(lambda preset, pruner=None: stu.build_student(cfg, V, preset, pruner=pruner),
                  cfg.preset, (tr["xs"], tr["packed"]), (va["xs"], va["packed"]),
                  QuantileDistillLoss(cfg.knots, tcfg["alpha"]), [gt_pinball_metric(cfg.knots)], tcfg,
                  out=args.out, monitor="val_gt_pinball", evaluate=evaluate)
    for stage in ("float", "qat_ptq", "prune", "qat"):
        if stage == "prune":
            if res["prune"] is not None:
                metrics["prune"] = res["prune"]
            continue
        if stage in res["metrics"]:
            metrics[stage] = res["metrics"][stage]
            if stage in res["seconds"]:
                metrics[stage]["train_seconds"] = res["seconds"][stage]
    history = res["history"]
    if is_quantized(cfg.preset):
        metrics["sparsity"] = stu.dense_sparsity(res["qat"])
        metrics["cost"] = fpga_cost.estimate(cfg, V, res["qat"])
    else:
        metrics["cost"] = fpga_cost.estimate(cfg, V, None)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (args.out / "history.json").write_text(json.dumps(history))
    best = metrics.get("qat", metrics["float"])
    print(f"[{cfg.name}] params={metrics['ops']['params']} float MASE={metrics['float']['mase']:.3f} "
          f"final MASE={best['mase']:.3f} WQL={best['wql']:.3f} | teacher MASE={metrics['teacher']['mase']:.3f} "
          f"| snaive {metrics['seasonal_naive_mase']:.3f} | fits GateMate: {metrics['cost'].get('fits')}")


if __name__ == "__main__":
    main()
