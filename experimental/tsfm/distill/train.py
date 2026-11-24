"""Train one student: float distillation, then quantisation-aware fine-tuning.

Example::

    python -m distill.train --config distill/configs/A_d64_L3.json --data runs/data/kdd.npz \
        --out runs/students/kdd/A_d64_L3

Outputs in ``--out``: config.json, float.weights.h5, qat.weights.h5, metrics.json (test
metrics for float/QAT student, teacher and seasonal naive on the same windows, sizes and
the GateMate cost estimate), history.json.

Pruning (class-B students): add ``"prune": {"sparsity": 0.7, "ramp_frac": 0.5}`` to the
``train`` block (or ``--sparsity 0.7``).  The QAT phase then runs a magnitude-pruning ramp
(qkeras.pruning PolynomialDecay 0 -> sparsity over ``ramp_frac`` of the QAT epochs, no early
stopping) followed by fine-tuning with frozen masks; the saved ``qat.weights.h5`` is a plain
QAT model (masks baked in), so RTL export and cost estimation are unchanged.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import keras
import numpy as np

import keras_timestral as kt

from . import data as dd
from . import fpga_cost
from . import student as stu
from . import teacher as tch
from .losses import DistillQuantileLoss, gt_pinball_metric

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


def fit(model, tr, va, cfg_train, knots, alpha, lr, epochs, out: Path, tag: str, extra_callbacks=(),
        early_stop: bool = True, save: bool = True):
    steps = max(1, int(np.ceil(len(tr["xs"]) / cfg_train["batch_size"]))) * epochs
    sched = keras.optimizers.schedules.CosineDecay(lr, decay_steps=steps, alpha=0.05,
                                                   warmup_target=None)
    model.compile(optimizer=keras.optimizers.AdamW(sched, weight_decay=cfg_train.get("weight_decay", 1e-4),
                                                   clipnorm=1.0),
                  loss=DistillQuantileLoss(knots, alpha), metrics=[gt_pinball_metric(knots)])
    cbs = list(extra_callbacks)
    if early_stop:
        cbs.append(keras.callbacks.EarlyStopping(monitor="val_gt_pinball", mode="min",
                                                 patience=cfg_train.get("patience", 6), restore_best_weights=True))
    t0 = time.time()
    h = model.fit(tr["xs"], tr["packed"], validation_data=(va["xs"], va["packed"]), epochs=epochs,
                  batch_size=cfg_train["batch_size"], callbacks=cbs, verbose=cfg_train.get("verbose", 2))
    if save:
        model.save_weights(str(out / f"{tag}.weights.h5"))
    return {k: [float(v) for v in vals] for k, vals in h.history.items()}, round(time.time() - t0, 1)


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

    # phase 1: float distillation
    fmodel = stu.build_student(cfg, V, "float")
    hist, sec = fit(fmodel, tr, va, tcfg, cfg.knots, tcfg["alpha"], tcfg["lr"], tcfg["float_epochs"], args.out, "float")
    metrics["float"] = dd.summarize(predict_raw(fmodel, te), te["y"], te["x"], cfg.knots, season, names)
    metrics["float"]["train_seconds"] = sec
    history = {"float": hist}

    # phase 2: QAT
    if PRESET_OK(cfg.preset):
        qmodel = stu.build_student(cfg, V, cfg.preset)
        stu.transfer_weights(fmodel, qmodel)
        metrics["qat_ptq"] = dd.summarize(predict_raw(qmodel, te), te["y"], te["x"], cfg.knots, season, names)
        if tcfg["qat_epochs"] > 0 and tcfg.get("prune"):
            qmodel, history["qat"], sec, metrics["prune"] = qat_with_pruning(fmodel, cfg, V, tr, va, tcfg, args.out)
            metrics["qat"] = dd.summarize(predict_raw(qmodel, te), te["y"], te["x"], cfg.knots, season, names)
            metrics["qat"]["train_seconds"] = sec
        elif tcfg["qat_epochs"] > 0:
            hist, sec = fit(qmodel, tr, va, tcfg, cfg.knots, tcfg["alpha"], tcfg["qat_lr"], tcfg["qat_epochs"],
                            args.out, "qat")
            history["qat"] = hist
            metrics["qat"] = dd.summarize(predict_raw(qmodel, te), te["y"], te["x"], cfg.knots, season, names)
            metrics["qat"]["train_seconds"] = sec
        metrics["sparsity"] = stu.dense_sparsity(qmodel)
        metrics["cost"] = fpga_cost.estimate(cfg, V, qmodel)
    else:
        metrics["cost"] = fpga_cost.estimate(cfg, V, None)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (args.out / "history.json").write_text(json.dumps(history))
    best = metrics.get("qat", metrics["float"])
    print(f"[{cfg.name}] params={metrics['ops']['params']} float MASE={metrics['float']['mase']:.3f} "
          f"final MASE={best['mase']:.3f} WQL={best['wql']:.3f} | teacher MASE={metrics['teacher']['mase']:.3f} "
          f"| snaive {metrics['seasonal_naive_mase']:.3f} | fits GateMate: {metrics['cost'].get('fits')}")


def qat_with_pruning(fmodel, cfg, V, tr, va, tcfg, out: Path):
    """QAT with a magnitude-pruning ramp, then frozen-mask fine-tuning; returns a plain QAT model."""
    from qkeras.pruning import MagnitudePruner, PolynomialDecay, UpdatePruningStep, get_pruning_summary

    pc = {"ramp_frac": 0.5, **tcfg["prune"]}
    target = float(pc["sparsity"])
    spe = max(1, int(np.ceil(len(tr["xs"]) / tcfg["batch_size"])))
    ramp_ep = max(1, int(round(pc["ramp_frac"] * tcfg["qat_epochs"])))
    tune_ep = max(1, tcfg["qat_epochs"] - ramp_ep)
    pruner = MagnitudePruner(PolynomialDecay(0.0, target, begin_step=0, end_step=ramp_ep * spe,
                                             frequency=max(1, spe // 4)))
    pmodel = stu.build_student(cfg, V, cfg.preset, pruner=pruner)
    stu.transfer_weights(fmodel, pmodel)
    # ramp: masks follow the schedule; no early stopping (restore_best_weights would also restore older masks)
    h1, s1 = fit(pmodel, tr, va, tcfg, cfg.knots, tcfg["alpha"], tcfg["qat_lr"], ramp_ep, out, "qat_ramp",
                 extra_callbacks=[UpdatePruningStep()], early_stop=False, save=False)
    # fine-tune: no callback -> masks stay fixed and pruned weights get zero gradient
    h2, s2 = fit(pmodel, tr, va, tcfg, cfg.knots, tcfg["alpha"], tcfg["qat_lr"], tune_ep, out, "qat_tune",
                 save=False)
    from qkeras.pruning import apply_pruning_masks
    apply_pruning_masks(pmodel)
    summary = get_pruning_summary(pmodel)
    qmodel = stu.build_student(cfg, V, cfg.preset)
    stu.transfer_weights(pmodel, qmodel)
    qmodel.save_weights(str(out / "qat.weights.h5"))
    info = {"target": target, "ramp_epochs": ramp_ep, "tune_epochs": tune_ep,
            "mask_sparsity": float(summary["__total__"])}
    hist = {f"ramp_{k}": v for k, v in h1.items()} | {f"tune_{k}": v for k, v in h2.items()}
    return qmodel, hist, round(s1 + s2, 1), info


def PRESET_OK(p):
    return p and stu.PRESETS.get(p) is not None


if __name__ == "__main__":
    main()
