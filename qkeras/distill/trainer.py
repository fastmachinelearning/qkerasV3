"""Two-phase distillation: float student -> QAT student (optionally with a pruning ramp).

The trainer is model-agnostic.  It needs

* ``build(preset, pruner=None) -> keras.Model`` - the student graph for a preset (``"float"``
  or a QAT preset); usually ``builder(..., factory=make_factory(preset, pruner=pruner))``,
* ``train = (x, y_packed)``, ``val = (x, y_packed)`` with ``y_packed`` from ``losses.pack_targets``,
* ``loss`` (a ``keras.losses.Loss``) and ``metrics`` for ``compile``.

Phase 1 trains the float student.  Phase 2 copies its weights into the QAT graph
(``transfer_weights``; the copy before any QAT step is the post-training-quantised model) and
fine-tunes.  With ``tcfg["prune"] = {"sparsity": s, "ramp_frac": r}`` phase 2 instead runs a
magnitude-pruning ramp ``0 -> s`` over ``r`` of the QAT epochs (no early stopping, which would
also restore older masks), then fine-tunes with frozen masks; the returned QAT model is a plain
(un-prunable) graph with the masks baked in, so export and cost estimation do not see pruning.

Optimiser: AdamW + cosine decay to 5 % of the learning rate, ``clipnorm=1``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import keras
import numpy as np

from .presets import is_quantized
from .weights import transfer_weights

TRAIN_DEFAULTS = {"batch_size": 64, "float_epochs": 40, "qat_epochs": 15, "lr": 2e-3, "qat_lr": 3e-4,
                  "alpha": 0.5}


def fit_phase(model, train, val, loss, metrics, lr: float, epochs: int, tcfg: dict, monitor: str = "val_loss",
              out: Path | None = None, tag: str | None = None, callbacks=(), early_stop: bool = True):
    """Compile and fit one phase; saves ``<out>/<tag>.weights.h5`` if both are given.

    ``tcfg`` keys used: ``batch_size``, ``weight_decay`` (1e-4), ``patience`` (6), ``verbose`` (2).
    Returns ``(history dict, seconds)``.
    """
    steps = max(1, int(np.ceil(len(train[0]) / tcfg["batch_size"]))) * epochs
    sched = keras.optimizers.schedules.CosineDecay(lr, decay_steps=steps, alpha=0.05, warmup_target=None)
    model.compile(optimizer=keras.optimizers.AdamW(sched, weight_decay=tcfg.get("weight_decay", 1e-4),
                                                   clipnorm=1.0),
                  loss=loss, metrics=list(metrics))
    cbs = list(callbacks)
    if early_stop:
        cbs.append(keras.callbacks.EarlyStopping(monitor=monitor, mode="min", patience=tcfg.get("patience", 6),
                                                 restore_best_weights=True))
    t0 = time.time()
    h = model.fit(train[0], train[1], validation_data=val, epochs=epochs, batch_size=tcfg["batch_size"],
                  callbacks=cbs, verbose=tcfg.get("verbose", 2))
    if out is not None and tag is not None:
        model.save_weights(str(Path(out) / f"{tag}.weights.h5"))
    return {k: [float(v) for v in vals] for k, vals in h.history.items()}, round(time.time() - t0, 1)


def prune_qat(build: Callable, src, preset, train, val, loss, metrics, tcfg: dict, out: Path | None = None,
              monitor: str = "val_loss"):
    """QAT with a magnitude-pruning ramp, then frozen-mask fine-tuning.

    Returns ``(plain QAT model, history, seconds, info)``; saves ``<out>/qat.weights.h5``.
    """
    from qkeras.pruning import (MagnitudePruner, PolynomialDecay, UpdatePruningStep, apply_pruning_masks,
                                get_pruning_summary)

    pc = {"ramp_frac": 0.5, **tcfg["prune"]}
    target = float(pc["sparsity"])
    spe = max(1, int(np.ceil(len(train[0]) / tcfg["batch_size"])))
    ramp_ep = max(1, int(round(pc["ramp_frac"] * tcfg["qat_epochs"])))
    tune_ep = max(1, tcfg["qat_epochs"] - ramp_ep)
    pruner = MagnitudePruner(PolynomialDecay(0.0, target, begin_step=0, end_step=ramp_ep * spe,
                                             frequency=max(1, spe // 4)))
    pmodel = build(preset, pruner=pruner)
    transfer_weights(src, pmodel)
    h1, s1 = fit_phase(pmodel, train, val, loss, metrics, tcfg["qat_lr"], ramp_ep, tcfg, monitor,
                       callbacks=[UpdatePruningStep()], early_stop=False)
    h2, s2 = fit_phase(pmodel, train, val, loss, metrics, tcfg["qat_lr"], tune_ep, tcfg, monitor)
    apply_pruning_masks(pmodel)
    summary = get_pruning_summary(pmodel)
    qmodel = build(preset)
    transfer_weights(pmodel, qmodel)
    if out is not None:
        qmodel.save_weights(str(Path(out) / "qat.weights.h5"))
    info = {"target": target, "ramp_epochs": ramp_ep, "tune_epochs": tune_ep,
            "mask_sparsity": float(summary["__total__"])}
    hist = {f"ramp_{k}": v for k, v in h1.items()} | {f"tune_{k}": v for k, v in h2.items()}
    return qmodel, hist, round(s1 + s2, 1), info


def distill(build: Callable, preset, train, val, loss, metrics, tcfg: dict, out: Path | None = None,
            monitor: str = "val_loss", evaluate: Callable | None = None) -> dict:
    """Run both phases.

    ``evaluate(stage, model) -> dict`` is called for stages ``"float"``, ``"qat_ptq"`` (QAT graph
    with the float weights, before any QAT step) and ``"qat"``.  Returns ``{"float": model,
    "qat": model | None, "history": {...}, "metrics": {stage: ...}, "seconds": {...}, "prune": info}``.
    """
    tcfg = {**TRAIN_DEFAULTS, **tcfg}
    ev = evaluate or (lambda stage, model: {})
    res = {"history": {}, "metrics": {}, "seconds": {}, "prune": None, "qat": None}

    fmodel = build("float")
    res["history"]["float"], res["seconds"]["float"] = fit_phase(
        fmodel, train, val, loss, metrics, tcfg["lr"], tcfg["float_epochs"], tcfg, monitor, out, "float")
    res["float"] = fmodel
    res["metrics"]["float"] = ev("float", fmodel)

    if not is_quantized(preset):
        return res
    qmodel = build(preset)
    transfer_weights(fmodel, qmodel)
    res["metrics"]["qat_ptq"] = ev("qat_ptq", qmodel)
    if tcfg["qat_epochs"] > 0 and tcfg.get("prune"):
        qmodel, res["history"]["qat"], res["seconds"]["qat"], res["prune"] = prune_qat(
            build, fmodel, preset, train, val, loss, metrics, tcfg, out, monitor)
        res["metrics"]["qat"] = ev("qat", qmodel)
    elif tcfg["qat_epochs"] > 0:
        res["history"]["qat"], res["seconds"]["qat"] = fit_phase(
            qmodel, train, val, loss, metrics, tcfg["qat_lr"], tcfg["qat_epochs"], tcfg, monitor, out, "qat")
        res["metrics"]["qat"] = ev("qat", qmodel)
    res["qat"] = qmodel
    return res
