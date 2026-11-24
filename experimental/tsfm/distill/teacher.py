"""Timestral-5M teacher: multivariate forecasts cached as soft labels.

The teacher sees a longer context than the students (default 512 vs 256 steps), so the
students also learn from information outside their own window.  Forecasts are stored in
raw space at the student's quantile knots; ``to_student_space`` maps them (and the ground
truth) into the student's normalised frame (``asinh((y - loc) / scale)`` with the
statistics of the student's own context).  Because ``asinh`` is monotone, quantiles map to
quantiles.

Weights: Mistral AI Modified MIT licence (see the checkpoint's LICENSE).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import keras_timestral as kt

DEFAULT_STUDENT_KNOTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def load_teacher(ckpt_dir: str | Path, num_variates: int, context_patches: int, horizon_patches: int):
    from safetensors.numpy import load_file

    ckpt_dir = Path(ckpt_dir)
    spec = kt.TimestralSpec.from_json(ckpt_dir / "config.json")
    weights = load_file(ckpt_dir / "model.safetensors")
    if num_variates > 1:
        model = kt.build_timestral_multivariate(spec, num_variates, context_patches, horizon_patches, name="teacher")
    else:
        model = kt.build_timestral(spec, context_patches, horizon_patches, name="teacher")
    kt.load_timestral_weights(model, weights, spec)
    return model, spec


def teacher_forecast(model, spec, x: np.ndarray, batch: int = 64, verbose: bool = False) -> np.ndarray:
    """Raw-space teacher quantiles ``(N, V, H, Q_teacher)`` for contexts ``(N, V, T)``."""
    out = []
    for i in range(0, len(x), batch):
        xb = x[i : i + batch]
        if xb.shape[1] == 1:
            q = kt.forecast(model, xb[:, 0], spec)[:, None]
        else:
            q = kt.forecast_mv(model, xb, spec)
        out.append(q.astype(np.float32))
        if verbose and (i // batch) % 20 == 0:
            print(f"  teacher {i + len(xb)}/{len(x)}", flush=True)
    return np.concatenate(out, axis=0)


def select_knots(q: np.ndarray, teacher_knots, student_knots) -> np.ndarray:
    tk = [round(float(k), 6) for k in teacher_knots]
    idx = [tk.index(round(float(k), 6)) for k in student_knots]
    return q[..., idx]


def cache_key(**kw) -> str:
    return hashlib.sha1(json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()[:12]


def cached_teacher_labels(cache_dir: str | Path, key_fields: dict, ckpt_dir: str | Path, x: np.ndarray,
                          horizon: int, student_knots=DEFAULT_STUDENT_KNOTS, batch: int = 64,
                          verbose: bool = True) -> np.ndarray:
    """Teacher quantiles at ``student_knots`` ``(N, V, H, Qs)`` (raw space), cached on disk."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(n=len(x), shape=list(x.shape), horizon=horizon, knots=list(student_knots),
                    checksum=float(np.nansum(x[:, :, -8:].astype(np.float64))), **key_fields)
    path = cache_dir / f"teacher_{key}.npz"
    if path.exists():
        return np.load(path)["q"]
    p = 32
    assert x.shape[-1] % p == 0 and horizon % p == 0, "teacher needs multiples of its patch size (32)"
    model, spec = load_teacher(ckpt_dir, x.shape[1], x.shape[-1] // p, horizon // p)
    q = teacher_forecast(model, spec, x, batch=batch, verbose=verbose)
    q = select_knots(q, spec.quantile_knots, student_knots)
    np.savez_compressed(path, q=q.astype(np.float32), meta=json.dumps(key_fields, default=str))
    return q


def student_frame(x_student: np.ndarray, patch_size: int, minimum_scale: float = 1e-6, use_asinh: bool = True):
    """Scale student contexts ``(N, V, T_s)``; returns ``(scaled, loc, scale)`` with ``loc/scale (N, V, 1)``."""
    return kt.scale_context_mv(x_student, patch_size, minimum_scale, use_asinh)


def to_student_space(values: np.ndarray, loc: np.ndarray, scale: np.ndarray, use_asinh: bool = True) -> np.ndarray:
    """Raw ``(N, V, H[, Q])`` -> student normalised space with per-row ``loc/scale (N, V, 1)``."""
    if values.ndim == 4:
        loc, scale = loc[..., None], scale[..., None]
    z = (values - loc) / scale
    return (np.arcsinh(z) if use_asinh else z).astype(np.float32)


def from_student_space(z: np.ndarray, loc: np.ndarray, scale: np.ndarray, use_asinh: bool = True) -> np.ndarray:
    if z.ndim == 4:
        loc, scale = loc[..., None], scale[..., None]
    return ((np.sinh(z) if use_asinh else z) * scale + loc).astype(np.float32)
