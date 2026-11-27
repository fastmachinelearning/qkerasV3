"""Multivariate data for student training: generators, loaders, windows and metrics.

All series are stored as float32 arrays ``(V, T)`` (variates x time).  Splits are
chronological per series (train | val | test) with a gap of one window between the parts,
so no window of a later split overlaps an earlier one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------- synthetic


def _first_order(u: np.ndarray, tau: float, delay: int) -> np.ndarray:
    """Delayed first-order lag ``y' = (u(t-d) - y) / tau`` (discrete, unit step)."""
    a = np.exp(-1.0 / max(tau, 1e-3))
    ud = np.concatenate([np.full(delay, u[0]), u[: len(u) - delay]]) if delay > 0 else u
    y = np.empty_like(u)
    y[0] = ud[0]
    for t in range(1, len(u)):
        y[t] = a * y[t - 1] + (1 - a) * ud[t]
    return y


def _driver(rng: np.random.Generator, length: int) -> np.ndarray:
    """Operator setpoint / load signal: piecewise constant steps, ramps, a daily cycle and noise."""
    t = np.arange(length, dtype=np.float64)
    y = np.zeros(length)
    n_steps = rng.integers(2, 10)
    for c in np.sort(rng.integers(0, length, n_steps)):
        if rng.random() < 0.7:
            y[c:] += rng.normal(0, 1.0)
        else:
            w = rng.integers(10, 80)
            y[c:] += np.clip((t[c:] - c) / w, 0, 1) * rng.normal(0, 1.0)
    if rng.random() < 0.6:
        y += rng.uniform(0.2, 0.8) * np.sin(2 * np.pi * t / rng.choice([24.0, 96.0, rng.uniform(30, 300)]) + rng.uniform(0, 6.3))
    return y + rng.normal(0, 0.03, length)


def synthetic_process(n_series: int, length: int, n_drivers: int = 2, n_targets: int = 3,
                      seed: int = 0) -> list[np.ndarray]:
    """Coupled multivariate series ``(n_drivers + n_targets, length)``.

    Drivers are setpoint-like signals.  Each target is a sum of delayed first-order
    responses to the drivers (random delays 4-96 steps, time constants 2-60 steps), one
    target can depend on another target (cascade: setpoint -> temperature -> emission),
    a random static nonlinearity (identity, exp, saturation) and AR(1) noise.  Because
    targets respond to drivers with a delay, the near-future of a target is partly
    determined by the drivers' recent past, which a model can only use if it looks across
    variates.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_series):
        drv = [_driver(rng, length) for _ in range(n_drivers)]
        tg = []
        for j in range(n_targets):
            y = np.zeros(length)
            srcs = list(range(n_drivers)) + list(range(j))
            for s in rng.choice(srcs, size=min(len(srcs), rng.integers(1, 3)), replace=False):
                u = drv[s] if s < n_drivers else tg[s - n_drivers]
                y += rng.normal(0, 1.0) * _first_order(u, rng.uniform(2, 60), int(rng.integers(4, 97)))
            kind = rng.random()
            if kind < 0.25:
                y = np.exp(np.clip(0.6 * y, -4, 4))
            elif kind < 0.45:
                y = np.tanh(y)
            phi, sig = rng.uniform(0.3, 0.95), rng.uniform(0.02, 0.15) * (np.std(y) + 1e-3)
            e = rng.normal(0, sig, length)
            ar = np.zeros(length)
            for t in range(1, length):
                ar[t] = phi * ar[t - 1] + e[t]
            tg.append(y + ar)
        v = np.stack(drv + tg)
        level = rng.normal(0, 50, (v.shape[0], 1))
        scale = rng.lognormal(0, 1.0, (v.shape[0], 1))
        out.append((level + scale * v).astype(np.float32))
    return out


# ----------------------------------------------------------------------------- real data
KDD_DEFAULT = ("PM2.5", "PM10", "NO2")


def load_kdd_multivariate(parquet: str | Path, measurements=KDD_DEFAULT, max_gap: int = 3) -> list[np.ndarray]:
    """KDD Cup 2018 air quality (Monash/Chronos parquet): one ``(V, T)`` array per station.

    Stations missing any of ``measurements`` are skipped.  Gaps of up to ``max_gap`` hours
    are linearly interpolated; longer gaps stay NaN and windows touching them are dropped.
    """
    import pandas as pd

    df = pd.read_parquet(parquet)
    out = []
    for _, g in df.groupby("station"):
        m = {r.measurement: np.asarray(r.target, np.float64) for r in g.itertuples()}
        if not all(k in m for k in measurements):
            continue
        n = min(len(m[k]) for k in measurements)
        arr = np.stack([_fill_short_gaps(m[k][:n], max_gap) for k in measurements])
        out.append(arr.astype(np.float32))
    return out


def _fill_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    x = x.copy()
    isn = np.isnan(x)
    if not isn.any():
        return x
    idx = np.arange(len(x))
    start = None
    for i in range(len(x) + 1):
        if i < len(x) and isn[i]:
            start = i if start is None else start
        elif start is not None:
            if i - start <= max_gap and start > 0 and i < len(x):
                x[start:i] = np.interp(idx[start:i], [start - 1, i], [x[start - 1], x[i]])
            start = None
    return x


def load_process_csv(csv_path: str | Path, columns: list[str] | None = None, time_col: str = "Datetime_UTC") -> list[np.ndarray]:
    """Semicolon/comma-decimal plant CSV (as used by ``multi_time.py``) -> one ``(V, T)`` array.

    ``columns`` defaults to all numeric columns.  Short dropouts are forward/back filled.
    """
    import pandas as pd

    df = pd.read_csv(csv_path, sep=";", decimal=",", encoding="utf-8-sig")
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col)
    cols = columns or [c for c in df.columns if c != time_col]
    for c in cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    cols = [c for c in cols if df[c].notna().mean() > 0.5]
    df[cols] = df[cols].ffill().bfill()
    return [df[cols].to_numpy(np.float32).T]


# ----------------------------------------------------------------------------- windows
@dataclass
class Windows:
    x: np.ndarray        # (N, V, ctx)
    y: np.ndarray        # (N, V, hor)
    series: np.ndarray   # (N,) source series index

    def __len__(self):
        return len(self.x)

    def subset(self, idx):
        return Windows(self.x[idx], self.y[idx], self.series[idx])


def chronological_split(series: list[np.ndarray], fractions=(0.7, 0.1, 0.2), gap: int = 0):
    """Split every ``(V, T)`` series in time into train/val/test parts separated by ``gap`` steps."""
    parts = ([], [], [])
    for s in series:
        t = s.shape[1]
        a = int(t * fractions[0])
        b = int(t * (fractions[0] + fractions[1]))
        parts[0].append(s[:, : max(a - gap, 0)])
        parts[1].append(s[:, a : max(b - gap, a)])
        parts[2].append(s[:, b:])
    return parts


def make_windows(series: list[np.ndarray], ctx: int, hor: int, stride: int, max_windows: int | None = None,
                 seed: int = 0, min_std: float = 1e-6) -> Windows:
    """All NaN-free windows ``(V, ctx+hor)`` with the given stride (optionally subsampled)."""
    xs, ys, sid = [], [], []
    for i, s in enumerate(series):
        t = s.shape[1]
        for st in range(0, t - ctx - hor + 1, stride):
            w = s[:, st : st + ctx + hor]
            if not np.isfinite(w).all():
                continue
            if np.min(np.std(w[:, :ctx], axis=1)) < min_std:
                continue
            xs.append(w[:, :ctx])
            ys.append(w[:, ctx:])
            sid.append(i)
    if not xs:
        v = series[0].shape[0] if series else 1
        return Windows(np.zeros((0, v, ctx), np.float32), np.zeros((0, v, hor), np.float32), np.zeros(0, int))
    w = Windows(np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(sid))
    if max_windows is not None and len(w) > max_windows:
        idx = np.sort(np.random.default_rng(seed).choice(len(w), max_windows, replace=False))
        w = w.subset(idx)
    return w


# ----------------------------------------------------------------------------- metrics
def mase(median: np.ndarray, y: np.ndarray, context: np.ndarray, season: int = 1) -> np.ndarray:
    """MASE per window and variate ``(N, V)``; scale = in-sample seasonal-naive MAE of the context."""
    scale = np.mean(np.abs(context[..., season:] - context[..., :-season]), axis=-1)
    return np.mean(np.abs(median - y), axis=-1) / np.maximum(scale, 1e-8)


def wql(q: np.ndarray, y: np.ndarray, knots, levels=None) -> np.ndarray:
    """Weighted quantile loss per variate ``(V,)`` (sum over windows/steps, / sum |y|)."""
    knots = [round(float(k), 6) for k in knots]
    levels = levels or [k for k in knots if k not in (0.01, 0.99)]
    tot = np.zeros(y.shape[1])
    for lv in levels:
        pred = q[..., knots.index(round(lv, 6))]
        d = y - pred
        tot += 2 * np.sum(np.maximum(lv * d, (lv - 1) * d), axis=(0, 2))
    return tot / len(levels) / np.maximum(np.sum(np.abs(y), axis=(0, 2)), 1e-8)


def coverage(q: np.ndarray, y: np.ndarray, knots, lo=0.1, hi=0.9) -> float:
    k = [round(float(v), 6) for v in knots]
    a, b = q[..., k.index(lo)], q[..., k.index(hi)]
    return float(np.mean((y >= a) & (y <= b)))


def seasonal_naive(context: np.ndarray, hor: int, season: int) -> np.ndarray:
    reps = int(np.ceil(hor / season))
    return np.tile(context[..., -season:], reps)[..., :hor]


def summarize(q: np.ndarray, y: np.ndarray, x: np.ndarray, knots, season: int, variates=None) -> dict:
    """Aggregate metrics: mean MASE, mean WQL, P10-P90 coverage, plus per-variate MASE/WQL."""
    m = list(round(float(k), 6) for k in knots).index(0.5)
    ms = mase(q[..., m], y, x, season)
    wq = wql(q, y, knots)
    out = {"mase": float(np.mean(ms)), "wql": float(np.mean(wq)), "coverage_80": coverage(q, y, knots)}
    names = variates or [f"v{i}" for i in range(y.shape[1])]
    out["per_variate"] = {n: {"mase": float(np.mean(ms[:, i])), "wql": float(wq[i])} for i, n in enumerate(names)}
    return out
