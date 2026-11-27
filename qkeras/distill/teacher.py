"""Teachers and the soft-label cache.

A *teacher* is anything with ``predict(x) -> np.ndarray`` (a Keras model, a wrapper around
a PyTorch / HuggingFace model, an API).  Teachers are usually large, so their outputs are
computed once per dataset and cached on disk (``cached``); training then reads the cache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Protocol

import numpy as np


class Teacher(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray: ...


class KerasTeacher:
    """Batched ``model.predict`` with optional pre/post-processing (e.g. scaling)."""

    def __init__(self, model, batch_size: int = 64, pre: Callable | None = None, post: Callable | None = None):
        self.model, self.batch_size, self.pre, self.post = model, batch_size, pre, post

    def predict(self, x: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(x), self.batch_size):
            xb = x[i: i + self.batch_size]
            xb = self.pre(xb) if self.pre else xb
            y = np.asarray(self.model(xb, training=False))
            out.append(self.post(y) if self.post else y)
        return np.concatenate(out, axis=0)


def cache_key(**fields) -> str:
    """Short stable hash of JSON-serialisable ``fields``."""
    return hashlib.sha1(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()[:12]


def cached(cache_dir: str | Path, key: str, compute: Callable[[], np.ndarray], meta: dict | None = None,
           prefix: str = "teacher") -> np.ndarray:
    """Return ``<cache_dir>/<prefix>_<key>.npz['q']``, computing and storing it on a miss."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{prefix}_{key}.npz"
    if path.exists():
        return np.load(path)["q"]
    q = np.asarray(compute(), dtype=np.float32)
    np.savez_compressed(path, q=q, meta=json.dumps(meta or {}, default=str))
    return q
