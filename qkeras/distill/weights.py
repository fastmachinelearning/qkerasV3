"""Moving weights between float, QAT and pruned variants of one graph, and sparsity stats."""

from __future__ import annotations

import keras
import numpy as np


def own_variables(layer) -> list:
    """Layer variables without qkeras.pruning masks (they live in a ``pruning_masks`` sublayer)."""
    return [v for v in layer.weights if "_pruning_mask" not in (getattr(v, "path", None) or v.name)]


def transfer_weights(src: keras.Model, dst: keras.Model) -> int:
    """Copy weights layer by layer (matching names and shapes, pruning masks ignored on both sides).

    A pruned source must have its masks baked in (``qkeras.pruning.apply_pruning_masks`` or
    ``UpdatePruningStep`` at the end of ``fit``) for the destination to be exactly sparse.
    Returns the number of layers copied.
    """
    n = 0
    for lyr in dst.layers:
        try:
            s = src.get_layer(lyr.name)
        except ValueError:
            continue
        sv, dv = own_variables(s), own_variables(lyr)
        if sv and len(sv) == len(dv) and all(tuple(a.shape) == tuple(b.shape) for a, b in zip(sv, dv)):
            for a, b in zip(sv, dv):
                b.assign(a)
            n += 1
    return n


def quantized_kernels(qmodel) -> list[tuple[str, np.ndarray]]:
    """``(layer name, quantised kernel)`` of every layer with a ``kernel_quantizer``."""
    out = []
    for lyr in qmodel.layers:
        q = getattr(lyr, "kernel_quantizer_internal", None)
        if q is None or not hasattr(lyr, "kernel"):
            continue
        out.append((lyr.name, np.asarray(q(lyr.kernel), dtype=np.float64)))
    return out


def dense_sparsity(qmodel) -> dict:
    """Zero fraction of the *quantised* kernels (pruned + quantised-to-zero weights)."""
    z = t = 0
    for _, k in quantized_kernels(qmodel):
        z += int(np.sum(k == 0))
        t += k.size
    return {"zero_fraction": z / t if t else 0.0, "nonzero": t - z, "total": t}
