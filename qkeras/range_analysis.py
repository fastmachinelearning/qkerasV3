# Copyright 2026 QKeras-V3 authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Range analysis and range-driven bit-width adaptation for QKeras layers.

The module observes the values that flow *into* every fixed-point quantizer of
a model (weights, biases, pre-activation accumulators, activations) and sizes
the integer part of each quantizer so that the observed range fits.  Two entry
points are provided:

* :func:`convert_to_lut` -- post-training conversion of the calibrated model
  into a LUT network (teacher -> student, see :mod:`qkeras.logic.convert`).
  It uses :func:`analyze_value_sets` (the finite set of values every
  quantised activation takes = how many wires it needs) and
  :func:`analyze_patterns` (the input bit-patterns every neuron actually sees
  = its care set; everything else is a don't-care).
* :func:`optimize_bitwidths` -- one-shot, post-training.  Calibrates on user
  data or on random inputs and rewrites the quantizers in place.
* :class:`RangeAnalysisCallback` -- the same procedure inside ``model.fit``,
  every ``frequency`` epochs, switchable on/off with ``enabled`` or with a
  ``start_epoch`` / ``stop_epoch`` window.

:func:`analyze_ranges` runs the calibration only and returns the report
without touching the model.

Why the quantizer *input* is recorded: a quantizer clips, so its output can
never show an overflow.  The value presented to the quantizer is what the
hardware word must hold.

Bit-width policy.  For a fixed-point quantizer with ``S`` sign bits, ``I``
integer bits and ``F = bits - S - I`` fractional bits:

* ``"keep_fractional"`` (default): ``F`` is kept, ``I`` is set from the range,
  so ``bits`` grows or shrinks with the range.  This removes unused MSBs
  without changing the resolution.
* ``"keep_total"``: ``bits`` is kept, the binary point moves.  A smaller range
  buys resolution; a larger range costs resolution.

Mechanics.  ``bits`` and ``integer`` are Python numbers that are baked into
the traced train/test/predict functions.  After a change, the compiled
functions are rebuilt (``make_train_function(force=True)``), so the new
widths are effective from the next step on every backend.  Calibration runs
eagerly at epoch end, after the JAX trainer has synced its state back into the
model.

Supported quantizers: ``quantized_bits`` with ``alpha`` None or 1 and
``quantized_relu`` without sigmoid.  Others (``auto``/``auto_po2`` scales,
po2, binary, ternary, ...) are recorded and reported but not changed; for
``auto_po2`` weight quantizers the per-channel scale already tracks the range.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import keras
import numpy as np

from .base_quantizer import BaseQuantizer
from .qlayers import Clip
from .quantizers import quantized_bits, quantized_relu

__all__ = [
    "ValueSet",
    "NeuronPatterns",
    "analyze_value_sets",
    "analyze_patterns",
    "quantizer_value_grid",
    "convert_to_lut",
    "RangeStats",
    "QuantizerRange",
    "RangeReport",
    "RangeAnalysisCallback",
    "analyze_ranges",
    "optimize_bitwidths",
    "apply_report",
    "random_calibration_data",
    "fixed_point_format",
]

_POLICIES = ("keep_fractional", "keep_total")


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
@dataclasses.dataclass
class RangeStats:
    """Running min/max of a tensor stream.

    ``low``/``high`` are the (optionally quantile-clipped) extremes used for
    sizing; ``abs_min``/``abs_max`` are always the true extremes.
    """

    low: float = math.inf
    high: float = -math.inf
    abs_min: float = math.inf
    abs_max: float = -math.inf
    count: int = 0

    def update(self, x: np.ndarray, quantile: float | None = None) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            return
        lo, hi = float(x.min()), float(x.max())
        self.abs_min = min(self.abs_min, lo)
        self.abs_max = max(self.abs_max, hi)
        if quantile is not None and x.size > 1:
            lo = float(np.quantile(x, 1.0 - quantile))
            hi = float(np.quantile(x, quantile))
        self.low = min(self.low, lo)
        self.high = max(self.high, hi)
        self.count += x.size

    @property
    def valid(self) -> bool:
        return self.count > 0

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


# -----------------------------------------------------------------------------
# Fixed-point format helpers
# -----------------------------------------------------------------------------
def _is_scalar_number(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(
        v, bool
    )


def _sign_bits(q: BaseQuantizer) -> int:
    if isinstance(q, quantized_relu):
        return int(q.negative_slope != 0.0)
    return int(bool(q.keep_negative))


def _unsupported_reason(q: BaseQuantizer) -> str | None:
    """Return why ``q`` cannot be resized, or None if it can."""
    if isinstance(q, quantized_bits):
        if isinstance(q.alpha, str):
            return f"alpha='{q.alpha}' (scale tracks range)"
        if q.alpha is not None and not (_is_scalar_number(q.alpha) and q.alpha == 1):
            return "non-unit alpha"
    elif isinstance(q, quantized_relu):
        if q.use_sigmoid:
            return "use_sigmoid"
    else:
        return f"{type(q).__name__} not supported"
    if not (_is_scalar_number(q.bits) and _is_scalar_number(q.integer)):
        return "bits/integer are tensors"
    if int(q.bits) - _sign_bits(q) <= 0:
        return "binary (1-bit) quantizer"
    return None


def fixed_point_format(q: BaseQuantizer) -> dict | None:
    """Describe a supported quantizer as a fixed-point word.

    Returns a dict with ``bits``, ``integer`` (QKeras convention, excludes the
    sign bit), ``sign``, ``fractional``, ``lsb``, ``min``, ``max`` and the
    hls4ml type string, or None for unsupported quantizers.
    """
    if _unsupported_reason(q) is not None:
        return None
    bits, integer, sign = int(q.bits), int(q.integer), _sign_bits(q)
    frac = bits - sign - integer
    lsb = 2.0**-frac
    vmax = 2.0**integer - lsb
    if isinstance(q, quantized_relu):
        vmin = -q.negative_slope * 2.0**integer if sign else 0.0
    elif sign:
        vmin = -(2.0**integer) + (lsb if q.symmetric else 0.0)
    else:
        vmin = 0.0
    hls = (
        f"fixed<{bits},{integer + 1}>" if sign else f"ufixed<{bits},{integer}>"
    )
    return {
        "bits": bits,
        "integer": integer,
        "sign": sign,
        "fractional": frac,
        "lsb": lsb,
        "min": vmin,
        "max": vmax,
        # below-min values are mapped to 0 by design (ReLU / unsigned)
        "clips_negative": vmin == 0.0,
        "hls4ml": hls,
    }


def _fits(q: BaseQuantizer, integer: int, frac: int, low: float, high: float) -> bool:
    """True if [low, high] is representable (after rounding) with I/F."""
    lsb = 2.0**-frac
    if high > 2.0**integer - lsb / 2.0:
        return False
    if isinstance(q, quantized_relu):
        if q.negative_slope != 0.0 and low < 0:
            return q.negative_slope * (-low) <= 2.0**integer
        return True
    if q.keep_negative and low < 0:
        return -low <= 2.0**integer
    return True


def _required_integer(
    q: BaseQuantizer, stats: RangeStats, policy: str, min_integer: int
) -> int:
    """Smallest integer-bit count that holds the observed range."""
    bits, sign, integer = int(q.bits), _sign_bits(q), int(q.integer)
    frac_old = bits - sign - integer
    low, high = stats.low, stats.high
    if isinstance(q, quantized_relu) and q.negative_slope == 0.0:
        low = 0.0
    if isinstance(q, quantized_bits) and not q.keep_negative:
        low = 0.0
    neg = -low if low < 0 else 0.0
    if isinstance(q, quantized_relu) and q.negative_slope != 0.0:
        neg *= q.negative_slope
    peak = max(high, neg)
    if peak <= 0.0:
        return min_integer
    i = max(min_integer, math.ceil(math.log2(peak)))
    # bump until it fits once rounding to the LSB is taken into account
    for _ in range(64):
        frac = frac_old if policy == "keep_fractional" else bits - sign - i
        if _fits(q, i, frac, low, high):
            return i
        i += 1
    return i


# -----------------------------------------------------------------------------
# Report containers
# -----------------------------------------------------------------------------
@dataclasses.dataclass
class QuantizerRange:
    """Range analysis result for one quantizer."""

    layer: str
    role: str
    quantizer: str
    stats: RangeStats
    supported: bool
    reason: str | None = None
    old_bits: int | None = None
    old_integer: int | None = None
    new_bits: int | None = None
    new_integer: int | None = None
    saturated_fraction: float = 0.0
    applied: bool = False
    # not serialised: live handles used by apply_report
    _layer: Any = dataclasses.field(default=None, repr=False, compare=False)
    _quantizer: Any = dataclasses.field(default=None, repr=False, compare=False)
    _attrs: tuple = dataclasses.field(default=(), repr=False, compare=False)

    @property
    def changed(self) -> bool:
        return self.supported and (
            self.new_bits != self.old_bits or self.new_integer != self.old_integer
        )

    def as_dict(self) -> dict:
        return {
            "layer": self.layer,
            "role": self.role,
            "quantizer": self.quantizer,
            "low": self.stats.low,
            "high": self.stats.high,
            "abs_min": self.stats.abs_min,
            "abs_max": self.stats.abs_max,
            "supported": self.supported,
            "reason": self.reason,
            "old_bits": self.old_bits,
            "old_integer": self.old_integer,
            "new_bits": self.new_bits,
            "new_integer": self.new_integer,
            "saturated_fraction": self.saturated_fraction,
            "applied": self.applied,
            "hls4ml": (fixed_point_format(self._quantizer) or {}).get("hls4ml")
            if self._quantizer is not None
            else None,
        }


@dataclasses.dataclass
class RangeReport:
    """Collection of :class:`QuantizerRange` entries plus layer I/O ranges."""

    entries: list[QuantizerRange]
    layer_ranges: dict[str, dict[str, RangeStats]]
    policy: str
    epoch: int | None = None

    def changes(self) -> list[QuantizerRange]:
        return [e for e in self.entries if e.changed]

    def bit_delta(self) -> int:
        """Sum of (new_bits - old_bits) over all changed quantizers."""
        return sum(e.new_bits - e.old_bits for e in self.changes())

    def as_dicts(self) -> list[dict]:
        return [e.as_dict() for e in self.entries]

    def hls4ml_precisions(self) -> dict[str, dict[str, str]]:
        """{layer: {role: 'fixed<W,I>'}} for the *current* quantizer settings."""
        out: dict[str, dict[str, str]] = {}
        for e in self.entries:
            fmt = fixed_point_format(e._quantizer) if e._quantizer else None
            if fmt is not None:
                out.setdefault(e.layer, {})[e.role] = fmt["hls4ml"]
        return out

    def to_dataframe(self):
        import pandas as pd  # noqa: PLC0415 (optional dependency)

        return pd.DataFrame(self.as_dicts())

    def summary(self, only_changes: bool = False) -> str:
        rows = self.changes() if only_changes else self.entries
        hdr = (
            f"{'layer':<18} {'role':<12} {'observed range':>23} {'sat%':>6} "
            f"{'bits,int':>9} -> {'bits,int':<9} note"
        )
        out = [hdr, "-" * len(hdr)]
        for e in rows:
            rng = (
                f"[{e.stats.low:9.4g},{e.stats.high:9.4g}]"
                if e.stats.valid
                else "      (no data)"
            )
            old = f"{e.old_bits},{e.old_integer}" if e.supported else "-"
            new = f"{e.new_bits},{e.new_integer}" if e.supported else "-"
            note = e.reason or ("changed" if e.changed else "")
            if e.changed and e.applied:
                note = "applied"
            out.append(
                f"{e.layer[:18]:<18} {e.role[:12]:<12} {rng:>23} "
                f"{100 * e.saturated_fraction:6.2f} {old:>9} -> {new:<9} {note}"
            )
        out.append(f"total bit change over quantizers: {self.bit_delta():+d}")
        return "\n".join(out)

    def __str__(self) -> str:
        return self.summary()


# -----------------------------------------------------------------------------
# Calibration pass
# -----------------------------------------------------------------------------
class _Recorder:
    """Transparent proxy that records the input of a quantizer."""

    def __init__(self, quantizer, stats: RangeStats, quantile, fmt_holder):
        self._q = quantizer
        self._stats = stats
        self._quantile = quantile
        self._fmt = fmt_holder  # dict with 'min'/'max'/'lsb' or None
        self._n_sat = 0
        self._n = 0

    def __call__(self, x, *args, **kwargs):
        xn = keras.ops.convert_to_numpy(x)
        self._stats.update(xn, self._quantile)
        if self._fmt is not None:
            half = self._fmt["lsb"] / 2.0
            over = xn > self._fmt["max"] + half
            if not self._fmt["clips_negative"]:
                over |= xn < self._fmt["min"] - half
            self._n_sat += int(np.count_nonzero(over))
            self._n += xn.size
        return self._q(x, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._q, name)


def _iter_layers(model) -> list:
    if hasattr(model, "_flatten_layers"):
        return [
            layer
            for layer in model._flatten_layers(include_self=False, recursive=True)
            if not isinstance(layer, keras.Model)
            or not getattr(layer, "layers", None)
        ]
    return list(model.layers)


def _quantizer_attrs(layer) -> dict[int, tuple[BaseQuantizer, list[str]]]:
    """Map id(quantizer) -> (quantizer, [attribute names]) for one layer."""
    found: dict[int, tuple[BaseQuantizer, list[str]]] = {}
    for name, value in vars(layer).items():
        if name.startswith("__") or not isinstance(value, BaseQuantizer):
            continue
        found.setdefault(id(value), (value, []))[1].append(name)
    return found


def _role_name(attrs: Sequence[str]) -> str:
    names = sorted({a.replace("_quantizer_internal", "").replace("_quantizer", "")
                    for a in attrs}, key=len)
    return names[0]


def _as_batches(data, batch_size: int, max_batches: int | None) -> Iterable:
    """Yield model inputs (x only) from arrays, datasets or iterables."""

    def _x_only(item):
        if isinstance(item, tuple) and len(item) in (2, 3):
            return item[0]
        return item

    if data is None:
        raise ValueError("No calibration data given.")
    if callable(data) and not hasattr(data, "__iter__"):
        data = data()
    is_array = isinstance(data, np.ndarray) or (
        isinstance(data, (list, tuple, dict))
        and all(
            isinstance(v, np.ndarray)
            for v in (data.values() if isinstance(data, dict) else data)
        )
    )
    if is_array:
        flat = keras.tree.flatten(data)
        n = flat[0].shape[0]
        n_batches = math.ceil(n / batch_size)
        if max_batches is not None:
            n_batches = min(n_batches, max_batches)
        for b in range(n_batches):
            sl = slice(b * batch_size, (b + 1) * batch_size)
            yield keras.tree.map_structure(lambda a, s=sl: a[s], data)
        return
    for i, item in enumerate(data):
        if max_batches is not None and i >= max_batches:
            break
        yield _x_only(item)


def _layer_filter(layer, include, exclude) -> bool:
    if getattr(layer, "range_analysis", True) is False:  # per-layer opt-out
        return False
    if include is not None and layer.name not in include:
        return False
    if exclude is not None and layer.name in exclude:
        return False
    return True


def analyze_ranges(
    model,
    data,
    *,
    batch_size: int = 256,
    max_batches: int | None = 8,
    quantile: float | None = None,
    record_layer_io: bool = True,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> RangeReport:
    """Run an eager calibration pass and collect value ranges.

    Args:
      model: a built Keras model containing QKeras layers.
      data: calibration inputs: numpy array(s), a ``tf.data.Dataset``, a
        generator/iterable yielding ``x`` or ``(x, y[, w])`` batches, or a
        zero-argument callable returning one of those.
      batch_size: batch size when ``data`` is an array.
      max_batches: upper bound on the number of batches used (None = all).
      quantile: if set (e.g. 0.9999), size from this quantile per batch
        instead of the absolute min/max, which ignores rare outliers.
      record_layer_io: also record every layer's input and output range.
      include / exclude: optional layer-name filters.  Layers with attribute
        ``range_analysis = False`` are always skipped.

    Returns:
      A :class:`RangeReport` whose entries have ``new_bits == old_bits``
      (no proposal yet; see :func:`propose_bitwidths`).
    """
    entries: list[QuantizerRange] = []
    recorders: list[tuple[Any, str, Any, _Recorder]] = []
    layer_ranges: dict[str, dict[str, RangeStats]] = {}
    patched_calls = []

    for layer in _iter_layers(model):
        if not _layer_filter(layer, include, exclude):
            continue
        for q, attrs in _quantizer_attrs(layer).values():
            reason = _unsupported_reason(q)
            fmt = fixed_point_format(q)
            stats = RangeStats()
            rec = _Recorder(q, stats, quantile, fmt)
            entry = QuantizerRange(
                layer=layer.name,
                role=_role_name(attrs),
                quantizer=str(q),
                stats=stats,
                supported=reason is None,
                reason=reason,
                old_bits=int(q.bits) if reason is None else None,
                old_integer=int(q.integer) if reason is None else None,
                _layer=layer,
                _quantizer=q,
                _attrs=tuple(attrs),
            )
            entry.new_bits, entry.new_integer = entry.old_bits, entry.old_integer
            entries.append(entry)
            for a in attrs:
                recorders.append((layer, a, q, rec))
        if record_layer_io:
            io = {"input": RangeStats(), "output": RangeStats()}
            layer_ranges[layer.name] = io
            orig_call = layer.call

            def _wrapped(*args, __orig=orig_call, __io=io, **kwargs):
                if args:
                    for t in keras.tree.flatten(args[0]):
                        __io["input"].update(keras.ops.convert_to_numpy(t))
                out = __orig(*args, **kwargs)
                for t in keras.tree.flatten(out):
                    __io["output"].update(keras.ops.convert_to_numpy(t))
                return out

            layer.call = _wrapped
            patched_calls.append(layer)

    try:
        for layer, attr, _q, rec in recorders:
            object.__setattr__(layer, attr, rec)
        for x in _as_batches(data, batch_size, max_batches):
            model(x, training=False)
    finally:
        for layer, attr, q, _rec in recorders:
            object.__setattr__(layer, attr, q)
        for layer in patched_calls:
            try:
                del layer.call
            except AttributeError:
                pass

    by_q = {}
    for _layer, _attr, q, rec in recorders:
        by_q[id(q)] = rec
    for e in entries:
        rec = by_q[id(e._quantizer)]
        e.saturated_fraction = rec._n_sat / rec._n if rec._n else 0.0

    return RangeReport(entries=entries, layer_ranges=layer_ranges, policy="none")


# -----------------------------------------------------------------------------
# Proposal and application
# -----------------------------------------------------------------------------
def propose_bitwidths(
    report: RangeReport,
    *,
    policy: str = "keep_fractional",
    headroom: int = 0,
    min_bits: int = 2,
    max_bits: int = 32,
    min_integer: int = -16,
    max_integer: int = 32,
    adjust_weights: bool = True,
    adjust_activations: bool = True,
    allow_shrink: bool = True,
    allow_grow: bool = True,
) -> RangeReport:
    """Fill ``new_bits`` / ``new_integer`` of every supported entry in place.

    ``headroom`` adds that many integer bits on top of the observed range.
    Roles ``kernel``/``bias``/``depthwise``/``pointwise``/``recurrent``/... are
    treated as weights; ``activation`` (incl. QActivation layers) as
    activations.
    """
    if policy not in _POLICIES:
        raise ValueError(f"policy must be one of {_POLICIES}, got {policy!r}")
    report.policy = policy
    for e in report.entries:
        if not e.supported:
            continue
        is_act = e.role == "activation"
        if (is_act and not adjust_activations) or (not is_act and not adjust_weights):
            e.reason = "skipped by role filter"
            continue
        if not e.stats.valid:
            e.reason = "no data reached quantizer"
            continue
        if e.stats.abs_min == 0.0 and e.stats.abs_max == 0.0:
            e.reason = "all values zero"
            continue
        q = e._quantizer
        sign = _sign_bits(q)
        i_req = _required_integer(q, e.stats, policy, min_integer) + headroom
        i_new = int(min(max(i_req, min_integer), max_integer))
        if i_new > e.old_integer and not allow_grow:
            i_new = e.old_integer
        if i_new < e.old_integer and not allow_shrink:
            i_new = e.old_integer
        if policy == "keep_fractional":
            frac = e.old_bits - sign - e.old_integer
            b_new = sign + i_new + frac
        else:
            b_new = e.old_bits
        b_new = int(min(max(b_new, min_bits, sign + 1), max_bits))
        e.new_bits, e.new_integer = b_new, i_new
    return report


def _invalidate_compiled_functions(model) -> None:
    """Force a retrace so new Python-level bit-widths take effect."""
    if getattr(model, "compiled", False) and getattr(model, "optimizer", None):
        if getattr(model, "train_function", None) is not None:
            model.make_train_function(force=True)
    for attr in ("test_function", "predict_function"):
        if hasattr(model, attr):
            setattr(model, attr, None)


def _update_weight_constraint(layer, attrs: Sequence[str], q) -> None:
    """Re-derive QKeras' auto Clip constraint from the resized quantizer."""
    max_value = max(1.0, float(q.max())) if hasattr(q, "max") else 1.0
    for a in attrs:
        if not a.endswith("_quantizer_internal") and not a.endswith("_quantizer"):
            continue
        prefix = a.replace("_quantizer_internal", "").replace("_quantizer", "")
        c = getattr(layer, f"{prefix}_constraint", None)
        if isinstance(c, Clip):
            c.min_value, c.max_value = -max_value, max_value


def apply_report(model, report: RangeReport) -> RangeReport:
    """Write the proposed bit-widths into the model's quantizers."""
    any_change = False
    for e in report.changes():
        q = e._quantizer
        q.bits = int(e.new_bits)
        q.integer = int(e.new_integer)
        if e.role != "activation":
            _update_weight_constraint(e._layer, e._attrs, q)
        e.applied = True
        any_change = True
    if any_change:
        _invalidate_compiled_functions(model)
    return report


def _snapshot(model) -> dict[int, tuple[int, int]]:
    snap = {}
    for layer in _iter_layers(model):
        for q, _attrs in _quantizer_attrs(layer).values():
            if _unsupported_reason(q) is None:
                snap[id(q)] = (int(q.bits), int(q.integer))
    return snap


def _finalize(report: RangeReport, original: dict[int, tuple[int, int]]) -> None:
    """Rebase a report onto the widths the model had before adaptation."""
    for e in report.entries:
        if not e.supported:
            continue
        q = e._quantizer
        e.old_bits, e.old_integer = original.get(id(q), (e.old_bits, e.old_integer))
        e.new_bits, e.new_integer = int(q.bits), int(q.integer)
        e.applied = e.changed


def _adapt(
    model,
    data,
    *,
    analyze_kwargs: dict,
    propose_kwargs: dict,
    max_iterations: int,
    dry_run: bool,
    first_pass_hook: Callable[[RangeReport], None] | None = None,
) -> RangeReport:
    """analyze -> propose -> apply until nothing changes.

    One pass sizes each quantizer on values that upstream quantizers have
    already clipped; when an upstream quantizer grows, downstream ranges grow
    too.  Iterating propagates this.  Passes after the first are grow-only so
    the result is monotone and terminates.
    """
    original = _snapshot(model)
    report = None
    for it in range(max(1, max_iterations)):
        report = analyze_ranges(model, data, **analyze_kwargs)
        kw = dict(propose_kwargs)
        if it > 0:
            kw["allow_shrink"] = False
        propose_bitwidths(report, **kw)
        if it == 0 and first_pass_hook is not None:
            first_pass_hook(report)
        if dry_run or not report.changes():
            break
        apply_report(model, report)
    else:
        report = analyze_ranges(model, data, **analyze_kwargs)
        propose_bitwidths(report, **{**propose_kwargs, "allow_shrink": False})
    if not dry_run:
        _finalize(report, original)
    return report


# -----------------------------------------------------------------------------
# Random calibration data
# -----------------------------------------------------------------------------
def random_calibration_data(
    model,
    n_samples: int = 1024,
    *,
    low: float | Sequence[float] = -1.0,
    high: float | Sequence[float] = 1.0,
    distribution: str = "uniform",
    seed: int | None = 0,
):
    """Random inputs matching ``model.inputs``.

    ``distribution="uniform"`` draws from [low, high] (worst-case style
    coverage of the input box); ``"normal"`` draws N(mean=(low+high)/2,
    std=(high-low)/4) clipped to [low, high].  ``low``/``high`` may be
    per-input sequences for multi-input models.
    """
    rng = np.random.default_rng(seed)
    inputs = model.inputs if getattr(model, "inputs", None) else None
    if not inputs:
        raise ValueError("Model has no symbolic inputs; build it on an Input first.")
    lows = low if isinstance(low, (list, tuple)) else [low] * len(inputs)
    highs = high if isinstance(high, (list, tuple)) else [high] * len(inputs)
    arrays = []
    for t, lo, hi in zip(inputs, lows, highs):
        shape = (n_samples,) + tuple(d if d is not None else 1 for d in t.shape[1:])
        if distribution == "uniform":
            a = rng.uniform(lo, hi, size=shape)
        elif distribution == "normal":
            a = np.clip(rng.normal((lo + hi) / 2, (hi - lo) / 4, size=shape), lo, hi)
        else:
            raise ValueError(f"unknown distribution {distribution!r}")
        arrays.append(a.astype("float32"))
    return arrays[0] if len(arrays) == 1 else arrays


# -----------------------------------------------------------------------------
# Public one-shot API
# -----------------------------------------------------------------------------
def optimize_bitwidths(
    model,
    data=None,
    *,
    random_samples: int = 1024,
    random_low: float | Sequence[float] = -1.0,
    random_high: float | Sequence[float] = 1.0,
    batch_size: int = 256,
    max_batches: int | None = 8,
    quantile: float | None = None,
    policy: str = "keep_fractional",
    headroom: int = 0,
    min_bits: int = 2,
    max_bits: int = 32,
    min_integer: int = -16,
    max_integer: int = 32,
    adjust_weights: bool = True,
    adjust_activations: bool = True,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    max_iterations: int = 8,
    dry_run: bool = False,
    verbose: int = 1,
) -> RangeReport:
    """Post-training range analysis and bit-width adjustment.

    If ``data`` is None, ``random_samples`` uniform random inputs in
    ``[random_low, random_high]`` are used.  With ``dry_run=True`` nothing is
    changed; the report still carries the proposal.  Calibration is
    repeated (at most ``max_iterations`` passes) until no quantizer changes,
    so growth of an upstream quantizer reaches the layers behind it.  The
    returned report holds the original widths as ``old_*``, the final widths
    as ``new_*`` and the ranges/saturation measured with the final widths.
    """
    if data is None:
        data = random_calibration_data(
            model, random_samples, low=random_low, high=random_high
        )
    report = _adapt(
        model,
        data,
        analyze_kwargs=dict(
            batch_size=batch_size,
            max_batches=max_batches,
            quantile=quantile,
            include=include,
            exclude=exclude,
        ),
        propose_kwargs=dict(
            policy=policy,
            headroom=headroom,
            min_bits=min_bits,
            max_bits=max_bits,
            min_integer=min_integer,
            max_integer=max_integer,
            adjust_weights=adjust_weights,
            adjust_activations=adjust_activations,
        ),
        max_iterations=max_iterations,
        dry_run=dry_run,
    )
    if verbose:
        print(report.summary())
    return report


# -----------------------------------------------------------------------------
# Training-time callback
# -----------------------------------------------------------------------------
class RangeAnalysisCallback(keras.callbacks.Callback):
    """Adapt quantizer bit-widths to observed ranges during ``fit``.

    Every ``frequency`` epochs inside ``[start_epoch, stop_epoch)`` the
    callback runs :func:`analyze_ranges` on ``calibration_data`` and updates
    the quantizers.  Growth (a range that no longer fits) is applied at once;
    shrinking is applied only after ``patience`` consecutive checks agree, so
    the width does not oscillate while weights are still moving.

    Turn it off/on at any time with ``callback.enabled = False/True``, or give
    a window with ``start_epoch`` / ``stop_epoch``.  ``dry_run=True`` only
    logs.  All proposals are kept in ``callback.history``.

    Args:
      calibration_data: inputs for the calibration pass (see
        :func:`analyze_ranges`), or ``"random"`` to draw
        ``random_samples`` uniform inputs in ``[random_low, random_high]``.
      Remaining arguments as in :func:`optimize_bitwidths`.
    """

    def __init__(
        self,
        calibration_data="random",
        *,
        frequency: int = 1,
        start_epoch: int = 0,
        stop_epoch: int | None = None,
        enabled: bool = True,
        patience: int = 2,
        policy: str = "keep_fractional",
        headroom: int = 0,
        quantile: float | None = None,
        min_bits: int = 2,
        max_bits: int = 32,
        min_integer: int = -16,
        max_integer: int = 32,
        adjust_weights: bool = True,
        adjust_activations: bool = True,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        batch_size: int = 256,
        max_batches: int | None = 4,
        max_iterations: int = 8,
        random_samples: int = 1024,
        random_low: float = -1.0,
        random_high: float = 1.0,
        dry_run: bool = False,
        verbose: int = 1,
        log_fn: Callable[[str], None] = print,
    ):
        super().__init__()
        if policy not in _POLICIES:
            raise ValueError(f"policy must be one of {_POLICIES}, got {policy!r}")
        self.calibration_data = calibration_data
        self.frequency = max(1, int(frequency))
        self.start_epoch = start_epoch
        self.stop_epoch = stop_epoch
        self.enabled = enabled
        self.patience = max(1, int(patience))
        self.policy = policy
        self.headroom = headroom
        self.quantile = quantile
        self.min_bits, self.max_bits = min_bits, max_bits
        self.min_integer, self.max_integer = min_integer, max_integer
        self.adjust_weights = adjust_weights
        self.adjust_activations = adjust_activations
        self.include, self.exclude = include, exclude
        self.batch_size, self.max_batches = batch_size, max_batches
        self.max_iterations = max_iterations
        self.random_samples = random_samples
        self.random_low, self.random_high = random_low, random_high
        self.dry_run = dry_run
        self.verbose = verbose
        self.log_fn = log_fn
        self.history: list[RangeReport] = []
        self._shrink_votes: dict[tuple[str, str], int] = {}
        self._random_cache = None

    def _active(self, epoch: int) -> bool:
        if not self.enabled or epoch < self.start_epoch:
            return False
        if self.stop_epoch is not None and epoch >= self.stop_epoch:
            return False
        return (epoch - self.start_epoch) % self.frequency == self.frequency - 1

    def _data(self):
        if isinstance(self.calibration_data, str):
            if self.calibration_data != "random":
                raise ValueError("calibration_data string must be 'random'")
            if self._random_cache is None:
                self._random_cache = random_calibration_data(
                    self.model,
                    self.random_samples,
                    low=self.random_low,
                    high=self.random_high,
                )
            return self._random_cache
        return self.calibration_data

    def run(self, epoch: int | None = None) -> RangeReport:
        """Run one analysis/adaptation step now (also usable outside fit)."""
        model = self.model
        if hasattr(model, "jax_state_sync"):
            model.jax_state_sync()

        def _hysteresis(report: RangeReport) -> None:
            # grow now, shrink only after `patience` agreeing checks
            for e in report.entries:
                key = (e.layer, e.role)
                if not e.changed:
                    self._shrink_votes.pop(key, None)
                    continue
                if e.new_integer < e.old_integer:
                    votes = self._shrink_votes.get(key, 0) + 1
                    self._shrink_votes[key] = votes
                    if votes < self.patience:
                        e.reason = f"shrink pending ({votes}/{self.patience})"
                        e.new_bits, e.new_integer = e.old_bits, e.old_integer
                        continue
                self._shrink_votes.pop(key, None)

        report = _adapt(
            model,
            self._data(),
            analyze_kwargs=dict(
                batch_size=self.batch_size,
                max_batches=self.max_batches,
                quantile=self.quantile,
                include=self.include,
                exclude=self.exclude,
            ),
            propose_kwargs=dict(
                policy=self.policy,
                headroom=self.headroom,
                min_bits=self.min_bits,
                max_bits=self.max_bits,
                min_integer=self.min_integer,
                max_integer=self.max_integer,
                adjust_weights=self.adjust_weights,
                adjust_activations=self.adjust_activations,
            ),
            max_iterations=self.max_iterations,
            dry_run=self.dry_run,
            first_pass_hook=_hysteresis,
        )
        report.epoch = epoch
        self.history.append(report)
        if self.verbose and (report.changes() or self.verbose > 1):
            tag = f"epoch {epoch + 1}" if epoch is not None else "manual"
            self.log_fn(
                f"\n[RangeAnalysis {tag}]"
                + (" (dry run)" if self.dry_run else "")
                + "\n"
                + report.summary(only_changes=self.verbose < 2)
            )
        return report

    def on_epoch_end(self, epoch, logs=None):
        if not self._active(epoch):
            return
        report = self.run(epoch)
        if logs is not None:
            logs["range_analysis_bit_delta"] = float(report.bit_delta())


# -----------------------------------------------------------------------------
# Value sets and input patterns (used by the LUT conversion)
# -----------------------------------------------------------------------------
def quantizer_value_grid(q, max_values: int = 1 << 16) -> np.ndarray | None:
    """All values a quantizer can output, sorted, or None if unknown/too many.

    Fixed-point quantizers (``quantized_bits`` with alpha 1/None,
    ``quantized_relu``) give their full grid; other quantizers are probed
    (``binary``/``bernoulli``/``ternary`` style outputs)."""
    if q is None:
        return None
    fmt = fixed_point_format(q) if isinstance(q, BaseQuantizer) else None
    if fmt is not None:
        n = int(round((fmt["max"] - fmt["min"]) / fmt["lsb"])) + 1
        if n > max_values:
            return None
        return (fmt["min"] + fmt["lsb"] * np.arange(n)).astype(np.float64)
    name = type(q).__name__
    if name in ("binary", "stochastic_binary", "ternary", "stochastic_ternary"):
        probe = np.linspace(-4.0, 4.0, 257, dtype=np.float32)
        try:
            v = np.unique(keras.ops.convert_to_numpy(q(probe)).astype(np.float64))
        except Exception:  # pragma: no cover - exotic quantizer
            return None
        return v
    return None


@dataclasses.dataclass
class ValueSet:
    """Finite set of values a tensor takes; defines its Boolean encoding.

    ``code(x)`` maps a value to its index in ``values`` (nearest value for
    unseen inputs), which is encoded on ``bits`` wires, LSB first.
    """

    name: str
    values: np.ndarray                 # sorted float64
    observed: np.ndarray = None        # sorted observed values (subset)
    source: str = "observed"           # "observed" | "quantizer"
    overflow: bool = False             # more distinct values than max_values

    @property
    def size(self) -> int:
        return int(len(self.values))

    @property
    def bits(self) -> int:
        return 0 if self.size <= 1 else int(math.ceil(math.log2(self.size)))

    def code(self, x) -> np.ndarray:
        x = np.asarray(x, np.float64)
        v = self.values
        i = np.clip(np.searchsorted(v, x), 1, max(1, len(v) - 1))
        if len(v) == 1:
            return np.zeros(x.shape, np.int64)
        lo, hi = v[i - 1], v[i]
        return np.where(np.abs(x - lo) <= np.abs(hi - x), i - 1, i).astype(np.int64)

    def decode(self, c) -> np.ndarray:
        return self.values[np.asarray(c, np.int64)]

    def is_uniform(self) -> bool:
        if self.size < 3:
            return True
        d = np.diff(self.values)
        return bool(np.allclose(d, d[0]))

    def as_dict(self) -> dict:
        return {"name": self.name, "size": self.size, "bits": self.bits,
                "source": self.source, "min": float(self.values[0]),
                "max": float(self.values[-1]), "overflow": self.overflow,
                "n_observed": int(len(self.observed)) if self.observed is not None else None}


def _activation_quantizer(layer):
    act = getattr(layer, "activation", None)
    if isinstance(act, BaseQuantizer):
        return act
    return None


def _tensor_model(model, tensors):
    inp = model.inputs[0] if len(model.inputs) == 1 else model.inputs
    return keras.Model(inp, tensors)


def analyze_value_sets(
    model,
    data,
    *,
    layer_names: Sequence[str] | None = None,
    grid: str = "range",
    max_values: int = 4096,
    batch_size: int = 256,
    max_batches: int | None = None,
) -> dict[str, ValueSet]:
    """Record the distinct values at the model input and at the output of every
    layer with a quantised activation (``QActivation`` or ``activation=<quantizer>``).

    Args:
      grid: ``"range"`` (default) -- the quantizer grid restricted to the
        observed ``[min, max]`` (uniform codes, unused MSBs cost no wires);
        ``"observed"`` -- only the observed values (fewest wires, non-uniform
        codes); ``"quantizer"`` -- the full output grid of the quantizer
        (exact for any input, most wires).  Tensors without a known grid fall
        back to the observed values.
      max_values: tensors with more distinct values are flagged ``overflow``
        (not convertible to Boolean wires exactly).

    Returns: dict layer name -> :class:`ValueSet`; the model input is stored
    under ``"__input__"``.
    """
    if grid not in ("range", "observed", "quantizer"):
        raise ValueError("grid must be 'range', 'observed' or 'quantizer'")
    cand = []
    for layer in _iter_layers(model):
        if layer_names is not None and layer.name not in layer_names:
            continue
        if _activation_quantizer(layer) is not None:
            cand.append(layer)
    tm = _tensor_model(model, [model.inputs[0]] + [l.output for l in cand])
    seen = [set() for _ in range(len(cand) + 1)]
    over = [False] * (len(cand) + 1)
    for x in _as_batches(data, batch_size, max_batches):
        outs = tm(x, training=False)
        outs = outs if isinstance(outs, (list, tuple)) else [outs]
        for i, t in enumerate(outs):
            if over[i]:
                continue
            u = np.unique(keras.ops.convert_to_numpy(t).astype(np.float64))
            seen[i].update(u.tolist())
            if len(seen[i]) > max_values:
                over[i] = True
    res = {}
    names = ["__input__"] + [l.name for l in cand]
    quants = [None] + [_activation_quantizer(l) for l in cand]
    for name, q, obs, ov in zip(names, quants, seen, over):
        observed = np.array(sorted(obs), np.float64)
        values, source = observed, "observed"
        g = quantizer_value_grid(q) if (grid != "observed" and q is not None) else None
        if g is not None and grid == "quantizer":
            values, source = g, "quantizer"
        elif g is not None and len(observed):
            tol = 1e-9 * max(1.0, float(np.abs(g).max()))
            sel = (g >= observed[0] - tol) & (g <= observed[-1] + tol)
            values, source = g[sel], "range"
        res[name] = ValueSet(name, values, observed, source, ov)
    return res


@dataclasses.dataclass
class NeuronPatterns:
    """Observed input patterns of the neurons (units / filters) of one layer.

    ``taps[u]`` are the flat input indices with a non-zero quantised weight
    (Dense: input feature; Conv2D: index into the flattened ``(kh, kw, cin)``
    window); ``bits[u]`` the number of input wires (sum of the code widths);
    ``patterns[u]`` the sorted observed pattern integers (code of tap ``i``
    occupies bits ``[off_i, off_i + w_i)``), or None if ``bits[u] > max_bits``.
    """

    layer: str
    kind: str                          # "dense" | "conv2d"
    taps: list
    bits: list
    patterns: list
    n_samples: int = 0

    def coverage(self, u: int) -> float | None:
        if self.patterns[u] is None:
            return None
        return len(self.patterns[u]) / float(1 << self.bits[u])


def _quantized_kernel(layer) -> np.ndarray:
    k = layer.kernel
    q = getattr(layer, "kernel_quantizer_internal", None)
    kq = q(k) if q else k
    return keras.ops.convert_to_numpy(kq).astype(np.float64)


def _pack_patterns(codes, widths):
    """codes (M, F) int, widths (F,) -> pattern ints (M,)."""
    off = np.concatenate([[0], np.cumsum(widths)[:-1]]).astype(np.int64)
    return (codes.astype(np.int64) << off[None, :]).sum(1)


def _conv_patches(x, kh, kw, sh, sw):
    """x (N, H, W, C) -> (N, OH, OW, kh*kw*C) valid patches, (kh, kw, c) order."""
    n, h, w, c = x.shape
    oh, ow = (h - kh) // sh + 1, (w - kw) // sw + 1
    sn, sh_, sw_, sc = x.strides
    view = np.lib.stride_tricks.as_strided(
        x, (n, oh, ow, kh, kw, c), (sn, sh_ * sh, sw_ * sw, sh_, sw_, sc), writeable=False)
    return view.reshape(n, oh, ow, kh * kw * c)


def analyze_patterns(
    model,
    data,
    value_sets: dict[str, ValueSet],
    *,
    layer_names: Sequence[str] | None = None,
    max_bits: int = 16,
    batch_size: int = 256,
    max_batches: int | None = None,
) -> dict[str, NeuronPatterns]:
    """Record, for every ``QDense``/``QConv2D`` layer whose input is encoded by
    a :class:`ValueSet`, the input patterns each neuron sees on ``data``.

    The input of a layer must be the model input or the output of a layer
    present in ``value_sets`` (possibly through Flatten/Reshape/InputLayer).
    """
    from .qconvolutional import QConv2D
    from .qlayers import QDense

    def _source(layer):
        t = layer.input
        node = getattr(t, "_keras_history", None)
        src = node[0] if node is not None else None
        while src is not None and isinstance(src, (keras.layers.Flatten, keras.layers.Reshape)):
            node = getattr(src.input, "_keras_history", None)
            src = node[0] if node is not None else None
        if src is None or isinstance(src, keras.layers.InputLayer):
            return "__input__"
        return src.name

    targets = []
    for layer in _iter_layers(model):
        if not isinstance(layer, (QDense, QConv2D)):
            continue
        if layer_names is not None and layer.name not in layer_names:
            continue
        src = _source(layer)
        if src not in value_sets:
            continue
        vs = value_sets[src]
        kq = _quantized_kernel(layer)
        if isinstance(layer, QConv2D):
            if layer.padding != "valid" or tuple(layer.dilation_rate) != (1, 1) or layer.groups != 1:
                continue
            kind = "conv2d"
            kflat = kq.reshape(-1, kq.shape[-1])
        else:
            kind = "dense"
            kflat = kq
        taps = [np.flatnonzero(kflat[:, u] != 0) for u in range(kflat.shape[1])]
        bits = [int(len(t) * vs.bits) for t in taps]
        targets.append((layer, src, vs, kind, taps, bits))

    res = {l.name: NeuronPatterns(l.name, kind, taps, bits,
                                  [set() if b <= max_bits else None for b in bits])
           for l, _, _, kind, taps, bits in targets}
    if not targets:
        return res
    tm = _tensor_model(model, [l.input for l, *_ in targets])
    for x in _as_batches(data, batch_size, max_batches):
        outs = tm(x, training=False)
        outs = outs if isinstance(outs, (list, tuple)) else [outs]
        for (layer, src, vs, kind, taps, bits), t in zip(targets, outs):
            a = keras.ops.convert_to_numpy(t)
            codes = vs.code(a)
            rec = res[layer.name]
            if kind == "conv2d":
                kh, kw = layer.kernel_size
                sh, sw = layer.strides
                p = _conv_patches(np.ascontiguousarray(codes), kh, kw, sh, sw)
                flat = p.reshape(-1, p.shape[-1])
            else:
                flat = codes.reshape(len(codes), -1)
            rec.n_samples += len(flat)
            w = np.full(flat.shape[1], vs.bits, np.int64)
            for u, tp in enumerate(taps):
                if rec.patterns[u] is None:
                    continue
                if len(tp) == 0:
                    rec.patterns[u].add(0)
                    continue
                rec.patterns[u].update(np.unique(_pack_patterns(flat[:, tp], w[tp])).tolist())
    for rec in res.values():
        rec.patterns = [None if p is None else np.array(sorted(p), np.int64) for p in rec.patterns]
    return res


def convert_to_lut(model, data, **kwargs):
    """Post-training range analysis -> LUT network (teacher -> student).

    Thin wrapper around :func:`qkeras.logic.convert.convert_to_lut`; see there
    for the arguments.  Returns a :class:`qkeras.logic.convert.Conversion`.
    """
    from .logic.convert import convert_to_lut as _convert

    return _convert(model, data, **kwargs)
