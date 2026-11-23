"""Post-training conversion of a quantised QKeras network (teacher) into a LUT
network (student), driven by range analysis.

Every quantised activation takes a finite set of values
(:func:`qkeras.range_analysis.analyze_value_sets`); each value gets a code of
``bits`` wires.  A neuron whose non-zero-weight inputs need at most
``max_exact_bits`` wires in total is a Boolean function of those wires and is
converted **exactly**: all ``2**bits`` input codes are pushed through the
teacher's own quantised arithmetic (kernel/bias quantizers, activation
quantizers), giving one truth table per output bit.  Input codes never seen on
the calibration data (:func:`qkeras.range_analysis.analyze_patterns`) and codes
that do not exist (code >= number of values) are marked as don't-cares, which
the backend may use to shrink the logic.

Wider neurons (e.g. a dense layer over all 784 MNIST pixels) cannot be
enumerated.  They are replaced by a **distilled** LUTDense block (the student)
that learns the teacher's output codes bit by bit (layer-wise hints), and is
then fine-tuned end-to-end through the (frozen, exactly converted) rest of the
teacher with a knowledge-distillation loss.

The final layer can be a linear ``QDense`` (logits): if exact, its integer
logits are emitted as ``words`` (argmax-exact w.r.t. the teacher); if
distilled, the student ends in a ``GroupSum`` head.

To train teachers that convert exactly, constrain the fan-in of every neuron
with :class:`FanInConstraint` (top-``fan_in`` weights per neuron, re-applied
after every optimiser step; :class:`FanInScheduler` anneals the fan-in).
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any

import keras
import numpy as np
from keras import ops

from ..qconvolutional import QConv2D
from ..qlayers import QActivation, QDense
from ..range_analysis import (ValueSet, _quantized_kernel, analyze_patterns,
                              analyze_value_sets)
from .layers import GroupSum, LUTDense
from .netlist import LUT, Netlist, simulate

__all__ = ["FanInConstraint", "FanInScheduler", "DistillConfig", "Conversion",
           "convert_to_lut", "fan_in_report"]


# =============================================================================
# Fan-in constraint (trains teachers that convert exactly)
# =============================================================================
@keras.saving.register_keras_serializable(package="qkeras")
class FanInConstraint(keras.constraints.Constraint):
    """Keep the ``fan_in`` largest-|w| inputs of every output neuron, zero the rest.

    Works for Dense kernels ``(n_in, units)`` and Conv kernels
    ``(kh, kw, cin, filters)`` (fan-in counted over the whole window).  The
    selection is re-evaluated after every step, so pruned weights can come
    back if their gradients make them large (dynamic sparsity).  ``clip``
    optionally clips to ``[-clip, clip]`` like QKeras' default constraint.
    """

    def __init__(self, fan_in: int, clip: float | None = None):
        self.fan_in = int(fan_in)
        self.clip = clip

    def __call__(self, w):
        shape = w.shape
        flat = ops.reshape(w, (-1, shape[-1]))
        n = flat.shape[0]
        if self.fan_in < n:
            a = ops.abs(ops.transpose(flat))                      # (units, n_in)
            kth = ops.top_k(a, k=self.fan_in, sorted=True)[0][:, -1:]
            mask = ops.cast(a >= kth, w.dtype)
            flat = flat * ops.transpose(mask)
        if self.clip is not None:
            flat = ops.clip(flat, -self.clip, self.clip)
        return ops.reshape(flat, shape)

    def get_config(self):
        return {"fan_in": self.fan_in, "clip": self.clip}


class FanInScheduler(keras.callbacks.Callback):
    """Anneal ``FanInConstraint.fan_in`` from ``start`` to the final value.

    ``schedule``: dict ``{layer_name: (start, end)}``; the fan-in decreases
    geometrically between ``begin_epoch`` and ``end_epoch``.  The compiled
    train function is rebuilt when a value changes (fan-in is static).
    """

    def __init__(self, schedule: dict, begin_epoch: int = 0, end_epoch: int = 5):
        super().__init__()
        self.schedule = schedule
        self.begin, self.end = begin_epoch, max(begin_epoch + 1, end_epoch)

    def _value(self, start, end, epoch):
        if epoch <= self.begin:
            return start
        if epoch >= self.end:
            return end
        t = (epoch - self.begin) / (self.end - self.begin)
        return int(round(start * (end / start) ** t))

    def on_epoch_begin(self, epoch, logs=None):
        changed = False
        for layer in self.model.layers:
            if layer.name not in self.schedule:
                continue
            c = getattr(layer, "kernel_constraint", None)
            if not isinstance(c, FanInConstraint):
                continue
            v = self._value(*self.schedule[layer.name], epoch)
            if v != c.fan_in:
                c.fan_in = v
                changed = True
        if changed:
            self.model.make_train_function(force=True)
            self.model.test_function = None
            self.model.predict_function = None

    def on_train_end(self, logs=None):
        """Guarantee the final fan-in: set it and project the kernels once more."""
        for layer in self.model.layers:
            if layer.name not in self.schedule:
                continue
            c = getattr(layer, "kernel_constraint", None)
            if isinstance(c, FanInConstraint):
                c.fan_in = self.schedule[layer.name][1]
                layer.kernel.assign(c(layer.kernel))
        self.model.test_function = None
        self.model.predict_function = None


def fan_in_report(model) -> dict:
    """Max/mean number of non-zero quantised weights per neuron, per layer."""
    out = {}
    for layer in model.layers:
        if isinstance(layer, (QDense, QConv2D)):
            k = _quantized_kernel(layer)
            nz = (k.reshape(-1, k.shape[-1]) != 0).sum(0)
            out[layer.name] = {"max": int(nz.max()), "mean": float(nz.mean())}
    return out


# =============================================================================
# Teacher parsing
# =============================================================================
_PASS = (keras.layers.InputLayer, keras.layers.Flatten, keras.layers.Reshape,
         keras.layers.Dropout)


@dataclasses.dataclass
class _Stage:
    layer: Any                       # QDense / QConv2D
    post: list                       # callables applied after the layer (activations)
    out_name: str | None             # value-set key of the stage output (None = logits)
    in_shape: tuple                  # shape of the layer input (without batch)
    out_shape: tuple


class _FrozenBN:
    """BatchNormalization in inference mode (moving statistics), as a post op."""

    def __init__(self, layer):
        self.layer = layer

    def __call__(self, t):
        return self.layer(t, training=False)


def _parse_teacher(model):
    """Return (input value-set key, stages)."""
    layers = [l for l in model.layers if not isinstance(l, keras.layers.InputLayer)]
    in_key, stages, cur = "__input__", [], None
    for layer in layers:
        if isinstance(layer, (keras.layers.Flatten, keras.layers.Reshape, keras.layers.Dropout)):
            continue
        if isinstance(layer, keras.layers.BatchNormalization):
            if cur is None or cur.out_name is not None:
                raise ValueError(f"{layer.name}: BatchNormalization must follow a QDense/QConv2D "
                                 "before its quantised activation")
            cur.post.append(_FrozenBN(layer))
            continue
        if isinstance(layer, QActivation):
            if cur is None:
                if stages:
                    raise ValueError(f"{layer.name}: QActivation not attached to a layer")
                in_key = layer.name           # input quantizer = input encoding
                continue
            cur.post.append(layer.activation)
            cur.out_name = layer.name
            continue
        if isinstance(layer, (QDense, QConv2D)):
            if cur is not None:
                stages.append(cur)
            post, out_name = [], None
            act = layer.activation
            if act is not None and getattr(act, "__name__", None) != "linear":
                post.append(act)
                if hasattr(act, "bits") or type(act).__name__ in ("binary", "ternary"):
                    out_name = layer.name
            cur = _Stage(layer, post, out_name, tuple(layer.input.shape[1:]),
                         tuple(layer.output.shape[1:]))
            continue
        raise ValueError(f"layer {layer.name} ({type(layer).__name__}) is not supported "
                         "by convert_to_lut (supported: QDense, QConv2D, BatchNormalization, "
                         "QActivation, Flatten, Reshape, Dropout)")
    if cur is not None:
        stages.append(cur)
    if not stages:
        raise ValueError("teacher has no QDense/QConv2D layers")
    for st in stages[:-1]:
        if st.out_name is None:
            raise ValueError(f"{st.layer.name}: hidden layers need a quantised activation")
    return in_key, stages


def _apply_post(stage, pre):
    """Apply the stage's post ops (BN, activations; Keras ops, float32).

    ``pre`` is ``(N, U)`` (one column per unit) and is evaluated in the layer's
    native output layout so per-channel ops (BatchNormalization) see the right
    axis."""
    pre = np.asarray(pre)
    if pre.ndim == 1:
        pre = pre[:, None]
    n, u = pre.shape
    shape = (n,) + (1,) * (len(stage.out_shape) - 1) + (u,)
    t = ops.convert_to_tensor(pre.astype(np.float32).reshape(shape))
    for f in stage.post:
        t = f(t)
    return ops.convert_to_numpy(t).astype(np.float64).reshape(n, u)


def _layer_post(stage, pres):
    """Apply post ops to a list of per-unit pre-activation vectors at once."""
    n = max(len(p) for p in pres)
    P = np.zeros((n, len(pres)))
    for u, p in enumerate(pres):
        P[:len(p), u] = p
    Q = _apply_post(stage, P)
    return [Q[:len(p), u] for u, p in enumerate(pres)]


def _bias(layer):
    if not layer.use_bias:
        return np.zeros(layer.kernel.shape[-1])
    q = getattr(layer, "bias_quantizer_internal", None)
    b = q(layer.bias) if q else layer.bias
    return ops.convert_to_numpy(b).astype(np.float64)


# =============================================================================
# Exact truth tables
# =============================================================================
def _enumerate_codes(widths):
    """All patterns of len(widths) codes -> (2**sum(w), F) code matrix."""
    B = int(sum(widths))
    a = np.arange(1 << B, dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(widths)[:-1]]).astype(np.int64)
    return np.stack([(a >> o) & ((1 << w) - 1) for o, w in zip(off, widths)], 1) if len(widths) \
        else np.zeros((1, 0), np.int64)


def _neuron_function(stage, vs_in, taps, wcol, b):
    """Evaluate one neuron on every input pattern.

    Returns (out_values (2**B,), valid (2**B,) bool)."""
    widths = [vs_in.bits] * len(taps)
    codes = _enumerate_codes(widths)
    valid = (codes < vs_in.size).all(1)
    vals = vs_in.values[np.minimum(codes, vs_in.size - 1)]
    pre = vals @ wcol[taps] + b
    return pre, valid


def _dyadic_exponent(x, max_e=20):
    """Smallest e >= 0 with x * 2**e integral for all x (dyadic values)."""
    x = np.asarray(x, np.float64)
    for e in range(max_e + 1):
        y = x * (2.0 ** e)
        if np.all(np.abs(y - np.round(y)) < 1e-9):
            return e
    raise ValueError("the logits are not fixed-point numbers: give the final layer a "
                     "fixed-point kernel_quantizer and bias_quantizer (or use_bias=False) so "
                     "the logits are exact integers")


# =============================================================================
# Distillation
# =============================================================================
@dataclasses.dataclass
class DistillConfig:
    """Student blocks for neurons that are too wide for exact conversion."""

    hidden: tuple = (1024,)          # LUTDense widths before the output LUT layer
    lut_rank: int = 6
    connections: str = "learnable"
    num_candidates: int | None = 8
    epochs: int = 10                 # layer-wise hint training
    kd_epochs: int = 5               # end-to-end knowledge distillation
    batch_size: int = 256
    learning_rate: float = 0.03
    temperature: float = 4.0         # KD softmax temperature
    alpha: float = 0.5               # weight of the KD term vs. hard labels
    seed: int = 0
    verbose: int = 0


def _bits_of(codes, nbits):
    """codes (N, U) int -> (N, U*nbits) {0,1} float32, unit-major, LSB first."""
    c = codes.astype(np.int64)[..., None] >> np.arange(nbits)
    return (c & 1).reshape(len(codes), -1).astype(np.float32)


class _DecodeBits(keras.layers.Layer):
    """Relaxed bits (N, U*b) -> values vmin + lsb * sum_j 2^j p_j (uniform codes)."""

    def __init__(self, units, nbits, vmin, lsb, **kw):
        super().__init__(**kw)
        self.units, self.nbits, self.vmin, self.lsb = units, nbits, float(vmin), float(lsb)

    def call(self, p):
        p = ops.reshape(p, (-1, self.units, self.nbits))
        w = ops.convert_to_tensor((2.0 ** np.arange(self.nbits)).astype("float32"))
        return self.vmin + self.lsb * ops.sum(p * w, axis=-1)


class _EncodeValues(keras.layers.Layer):
    """Values -> code bits (no gradient); used when a distilled block follows an
    exact stage during end-to-end distillation."""

    def __init__(self, vs: ValueSet, **kw):
        super().__init__(**kw)
        self.vmin = float(vs.values[0])
        self.lsb = float(vs.values[1] - vs.values[0]) if vs.size > 1 else 1.0
        self.nbits, self.nmax = vs.bits, vs.size - 1

    def call(self, v):
        v = ops.stop_gradient(v)
        c = ops.clip(ops.round((v - self.vmin) / self.lsb), 0, self.nmax)
        c = ops.cast(ops.reshape(c, (-1, int(np.prod(v.shape[1:])))), "int32")
        bits = [ops.cast(ops.bitwise_and(ops.right_shift(c, j), 1), "float32")
                for j in range(self.nbits)]
        return ops.reshape(ops.stack(bits, -1), (-1, int(np.prod(v.shape[1:])) * self.nbits))


class _TeacherStage(keras.layers.Layer):
    """Frozen teacher stage on values (used inside the KD model)."""

    def __init__(self, stage, **kw):
        super().__init__(trainable=False, **kw)
        self._stage = stage

    def call(self, v):
        s = self._stage
        v = ops.reshape(v, (-1,) + tuple(s.in_shape))
        y = s.layer(v)
        for f in s.post:
            y = f(y)
        return y


def _make_block(n_out, cfg, name):
    lays, rng = [], cfg.seed
    for i, h in enumerate(cfg.hidden):
        lays.append(LUTDense(h, lut_rank=cfg.lut_rank, connections=cfg.connections if i == 0 else "random",
                             num_candidates=cfg.num_candidates if i == 0 else None,
                             seed=rng + i, name=f"{name}_lut{i}"))
    lays.append(LUTDense(n_out, lut_rank=cfg.lut_rank, connections="random", seed=rng + 99,
                         name=f"{name}_out"))
    return lays


# =============================================================================
# Result container
# =============================================================================
@dataclasses.dataclass
class Conversion:
    """Result of :func:`convert_to_lut`."""

    netlist: Netlist
    report: list                     # per-stage dicts
    value_sets: dict
    patterns: dict
    student: Any = None              # Keras KD model (if any stage was distilled)
    seconds: float = 0.0

    def predict(self, x):
        """Class predictions of the converted network on raw inputs."""
        return simulate(self.netlist, x)[1]

    def evaluate(self, x, y, teacher=None):
        pred = self.predict(x)
        out = {"accuracy": float((pred == np.asarray(y).ravel()).mean())}
        if teacher is not None:
            tp = np.asarray(teacher.predict(x, verbose=0)).argmax(-1)
            out["teacher_accuracy"] = float((tp == np.asarray(y).ravel()).mean())
            out["agreement"] = float((tp == pred).mean())
        return out

    def summary(self) -> str:
        rows = [f"{'stage':<18}{'mode':<10}{'units':>7}{'in bits':>9}{'out bits':>9}"
                f"{'LUTs':>8}{'care %':>8}"]
        for r in self.report:
            care = "" if r.get("care_fraction") is None else f"{100 * r['care_fraction']:.1f}"
            rows.append(f"{r['layer']:<18}{r['mode']:<10}{r['units']:>7}{r['max_in_bits']:>9}"
                        f"{r['out_bits']:>9}{r['luts']:>8}{care:>8}")
        s = self.netlist.stats()
        rows.append(f"netlist: {s['luts']} LUTs, depth {s['depth']}, max k {s['max_k']}, "
                    f"head {self.netlist.head_type}")
        return "\n".join(rows)


# =============================================================================
# Main entry point
# =============================================================================
def convert_to_lut(
    teacher,
    data,
    *,
    labels=None,
    grid: str = "range",
    dont_care: str = "observed",
    max_exact_bits: int = 16,
    head: str = "linear",
    distill: DistillConfig | dict | None = None,
    batch_size: int = 1024,
    max_batches: int | None = None,
    name: str | None = None,
) -> Conversion:
    """Convert a trained QKeras model into a LUT netlist (teacher -> student).

    Args:
      teacher: Keras model built from ``[QActivation (input quantizer)]``,
        ``QDense`` / ``QConv2D`` (valid padding) layers with quantised
        activations (inline or ``QActivation``), ``Flatten``/``Reshape``/
        ``Dropout``, and a final linear ``QDense`` (logits) or quantised layer.
      data: calibration inputs (numpy array) used for range analysis, the care
        sets and, for distilled stages, as training data.
      labels: integer labels of ``data`` (only needed for distillation).
      grid: code book of each activation, see
        :func:`qkeras.range_analysis.analyze_value_sets`.
      dont_care: ``"observed"`` -- input patterns not seen on ``data`` are
        don't-cares; ``"none"`` -- fully specified tables (only codes that do
        not exist are don't-cares); the student is then exact for *any* input
        inside the code range.
      max_exact_bits: neurons with more input wires are distilled.
      head: output layer (linear logits) as ``"linear"`` -- an exact integer
        weighted sum of the previous layer's bits (adder tree in hardware, no
        fan-in limit) -- or ``"table"`` -- truth tables of the logit words
        (needs a small fan-in).
      distill: :class:`DistillConfig` (or dict) for wide stages; if None and a
        wide stage exists, a ``ValueError`` is raised.

    Returns: :class:`Conversion` with the netlist and a per-stage report.
    """
    t0 = time.time()
    if dont_care not in ("observed", "none"):
        raise ValueError("dont_care must be 'observed' or 'none'")
    if head not in ("linear", "table"):
        raise ValueError("head must be 'linear' or 'table'")
    if isinstance(distill, dict):
        distill = DistillConfig(**distill)
    data = np.asarray(data)
    in_key, stages = _parse_teacher(teacher)
    vsets = analyze_value_sets(teacher, data, grid=grid, batch_size=batch_size,
                               max_batches=max_batches)
    if in_key not in vsets:
        raise ValueError(f"input value set {in_key} missing")
    vs_input = vsets[in_key]
    if vs_input.overflow or (in_key == "__input__" and vs_input.size > 2):
        raise ValueError("model input is not Boolean; add an input quantizer "
                         "(QActivation) as the first layer so the input is finite")
    pats = analyze_patterns(teacher, data, vsets, max_bits=max_exact_bits,
                            batch_size=batch_size, max_batches=max_batches)

    # ----------------------------------------------------------- input wires
    n_feat = int(np.prod(teacher.inputs[0].shape[1:]))
    b0 = vs_input.bits
    if in_key == "__input__":
        thr = float(vs_input.values.mean()) if vs_input.size == 2 else 0.5
        encoder = {"type": "threshold", "threshold": thr}
        input_threshold = thr
    else:
        if not vs_input.is_uniform():
            raise ValueError("input quantizer grid must be uniform (use grid='range')")
        lsb = float(vs_input.values[1] - vs_input.values[0]) if vs_input.size > 1 else 1.0
        zmin = int(round(vs_input.values[0] / lsb))
        encoder = {"type": "uniform", "scale": [1.0 / lsb] * n_feat, "zmin": [zmin] * n_feat,
                   "bits": [b0] * n_feat}
        input_threshold = None
    # element e of the current tensor (flattened) -> list of wire indices (prev layer)
    wires = np.arange(n_feat * b0, dtype=np.int64).reshape(n_feat, b0)
    cur_vs = vs_input

    layers, report = [], []
    distilled_any = False
    kd_parts = []                     # for the end-to-end KD model
    head_opt, head = head, dict(type="bits")

    for si, st in enumerate(stages):
        L = st.layer
        last = si == len(stages) - 1
        kq = _quantized_kernel(L)
        b = _bias(L)
        conv = isinstance(L, QConv2D)
        kflat = kq.reshape(-1, kq.shape[-1])
        U = kflat.shape[1]
        taps = [np.flatnonzero(kflat[:, u] != 0) for u in range(U)]
        in_bits = [len(t) * cur_vs.bits for t in taps]
        exact = max(in_bits) <= max_exact_bits
        vs_out = vsets.get(st.out_name) if st.out_name else None
        rec = pats.get(L.name)

        # ------------------------------------------------- linear output head
        if last and vs_out is None and head_opt == "linear" and not conv and not st.post:
            if not cur_vs.is_uniform():
                raise ValueError("linear head needs uniform input codes (grid='range' or "
                                 "'quantizer'); use head='table'")
            v0 = float(cur_vs.values[0])
            lsb = float(cur_vs.values[1] - cur_vs.values[0]) if cur_vs.size > 1 else 0.0
            nb = cur_vs.bits
            coefs, biases = [], []
            for u in range(U):
                t = []
                bias = float(b[u])
                for i in taps[u]:
                    w = float(kflat[i, u])
                    bias += w * v0
                    for j in range(nb):
                        t.append((int(wires[i, j]), w * lsb * (1 << j)))
                coefs.append(t)
                biases.append(bias)
            allv = [c for t in coefs for _, c in t] + biases
            e = _dyadic_exponent(np.array(allv)) if allv else 0
            terms = [[(p, int(round(c * 2.0 ** e))) for p, c in t if round(c * 2.0 ** e) != 0]
                     for t in coefs]
            bias_i = [int(round(bb * 2.0 ** e)) for bb in biases]
            lo = min(bb + sum(min(0, w) for _, w in t) for t, bb in zip(terms, bias_i))
            hi = max(bb + sum(max(0, w) for _, w in t) for t, bb in zip(terms, bias_i))
            word = max(1, int(math.ceil(math.log2(hi - lo + 1))))
            head_info = dict(type="linear", num_classes=U, word_bits=word, lsb_exponent=e, zmin=lo,
                             terms=terms, bias=bias_i)
            report.append(dict(layer=L.name, mode="linear", units=U, conv=False, positions=1,
                               max_in_bits=int(max(in_bits)), mean_in_bits=float(np.mean(in_bits)),
                               out_bits=word, luts=0, care_fraction=None,
                               terms=int(sum(len(t) for t in terms))))
            kd_parts.append(("exact", st, None))
            head = head_info
            continue

        if conv:
            if L.padding != "valid" or tuple(L.dilation_rate) != (1, 1) or L.groups != 1:
                raise ValueError(f"{L.name}: only valid, undilated, ungrouped Conv2D supported")
            H, W, C = st.in_shape
            kh, kw = L.kernel_size
            sh, sw = L.strides
            OH, OW = (H - kh) // sh + 1, (W - kw) // sw + 1
            # window element p=(i,j,c) at output (r,s) -> input element index
            ii, jj, cc = np.meshgrid(np.arange(kh), np.arange(kw), np.arange(C), indexing="ij")
            ii, jj, cc = ii.ravel(), jj.ravel(), cc.ravel()
            positions = [(r, s) for r in range(OH) for s in range(OW)]
        if not exact:
            if conv:
                raise ValueError(f"{L.name}: wide Conv2D neurons cannot be distilled; "
                                 "constrain the kernel fan-in")
            if distill is None:
                raise ValueError(f"{L.name}: neurons need up to {max(in_bits)} input bits "
                                 f"> max_exact_bits={max_exact_bits}; pass distill=DistillConfig()")

        # ------------------------------------------------------------ exact
        if exact:
            tables = []                # per unit: (out_codes or words, care)
            care_tot = care_n = 0
            word = None
            if last and vs_out is None:
                # logits -> exact integer words (argmax-exact)
                pres, valids = [], []
                for u in range(U):
                    pre, valid = _neuron_function(st, cur_vs, taps[u], kflat[:, u], b[u])
                    pres.append(pre)
                    valids.append(valid)
                if st.post:   # e.g. BatchNormalization after the logits layer
                    pres = _layer_post(st, pres)
                allv = np.concatenate([p[v] for p, v in zip(pres, valids)])
                e = _dyadic_exponent(allv)
                ints = [np.round(p * 2.0 ** e).astype(np.int64) for p in pres]
                zmin = min(int(i[v].min()) for i, v in zip(ints, valids))
                zmax = max(int(i[v].max()) for i, v in zip(ints, valids))
                word = max(1, int(math.ceil(math.log2(zmax - zmin + 1))))
                outs = [np.clip(i - zmin, 0, (1 << word) - 1) for i in ints]
                nbits_out = word
                head = dict(type="words", num_classes=U, word_bits=word, lsb_exponent=e, zmin=zmin)
            else:
                if vs_out is None:
                    raise ValueError(f"{L.name}: output is not quantised")
                pres, valids = [], []
                for u in range(U):
                    pre, valid = _neuron_function(st, cur_vs, taps[u], kflat[:, u], b[u])
                    pres.append(pre)
                    valids.append(valid)
                outs = [vs_out.code(v) for v in _layer_post(st, pres)]
                nbits_out = vs_out.bits
            cares = []
            for u in range(U):
                care = valids[u].copy()
                if dont_care == "observed" and rec is not None and rec.patterns[u] is not None:
                    obs = np.zeros(len(care), bool)
                    obs[rec.patterns[u]] = True
                    care &= obs
                cares.append(care)
                care_tot += int(care.sum())
                care_n += len(care)
            # build LUTs
            lay = []
            n_out_elems = U * (len(positions) if conv else 1)
            out_wires = np.zeros((n_out_elems, nbits_out), np.int64)
            if conv:
                for pi, (r, s) in enumerate(positions):
                    elem = ((r * sh + ii) * W + (s * sw + jj)) * C + cc
                    for u in range(U):
                        pins = wires[elem[taps[u]]].ravel() if len(taps[u]) else np.zeros(0, np.int64)
                        for j in range(nbits_out):
                            out_wires[pi * U + u, j] = len(lay)
                            lay.append(LUT(tuple(int(p) for p in pins),
                                           ((outs[u] >> j) & 1).astype(bool), cares[u]))
            else:
                for u in range(U):
                    pins = wires[taps[u]].ravel() if len(taps[u]) else np.zeros(0, np.int64)
                    for j in range(nbits_out):
                        out_wires[u, j] = len(lay)
                        lay.append(LUT(tuple(int(p) for p in pins),
                                       ((outs[u] >> j) & 1).astype(bool), cares[u]))
            layers.append(lay)
            wires = out_wires
            report.append(dict(layer=L.name, mode="exact", units=U, conv=conv,
                               positions=len(positions) if conv else 1,
                               max_in_bits=int(max(in_bits)), mean_in_bits=float(np.mean(in_bits)),
                               out_bits=int(nbits_out), luts=len(lay),
                               care_fraction=care_tot / care_n if care_n else None))
            kd_parts.append(("exact", st, vs_out))
        # -------------------------------------------------------- distilled
        else:
            distilled_any = True
            if last and vs_out is None:
                nbits_out, groupsum = None, True
                n_out = U * max(1, distill.hidden[-1] // max(1, U) // 4) if distill.hidden else U * 16
                n_out = max(U * 8, n_out - n_out % U)
            else:
                nbits_out, groupsum = vs_out.bits, False
                n_out = U * nbits_out
            block = _make_block(n_out, distill, f"student_{L.name}")
            kd_parts.append(("distill", st, vs_out, block, groupsum, n_out))
            report.append(dict(layer=L.name, mode="distilled", units=U, conv=False, positions=1,
                               max_in_bits=int(max(in_bits)), mean_in_bits=float(np.mean(in_bits)),
                               out_bits=int(nbits_out or 0), luts=None, care_fraction=None,
                               block=[*distill.hidden, n_out]))
            layers.append(("block", len(kd_parts) - 1, wires))
            # outputs of the block's last LUT layer: unit-major, LSB first
            wires = None if groupsum else np.arange(n_out, dtype=np.int64).reshape(U, nbits_out)
            if groupsum:
                head = dict(type="groupsum", num_classes=U, tau=float(n_out // U) / 16.0)
        cur_vs = vs_out if vs_out is not None else cur_vs

    student = None
    if distilled_any:
        student = _train_students(teacher, data, labels, in_key, vs_input, kd_parts, distill,
                                  batch_size, n_feat)
        layers = _materialize_blocks(layers, kd_parts, report)

    if head["type"] == "linear":
        if not layers:
            raise ValueError("a single linear layer has no LUTs; nothing to convert")
        net = Netlist(n_feat * b0, layers, head["num_classes"], input_threshold,
                      name or f"{teacher.name}_lut", {"conversion": report, "lsb_exponent": head["lsb_exponent"],
                                                     "zmin": head["zmin"]},
                      head_type="linear", word_bits=head["word_bits"], input_encoder=encoder,
                      linear={"terms": head["terms"], "bias": head["bias"]})
    elif head["type"] == "words":
        net = Netlist(n_feat * b0, layers, head["num_classes"], input_threshold,
                      name or f"{teacher.name}_lut", {"conversion": report, "lsb_exponent": head["lsb_exponent"],
                                                     "zmin": head["zmin"]},
                      head_type="words", word_bits=head["word_bits"], input_encoder=encoder)
    elif head["type"] == "groupsum":
        net = Netlist(n_feat * b0, layers, head["num_classes"], input_threshold,
                      name or f"{teacher.name}_lut", {"conversion": report, "tau": head["tau"]},
                      head_type="groupsum", input_encoder=encoder)
    else:
        net = Netlist(n_feat * b0, layers, None, input_threshold, name or f"{teacher.name}_lut",
                      {"conversion": report}, head_type="bits", input_encoder=encoder)
    return Conversion(net, report, vsets, pats, student, time.time() - t0)


# =============================================================================
# Student training and materialisation
# =============================================================================
def _stage_inputs_and_targets(teacher, data, kd_parts, in_key, batch_size):
    """Teacher tensors needed for hint training: input/output of every stage."""
    outs = []
    for part in kd_parts:
        st = part[1]
        outs += [st.layer.input, (teacher.get_layer(st.out_name).output if st.out_name
                                  else st.layer.output)]
    inp = teacher.inputs[0]
    m = keras.Model(inp, outs)
    res = m.predict(data, batch_size=batch_size, verbose=0)
    return [(res[2 * i], res[2 * i + 1]) for i in range(len(kd_parts))]


def _train_students(teacher, data, labels, in_key, vs_input, kd_parts, cfg, batch_size, n_feat):
    keras.utils.set_random_seed(cfg.seed)
    io = _stage_inputs_and_targets(teacher, data, kd_parts, in_key, batch_size)
    vs_prev = vs_input
    # ---- layer-wise hints: each block learns teacher in-codes -> out-codes
    for i, part in enumerate(kd_parts):
        if part[0] != "distill":
            vs_prev = part[2] if part[2] is not None else vs_prev
            continue
        _, st, vs_out, block, groupsum, n_out = part
        xin, yout = io[i]
        x_bits = _bits_of(vs_prev.code(xin.reshape(len(xin), -1)), vs_prev.bits)
        if groupsum:
            if labels is None:
                raise ValueError("labels are required to distill the output layer")
            inp = keras.Input((x_bits.shape[1],))
            h = inp
            for l in block:
                h = l(h)
            U = st.layer.units
            logits = GroupSum(U, tau=float(n_out // U) / 16.0)(h)
            m = keras.Model(inp, logits)
            T = cfg.temperature
            t_soft = keras.ops.convert_to_numpy(keras.activations.softmax(yout / T)).astype("float32")
            m.compile(keras.optimizers.Adam(cfg.learning_rate),
                      keras.losses.CategoricalCrossentropy(from_logits=True))
            m.fit(x_bits, t_soft, epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=cfg.verbose)
        else:
            y_bits = _bits_of(vs_out.code(yout.reshape(len(yout), -1)), vs_out.bits)
            inp = keras.Input((x_bits.shape[1],))
            h = inp
            for l in block:
                h = l(h)
            m = keras.Model(inp, h)
            m.compile(keras.optimizers.Adam(cfg.learning_rate), keras.losses.BinaryCrossentropy())
            m.fit(x_bits, y_bits, epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=cfg.verbose)
        vs_prev = vs_out if vs_out is not None else vs_prev
    # ---- end-to-end KD through the frozen teacher stages
    if cfg.kd_epochs <= 0:
        return None
    inp = keras.Input(tuple(teacher.inputs[0].shape[1:]))
    v = inp
    vs_prev = vs_input
    if in_key != "__input__":
        v = teacher.get_layer(in_key)(v)
    first = True
    for part in kd_parts:
        if part[0] == "exact":
            _, st, vs_out = part
            v = _TeacherStage(st)(v)
            vs_prev = vs_out if vs_out is not None else vs_prev
        else:
            _, st, vs_out, block, groupsum, n_out = part
            h = _EncodeValues(vs_prev)(keras.layers.Flatten()(v) if len(v.shape) > 2 else v)
            for l in block:
                h = l(h)
            if groupsum:
                v = GroupSum(st.layer.units, tau=float(n_out // st.layer.units) / 16.0)(h)
            else:
                lsb = float(vs_out.values[1] - vs_out.values[0]) if vs_out.size > 1 else 1.0
                v = _DecodeBits(st.layer.units, vs_out.bits, vs_out.values[0], lsb)(h)
                v = keras.layers.Reshape(st.out_shape)(v)
            vs_prev = vs_out if vs_out is not None else vs_prev
        first = False
    kd = keras.Model(inp, v, name="kd_student")
    t_logits = teacher.predict(data, batch_size=batch_size, verbose=0)
    T, a = cfg.temperature, cfg.alpha
    y = np.zeros_like(t_logits) if labels is None else \
        keras.utils.to_categorical(np.asarray(labels).ravel(), t_logits.shape[-1])
    target = np.concatenate([t_logits, y], 1).astype("float32")
    C = t_logits.shape[-1]

    def kd_loss(yt, yp):
        tl, yh = yt[:, :C], yt[:, C:]
        soft = keras.ops.softmax(tl / T)
        l_kd = keras.losses.categorical_crossentropy(soft, yp / T, from_logits=True) * (T * T)
        if labels is None:
            return l_kd
        l_ce = keras.losses.categorical_crossentropy(yh, yp, from_logits=True)
        return a * l_kd + (1 - a) * l_ce

    kd.compile(keras.optimizers.Adam(cfg.learning_rate / 3), kd_loss)
    kd.fit(data, target, epochs=cfg.kd_epochs, batch_size=cfg.batch_size, verbose=cfg.verbose)
    return kd


def _materialize_blocks(layers, kd_parts, report):
    """Replace ('block', i, in_wires) placeholders by the trained LUT layers.
    The stage after a block was already wired to the block's output LUTs."""
    out = []
    for item in layers:
        if isinstance(item, tuple) and item[0] == "block":
            _, i, in_wires = item
            _, st, vs_out, block, groupsum, n_out = kd_parts[i]
            src = np.asarray(in_wires).ravel()       # element-major, LSB first
            n_luts = 0
            for li, l in enumerate(block):
                idx, tab = l.get_luts()
                if li == 0:
                    idx = src[idx]
                out.append([LUT(tuple(int(p) for p in idx[u]), tab[u].copy())
                            for u in range(l.units)])
                n_luts += l.units
            rep = next(r for r in report if r["layer"] == st.layer.name)
            rep["luts"] = n_luts
        else:
            out.append(item)
    return out
