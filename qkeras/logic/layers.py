"""Trainable k-input LUT layers (k = 1..6) for Keras 3.

Every unit of a :class:`LUTDense` layer is one k-input look-up table: it reads
``lut_rank`` wires of the previous layer and learns its full ``2**lut_rank``
truth table.  After training the layer is a plain Boolean netlist that maps
1:1 onto FPGA LUTs (see :mod:`qkeras.logic.netlist`; HDL generation is in the ``logicnet-fpga`` backend).

Conventions (shared with the exporters)
---------------------------------------
* pin ``j`` of a LUT drives address bit ``j``  (pin 0 = LSB)
* truth-table bit ``a`` is the LUT output for address ``a``

Training vs. inference
----------------------
``training=True``  : relaxed forward (multilinear interpolation of a soft table,
                     optional straight-through hard sampling).
``training=False`` : exact discrete forward (thresholded table lookup, argmax
                     routing).  ``model.predict`` / ``model.evaluate`` / Keras
                     validation therefore report the accuracy of the hardware.

Parametrizations
----------------
``"light"`` : one logit per table entry, ``sigmoid`` then interpolate
              (torchlogix "light", the original 6-LUT prototype).
``"walsh"`` : Walsh-Hadamard coefficients ``w``; table logits ``Z = H w`` are
              interpolated first and squashed afterwards (torchlogix "warp").
Both expose the same table logits ``Z``; the hardware LUT is ``Z > 0``.
"""

import numpy as np
import keras
from keras import ops

__all__ = [
    "LUTDense",
    "GroupSum",
    "BinaryThreshold",
    "LUTTemperatureScheduler",
    "set_lut_temperature",
    "lut_layers",
]

_PKG = "qkeras_logic"
MAX_LUT_RANK = 6


def _address_bits(k):
    """(2**k, k) array, element [a, j] = bit j of address a."""
    a = np.arange(1 << k)
    return ((a[:, None] >> np.arange(k)) & 1).astype(np.int64)


def walsh_matrix(k):
    """H[a, S] = prod_{j in S} (2*bit_j(a) - 1); Z = w @ H.T."""
    bits = _address_bits(k)                    # (A, k)
    sets = _address_bits(k)                    # (S, k) same enumeration
    # sign = (-1)^{|S \ a|}
    neg = (sets[None, :, :] * (1 - bits[:, None, :])).sum(-1)
    return np.where(neg % 2 == 0, 1.0, -1.0).astype("float32")


def random_connections(in_dim, units, k, rng):
    """(units, k) input indices, distinct within a LUT, all inputs covered
    when units*k >= in_dim (concatenated random permutations)."""
    if k > in_dim:
        raise ValueError(f"lut_rank={k} > input dimension {in_dim}")
    n = units * k
    reps = -(-n // in_dim)
    flat = np.concatenate([rng.permutation(in_dim) for _ in range(reps)])[:n]
    idx = flat.reshape(units, k)
    # repair duplicates inside one LUT (rare, only near permutation seams)
    for u in range(units):
        row = idx[u]
        if len(set(row.tolist())) < k:
            free = np.setdiff1d(np.arange(in_dim), row)
            seen = set()
            for j in range(k):
                if row[j] in seen:
                    row[j] = rng.choice(free)
                    free = free[free != row[j]]
                seen.add(int(row[j]))
    return idx.astype("int32")


@keras.saving.register_keras_serializable(package=_PKG)
class LUTDense(keras.layers.Layer):
    """Fully connected layer of trainable k-input LUTs.

    Args:
      units: number of LUTs (output bits).
      lut_rank: LUT inputs k, 1..6.
      parametrization: ``"light"`` or ``"walsh"`` (see module docstring).
      connections: ``"random"`` - fixed random wiring (cheap, torchlogix
        default) or ``"learnable"`` - each pin learns which wire to read.
      num_candidates: for learnable wiring, number of random candidate wires per
        pin; ``None`` means every input wire is a candidate.
      init: ``"residual"`` - every LUT starts as the identity of pin 0 (keeps
        deep stacks trainable) or ``"random"``.
      residual_logit: magnitude of the residual init logits.
      temperature: initial temperature of LUT sigmoid / routing softmax
        (non-trainable variable, anneal with :class:`LUTTemperatureScheduler`).
      forward_sampling: ``"soft"`` or ``"hard"`` (straight-through: forward
        uses the thresholded table, gradient flows through the sigmoid).
      routing_sampling: ``"hard"`` (STE one-hot, default) or ``"soft"``.
      seed: seed for wiring and initialisation.
    """

    def __init__(
        self,
        units,
        lut_rank=6,
        parametrization="light",
        connections="random",
        num_candidates=None,
        init="residual",
        residual_logit=3.0,
        temperature=1.0,
        forward_sampling="soft",
        routing_sampling="hard",
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not 1 <= int(lut_rank) <= MAX_LUT_RANK:
            raise ValueError(f"lut_rank must be 1..{MAX_LUT_RANK}, got {lut_rank}")
        if parametrization not in ("light", "walsh"):
            raise ValueError(f"unknown parametrization {parametrization!r}")
        if connections not in ("random", "learnable"):
            raise ValueError(f"unknown connections {connections!r}")
        if init not in ("residual", "random"):
            raise ValueError(f"unknown init {init!r}")
        if forward_sampling not in ("soft", "hard"):
            raise ValueError(f"unknown forward_sampling {forward_sampling!r}")
        if routing_sampling not in ("soft", "hard"):
            raise ValueError(f"unknown routing_sampling {routing_sampling!r}")
        self.units = int(units)
        self.lut_rank = int(lut_rank)
        self.lut_size = 1 << self.lut_rank
        self.parametrization = parametrization
        self.connections = connections
        self.num_candidates = None if num_candidates in (None, -1) else int(num_candidates)
        self.init = init
        self.residual_logit = float(residual_logit)
        self.initial_temperature = float(temperature)
        self.forward_sampling = forward_sampling
        self.routing_sampling = routing_sampling
        self.seed = seed

    # ------------------------------------------------------------------ build
    def build(self, input_shape):
        in_dim = input_shape[-1]
        if in_dim is None:
            raise ValueError("LUTDense needs a known input dimension")
        self.input_dim = int(in_dim)
        k, U, A = self.lut_rank, self.units, self.lut_size
        rng = np.random.default_rng(self.seed)

        self.temperature = self.add_weight(
            shape=(), name="temperature", trainable=False,
            initializer=keras.initializers.Constant(self.initial_temperature))

        # ---- wiring
        if self.connections == "random":
            idx = random_connections(self.input_dim, U, k, rng)
            self.indices = self.add_weight(
                shape=(U, k), dtype="int32", trainable=False, name="indices",
                initializer=lambda shape, dtype=None: idx)
        else:
            C = self.num_candidates or self.input_dim
            self._all_candidates = self.num_candidates is None
            if not self._all_candidates:
                cand = np.stack(
                    [random_connections(self.input_dim, U, k, rng) for _ in range(C)],
                    axis=-1)                                   # (U, k, C)
                self.candidates = self.add_weight(
                    shape=(U, k, C), dtype="int32", trainable=False,
                    name="candidates", initializer=lambda shape, dtype=None: cand)
            # small random logits; break symmetry so pins pick different wires
            logits = rng.normal(0.0, 1.0, size=(U, k, C)).astype("float32")
            self.routing_logits = self.add_weight(
                shape=(U, k, C), name="routing_logits",
                initializer=lambda shape, dtype=None: logits)

        # ---- truth table
        if self.init == "residual":
            # identity of pin 0 with N(0,1) noise (torchlogix-style)
            ident = np.where(_address_bits(k)[:, 0] == 1, 1.0, -1.0)  # (A,)
            if self.parametrization == "light":
                w0 = self.residual_logit * ident[None, :] + rng.normal(0, 1, (U, A))
            else:  # walsh: only coefficient of S={pin 0} (index 1) non-zero
                w0 = rng.normal(0, 0.1, (U, A))
                w0[:, 1] += self.residual_logit
        else:
            w0 = rng.normal(0, 1, (U, A))
        w0 = w0.astype("float32")
        self.lut_weights = self.add_weight(
            shape=(U, A), name="lut_weights",
            initializer=lambda shape, dtype=None: w0)
        if self.parametrization == "walsh":
            self._walsh = ops.convert_to_tensor(walsh_matrix(k))
        self._pow2 = ops.convert_to_tensor((1 << np.arange(k)).astype("int32"))
        super().build(input_shape)

    # --------------------------------------------------------------- helpers
    def table_logits(self):
        """(units, 2**k) table logits Z; the hardware LUT is Z > 0."""
        if self.parametrization == "light":
            return self.lut_weights
        return ops.matmul(self.lut_weights, ops.transpose(self._walsh))

    def _hard_indices(self):
        if self.connections == "random":
            return self.indices
        choice = ops.argmax(self.routing_logits, axis=-1)          # (U, k)
        if self._all_candidates:
            return ops.cast(choice, "int32")
        return ops.cast(ops.squeeze(ops.take_along_axis(
            self.candidates, ops.expand_dims(choice, -1), axis=-1), -1), "int32")

    def _route_train(self, x):
        """(B, in) -> (B, U, k) selected (soft) input values."""
        if self.connections == "random":
            return ops.take(x, self.indices, axis=1)
        w = ops.softmax(self.routing_logits / self.temperature, axis=-1)
        if self.routing_sampling == "hard":
            hard = ops.one_hot(ops.argmax(w, axis=-1), w.shape[-1], dtype=w.dtype)
            w = w + ops.stop_gradient(hard - w)
        if self._all_candidates:   # dense matmul over all wires
            U, k, C = w.shape
            sel = ops.matmul(x, ops.transpose(ops.reshape(w, (U * k, C))))
            return ops.reshape(sel, (-1, U, k))
        # scatter the candidate weights into a dense (in_dim, U*k) routing
        # matrix and use one matmul; same values/gradients as gathering
        # x[:, candidates] (B,U,k,C) but far cheaper on CPU/GPU.
        U, k, C = w.shape
        rows = ops.reshape(self.candidates, (-1, 1))
        cols = ops.reshape(ops.repeat(ops.arange(U * k, dtype="int32")[:, None], C, axis=1), (-1, 1))
        M = ops.scatter(ops.concatenate([rows, cols], axis=1), ops.reshape(w, (-1,)),
                        (self.input_dim, U * k))
        return ops.reshape(ops.matmul(x, M), (-1, U, k))

    def _interpolate(self, table, sel):
        """Multilinear interpolation, one pin at a time (pin 0 = LSB).

        table: (U, 2**k)   sel: (B, U, k)  ->  (B, U)
        """
        v = table[None]                                            # (1,U,A)
        for j in range(self.lut_rank):
            v = ops.reshape(v, (-1, self.units, v.shape[-1] // 2, 2))
            xj = sel[:, :, j:j + 1]
            v = v[..., 0] + xj * (v[..., 1] - v[..., 0])
        return v[..., 0]

    # ------------------------------------------------------------------ call
    def call(self, inputs, training=None):
        if training:
            return self._call_relaxed(inputs)
        return self._call_discrete(inputs)

    def _call_relaxed(self, x):
        z = self.table_logits()
        sel = self._route_train(x)
        t = self.temperature
        if self.parametrization == "light":
            p = ops.sigmoid(z / t)
            if self.forward_sampling == "hard":
                p = p + ops.stop_gradient(ops.cast(z > 0, p.dtype) - p)
            return self._interpolate(p, sel)
        y = ops.sigmoid(self._interpolate(z, sel) / t)
        if self.forward_sampling == "hard":
            y = y + ops.stop_gradient(ops.cast(y > 0.5, y.dtype) - y)
        return y

    def _call_discrete(self, x):
        table = ops.cast(self.table_logits() > 0, "float32")       # (U, A)
        bits = ops.cast(ops.take(x, self._hard_indices(), axis=1) > 0.5, "int32")
        addr = ops.sum(bits * self._pow2, axis=-1)                 # (B, U)
        offs = ops.arange(self.units, dtype="int32") * self.lut_size
        out = ops.take(ops.reshape(table, (-1,)), addr + offs[None, :])
        return ops.cast(out, self.compute_dtype)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.units,)

    # ---------------------------------------------------------------- export
    def get_luts(self):
        """Return ``(indices (U,k) int, tables (U, 2**k) bool)`` in numpy."""
        idx = ops.convert_to_numpy(self._hard_indices()).astype(np.int64)
        tab = ops.convert_to_numpy(self.table_logits()) > 0
        return idx, tab

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            units=self.units, lut_rank=self.lut_rank,
            parametrization=self.parametrization, connections=self.connections,
            num_candidates=self.num_candidates, init=self.init,
            residual_logit=self.residual_logit,
            temperature=self.initial_temperature,
            forward_sampling=self.forward_sampling,
            routing_sampling=self.routing_sampling, seed=self.seed)
        return cfg


@keras.saving.register_keras_serializable(package=_PKG)
class GroupSum(keras.layers.Layer):
    """Split the last axis into ``num_classes`` equal groups and popcount them.

    Output logits are ``popcount / tau``.  In hardware this is a popcount per
    class followed by an argmax (lowest index wins ties, like ``np.argmax``).
    """

    def __init__(self, num_classes=10, tau=1.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.tau = float(tau)

    def build(self, input_shape):
        n = input_shape[-1]
        if n is None or n % self.num_classes:
            raise ValueError(f"{n} inputs not divisible by {self.num_classes} classes")
        self.group_size = n // self.num_classes
        super().build(input_shape)

    def call(self, inputs):
        g = ops.reshape(inputs, (-1, self.num_classes, self.group_size))
        return ops.sum(g, axis=-1) / self.tau

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.num_classes,)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(num_classes=self.num_classes, tau=self.tau)
        return cfg


@keras.saving.register_keras_serializable(package=_PKG)
class BinaryThreshold(keras.layers.Layer):
    """``x > threshold`` as float {0, 1}; marks the Boolean network input."""

    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = float(threshold)

    def call(self, inputs):
        return ops.cast(inputs > self.threshold, self.compute_dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update(threshold=self.threshold)
        return cfg


def lut_layers(model):
    return [l for l in model.layers if isinstance(l, LUTDense)]


def set_lut_temperature(model, value):
    for layer in lut_layers(model):
        layer.temperature.assign(float(value))


class LUTTemperatureScheduler(keras.callbacks.Callback):
    """Geometric temperature annealing ``start -> end`` over ``epochs``."""

    def __init__(self, start=1.0, end=0.1, epochs=10, verbose=0):
        super().__init__()
        self.start, self.end = float(start), float(end)
        self.epochs, self.verbose = max(1, int(epochs)), verbose

    def on_epoch_begin(self, epoch, logs=None):
        p = min(epoch, self.epochs - 1) / max(1, self.epochs - 1)
        t = self.start * (self.end / self.start) ** p
        set_lut_temperature(self.model, t)
        if self.verbose:
            print(f"LUT temperature = {t:.4f}")
