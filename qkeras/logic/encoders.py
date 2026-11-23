"""Input encoders for LUT networks.

LUT networks read single bits.  How a continuous feature is turned into bits
matters a lot: a binary fixed-point code (``quantized_bits``) hands the first
LUT layer individual code bits (MSB, LSB, ...) which are poor features, while a
*thermometer* code ``x > t_1, x > t_2, ...`` hands it monotone comparisons
(the distributive thermometer of DWN, Bacellar et al. 2024).

:class:`Thermometer` maps ``(B, F)`` features to ``(B, F * T)`` bits, feature
major (wire ``f * T + j`` is ``x_f > t_{f, j}``).  Thresholds start at
per-feature quantiles of the training data (:meth:`Thermometer.adapt`) and can
be trained with a straight-through sigmoid.

Hardware: the network input is a ``bits``-bit fixed-point word per feature
(``quantized_bits(bits, integer, 1, alpha=1)`` grid, see :func:`fixed_point`).
Thresholds are snapped to half-steps of that grid, so ``x > t`` on the
quantised feature is exact.  :func:`qkeras.logic.netlist.materialize_inputs`
turns the thermometer into a first layer of comparator LUTs over the words,
so the encoder's area is part of the netlist.
"""

import numpy as np
import keras
from keras import ops

__all__ = ["Thermometer", "fixed_point", "quantile_thresholds"]


def fixed_point(x, bits=8, integer=3):
    """Quantise to the signed ``quantized_bits(bits, integer, 1, alpha=1)`` grid
    (round half to even, saturate).  Returns float32 values on the grid."""
    scale = 2.0 ** (bits - 1 - integer)
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return (np.clip(np.rint(np.asarray(x, np.float32) * scale), lo, hi) / scale).astype(np.float32)


def quantile_thresholds(x, num_thresholds):
    """(F, T) thresholds at the ``j / (T + 1)`` quantiles of each feature."""
    q = np.arange(1, num_thresholds + 1) / (num_thresholds + 1)
    return np.quantile(np.asarray(x, np.float64), q, axis=0).T.astype(np.float32)


class Thermometer(keras.layers.Layer):
    """Thermometer encoder ``(B, F) -> (B, F * T)`` bits.

    Args:
      num_thresholds: T, thresholds per feature.
      trainable_thresholds: learn the thresholds (straight-through sigmoid).
      temperature: slope of the training sigmoid in feature units.
      grid: ``(bits, integer)`` of the fixed-point input word.  Thresholds are
        snapped to half-steps of this grid in the discrete (inference /
        export) path; training uses the continuous thresholds.
      thresholds: optional initial ``(F, T)`` array (else call :meth:`adapt`).
    """

    def __init__(self, num_thresholds=16, trainable_thresholds=False, temperature=0.1,
                 grid=(8, 3), thresholds=None, **kwargs):
        super().__init__(**kwargs)
        self.num_thresholds = int(num_thresholds)
        self.trainable_thresholds = bool(trainable_thresholds)
        self.temperature_value = float(temperature)
        self.grid = None if grid is None else (int(grid[0]), int(grid[1]))
        self._init_thresholds = None if thresholds is None else np.asarray(thresholds, np.float32)

    def build(self, input_shape):
        F = int(input_shape[-1])
        self.num_features = F
        init = self._init_thresholds
        if init is None:
            init = np.tile(np.linspace(-2, 2, self.num_thresholds, dtype=np.float32), (F, 1))
        self.thresholds = self.add_weight(
            shape=(F, self.num_thresholds), name="thresholds",
            trainable=self.trainable_thresholds,
            initializer=lambda shape, dtype=None: np.asarray(init, np.float32))
        super().build(input_shape)

    def adapt(self, x):
        """Initialise thresholds to per-feature quantiles of ``x`` (raw features)."""
        x = np.asarray(x, np.float32)
        if not self.built:
            self._init_thresholds = quantile_thresholds(x, self.num_thresholds)
            self.build((None, x.shape[-1]))
        else:
            self.thresholds.assign(quantile_thresholds(x, self.num_thresholds))

    # thresholds actually used in hardware
    def snapped_thresholds(self):
        t = ops.convert_to_numpy(self.thresholds).astype(np.float64)
        if self.grid is None:
            return t.astype(np.float32)
        bits, integer = self.grid
        scale = 2.0 ** (bits - 1 - integer)
        lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        c = np.clip(np.floor(t * scale), lo - 1, hi)   # x > t  <=>  code > c  for grid x
        return ((c + 0.5) / scale).astype(np.float32)

    def call(self, x, training=None):
        x = ops.expand_dims(x, -1)                                # (B, F, 1)
        if training:
            t = self.thresholds
            soft = ops.sigmoid((x - t) / self.temperature_value)
            hard = ops.cast(x > t, soft.dtype)
            y = soft + ops.stop_gradient(hard - soft)
        else:
            t = self.thresholds if self.grid is None else self._snapped_tensor()
            y = ops.cast(x > t, self.compute_dtype)
        return ops.reshape(y, (-1, self.num_features * self.num_thresholds))

    def _snapped_tensor(self):
        bits, integer = self.grid
        scale = 2.0 ** (bits - 1 - integer)
        lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        c = ops.clip(ops.floor(self.thresholds * scale), lo - 1, hi)
        return (c + 0.5) / scale

    def encoder_spec(self):
        """``input_encoder`` dict for the netlist (thermometer on raw features)."""
        t = self.snapped_thresholds()
        F, T = t.shape
        spec = {"type": "thermometer", "feature": np.repeat(np.arange(F), T).tolist(),
                "threshold": [float(v) for v in t.reshape(-1)]}
        if self.grid is not None:
            spec["grid"] = {"bits": self.grid[0], "integer": self.grid[1]}
        return spec

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (int(input_shape[-1]) * self.num_thresholds,)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(num_thresholds=self.num_thresholds,
                   trainable_thresholds=self.trainable_thresholds,
                   temperature=self.temperature_value, grid=self.grid)
        return cfg
