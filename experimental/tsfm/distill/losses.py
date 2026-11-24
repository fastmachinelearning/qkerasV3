"""Losses for quantile distillation.

``y_true`` packs ground truth and teacher quantiles: ``(B, V, H, 1 + Q)`` with
``[..., 0]`` = ground truth and ``[..., 1:]`` = teacher quantiles at the student's knots,
all in the student's normalised (asinh) space.  ``y_pred`` is ``(B, V, H, Q)``.
"""

from __future__ import annotations

import keras
from keras import ops


def _pinball(y, q, taus):
    err = y[..., None] - q
    return ops.mean(ops.maximum(taus * err, (taus - 1.0) * err))


@keras.saving.register_keras_serializable(package="distill")
class DistillQuantileLoss(keras.losses.Loss):
    """``alpha * pinball(q, y) + (1 - alpha) * |q - q_teacher|``.

    The L1 term is the pinball loss of a point-mass target at every level (the teacher's
    quantile is the target for the same level), i.e. it matches quantile to quantile.
    """

    def __init__(self, knots, alpha: float = 0.5, name: str = "distill_quantile_loss", **kw):
        super().__init__(name=name, **kw)
        self.knots = tuple(float(k) for k in knots)
        self.alpha = float(alpha)

    def call(self, y_true, y_pred):
        taus = ops.convert_to_tensor(self.knots, dtype=y_pred.dtype)
        gt = _pinball(y_true[..., 0], y_pred, taus)
        if self.alpha >= 1.0:
            return gt
        kd = ops.mean(ops.abs(y_pred - y_true[..., 1:]))
        return self.alpha * gt + (1.0 - self.alpha) * kd

    def get_config(self):
        return {**super().get_config(), "knots": list(self.knots), "alpha": self.alpha}


def gt_pinball_metric(knots):
    taus = tuple(float(k) for k in knots)

    def gt_pinball(y_true, y_pred):
        return _pinball(y_true[..., 0], y_pred, ops.convert_to_tensor(taus, dtype=y_pred.dtype))

    return gt_pinball
