"""Distillation losses.

All losses take a *packed* ``y_true`` so a plain ``model.fit(x, y_packed)`` works on every
backend: ``y_true[..., 0]`` is the ground truth, ``y_true[..., 1:]`` the teacher output.
``pack_targets`` builds it.

``QuantileDistillLoss``   probabilistic regression (forecasters):
    ``alpha * pinball(student, truth) + (1 - alpha) * |student - teacher|`` per quantile.
``LogitDistillLoss``      classification / language models (Hinton et al. 2015):
    ``alpha * CE(student, label) + (1 - alpha) * T^2 * KL(softmax(t/T) || softmax(s/T))``.
``FeatureDistillLoss``    hidden-state matching (``alpha`` MSE to truth, ``1 - alpha`` to teacher).
"""

from __future__ import annotations

import keras
import numpy as np
from keras import ops

_PKG = "qkeras.distill"


def pack_targets(truth: np.ndarray, teacher: np.ndarray) -> np.ndarray:
    """``truth (..., )`` + ``teacher (..., K)`` -> ``(..., 1 + K)`` float32."""
    truth = np.asarray(truth, np.float32)
    return np.concatenate([truth[..., None], np.asarray(teacher, np.float32)], axis=-1)


def pinball(y, q, taus):
    """Mean pinball loss of quantiles ``q (..., Q)`` at levels ``taus`` for truth ``y (...)``."""
    err = y[..., None] - q
    return ops.mean(ops.maximum(taus * err, (taus - 1.0) * err))


@keras.saving.register_keras_serializable(package=_PKG)
class QuantileDistillLoss(keras.losses.Loss):
    """``alpha * pinball(q, y) + (1 - alpha) * |q - q_teacher|``.

    The L1 term is the pinball loss of a point-mass target at every level (the teacher's
    quantile is the target for the same level), i.e. it matches quantile to quantile.
    ``alpha = 1`` trains on ground truth only.
    """

    def __init__(self, knots, alpha: float = 0.5, name: str = "distill_quantile_loss", **kw):
        super().__init__(name=name, **kw)
        self.knots = tuple(float(k) for k in knots)
        self.alpha = float(alpha)

    def call(self, y_true, y_pred):
        taus = ops.convert_to_tensor(self.knots, dtype=y_pred.dtype)
        gt = pinball(y_true[..., 0], y_pred, taus)
        if self.alpha >= 1.0:
            return gt
        kd = ops.mean(ops.abs(y_pred - y_true[..., 1:]))
        return self.alpha * gt + (1.0 - self.alpha) * kd

    def get_config(self):
        return {**super().get_config(), "knots": list(self.knots), "alpha": self.alpha}


def gt_pinball_metric(knots):
    """Metric ``gt_pinball``: pinball loss against the ground truth only (monitor ``val_gt_pinball``)."""
    taus = tuple(float(k) for k in knots)

    def gt_pinball(y_true, y_pred):
        return pinball(y_true[..., 0], y_pred, ops.convert_to_tensor(taus, dtype=y_pred.dtype))

    return gt_pinball


@keras.saving.register_keras_serializable(package=_PKG)
class LogitDistillLoss(keras.losses.Loss):
    """Soft-label distillation for classifiers and language models.

    ``y_true[..., 0]`` = integer label (as float), ``y_true[..., 1:]`` = teacher logits ``(..., C)``;
    ``y_pred`` = student logits ``(..., C)``.  ``temperature`` softens both distributions; the KL
    term is scaled by ``T^2`` so its gradient magnitude does not depend on ``T``.
    """

    def __init__(self, alpha: float = 0.5, temperature: float = 2.0, name: str = "distill_logit_loss", **kw):
        super().__init__(name=name, **kw)
        self.alpha = float(alpha)
        self.temperature = float(temperature)

    def call(self, y_true, y_pred):
        labels = ops.cast(y_true[..., 0], "int32")
        ce = ops.mean(keras.losses.sparse_categorical_crossentropy(labels, y_pred, from_logits=True))
        if self.alpha >= 1.0:
            return ce
        t = self.temperature
        p_t = ops.softmax(y_true[..., 1:] / t, axis=-1)
        log_p_t = ops.log_softmax(y_true[..., 1:] / t, axis=-1)
        log_p_s = ops.log_softmax(y_pred / t, axis=-1)
        kl = ops.mean(ops.sum(p_t * (log_p_t - log_p_s), axis=-1))
        return self.alpha * ce + (1.0 - self.alpha) * (t * t) * kl

    def get_config(self):
        return {**super().get_config(), "alpha": self.alpha, "temperature": self.temperature}


@keras.saving.register_keras_serializable(package=_PKG)
class FeatureDistillLoss(keras.losses.Loss):
    """``alpha * MSE(pred, truth) + (1 - alpha) * MSE(pred, teacher)`` for point outputs or hidden
    states; ``y_true (..., 2)`` packs ``[truth, teacher]`` per element."""

    def __init__(self, alpha: float = 0.5, name: str = "distill_feature_loss", **kw):
        super().__init__(name=name, **kw)
        self.alpha = float(alpha)

    def call(self, y_true, y_pred):
        gt = ops.mean(ops.square(y_pred - y_true[..., 0]))
        if self.alpha >= 1.0:
            return gt
        kd = ops.mean(ops.square(y_pred - y_true[..., 1]))
        return self.alpha * gt + (1.0 - self.alpha) * kd

    def get_config(self):
        return {**super().get_config(), "alpha": self.alpha}
