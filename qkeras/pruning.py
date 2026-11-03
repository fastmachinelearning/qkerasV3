# Copyright 2025 QKeras-v3 authors.
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
"""Keras-3 native magnitude pruning for QKeras layers.

This module replaces the ``tensorflow-model-optimization`` (tfmot) sparsity API,
which does not support Keras 3. It works with the TensorFlow, JAX and PyTorch
backends.

Design
------
* A *pruner* (e.g. :class:`MagnitudePruner`) is a stateless, serializable
  configuration object that holds a :class:`PruningSchedule`.
* A prunable QKeras layer (``QDense``, ``QConv1D``, ``QConv2D``, ...) accepts a
  ``kernel_pruner`` argument. When it is set, the layer creates one
  non-trainable ``<weight>_pruning_mask`` variable per prunable weight and uses
  ``weight * mask`` in its forward pass, *before* quantization. Layers without a
  pruner are unchanged (no extra weights).
* Mask updates are not part of the compiled train step. They are performed by
  the :class:`UpdatePruningStep` callback between steps, following the schedule.
  On the JAX backend the callback synchronises the functional training state
  before reading/writing variables.
* At the end of training the masks are baked into the weights. Use
  :func:`strip_pruning` to obtain a plain QKeras model (e.g. for hls4ml).

Typical usage::

    from qkeras import QDense
    from qkeras.pruning import (MagnitudePruner, PolynomialDecay,
                                UpdatePruningStep, strip_pruning)

    pruner = MagnitudePruner(PolynomialDecay(0.0, 0.75, begin_step=200,
                                             end_step=2000, frequency=100))
    x = QDense(64, kernel_quantizer="quantized_bits(6,0,1)",
               kernel_pruner=pruner)(inputs)
    ...
    model.fit(..., callbacks=[UpdatePruningStep()])
    model = strip_pruning(model)

or, migrating tfmot code::

    model = prune_low_magnitude(model, MagnitudePruner(...))
"""

import math
import warnings

import keras
import numpy as np
from keras import ops

__all__ = [
    "PruningSchedule",
    "ConstantSparsity",
    "PolynomialDecay",
    "Pruner",
    "MagnitudePruner",
    "PrunableLayerMixin",
    "PruningMasks",
    "UpdatePruningStep",
    "PruningCallback",
    "get_prunable_layers",
    "get_pruning_summary",
    "apply_pruning_masks",
    "prune_low_magnitude",
    "strip_pruning",
]

MASK_SUFFIX = "_pruning_mask"


# -----------------------------------------------------------------------------
# Schedules
# -----------------------------------------------------------------------------
class PruningSchedule:
    """Base class: decides *when* to update masks and to *which* sparsity.

    Updates happen at ``begin_step + k * frequency`` (k >= 0) and at
    ``end_step``. ``end_step = -1`` means "never stop".

    Steps count optimizer updates (``model.optimizer.iterations``).
    """

    def __init__(self, begin_step=0, end_step=-1, frequency=100):
        begin_step, end_step, frequency = int(begin_step), int(end_step), int(frequency)
        if begin_step < 0:
            raise ValueError(f"begin_step must be >= 0, got {begin_step}.")
        if end_step != -1 and end_step < begin_step:
            raise ValueError(
                f"end_step ({end_step}) must be -1 or >= begin_step ({begin_step})."
            )
        if frequency < 1:
            raise ValueError(f"frequency must be >= 1, got {frequency}.")
        self.begin_step = begin_step
        self.end_step = end_step
        self.frequency = frequency

    def target_sparsity(self, step):
        """Sparsity the schedule prescribes at ``step`` (float in [0, 1])."""
        raise NotImplementedError

    def __call__(self, step, prev_step=None):
        """Returns the sparsity to apply if an update is due in (prev_step, step].

        ``prev_step`` defaults to ``step - 1`` (i.e. only ``step`` is checked).
        Checking a window makes the schedule robust to ``steps_per_execution > 1``,
        where callbacks are not invoked on every optimizer step.

        Returns:
          ``None`` if no update is due, else the target sparsity.
        """
        step = int(step)
        prev_step = step - 1 if prev_step is None else int(prev_step)
        lo = max(prev_step + 1, self.begin_step)
        hi = step if self.end_step == -1 else min(step, self.end_step)
        if lo > hi:
            return None
        k = math.ceil((lo - self.begin_step) / self.frequency)
        due = self.begin_step + k * self.frequency <= hi
        if self.end_step != -1 and lo <= self.end_step <= hi:
            due = True
        if not due:
            return None
        return float(self.target_sparsity(hi))

    def get_config(self):
        return {
            "begin_step": self.begin_step,
            "end_step": self.end_step,
            "frequency": self.frequency,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_config().items())
        return f"{type(self).__name__}({args})"


def _check_sparsity(name, value):
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {value}.")
    return value


@keras.saving.register_keras_serializable(package="qkeras")
class ConstantSparsity(PruningSchedule):
    """Constant target sparsity, applied every ``frequency`` steps.

    Equivalent to ``tfmot.sparsity.keras.ConstantSparsity``.
    """

    def __init__(self, target_sparsity, begin_step=0, end_step=-1, frequency=100):
        super().__init__(begin_step, end_step, frequency)
        self.target_sparsity_value = _check_sparsity("target_sparsity", target_sparsity)

    def target_sparsity(self, step):
        return self.target_sparsity_value

    def get_config(self):
        return {"target_sparsity": self.target_sparsity_value, **super().get_config()}


@keras.saving.register_keras_serializable(package="qkeras")
class PolynomialDecay(PruningSchedule):
    """Sparsity ramps from ``initial_sparsity`` to ``final_sparsity``.

    ``s(t) = s_f + (s_i - s_f) * (1 - p)**power`` with
    ``p = (t - begin_step) / (end_step - begin_step)``, clipped to [0, 1].
    Equivalent to ``tfmot.sparsity.keras.PolynomialDecay`` (Zhu & Gupta, 2017).
    """

    def __init__(
        self,
        initial_sparsity,
        final_sparsity,
        begin_step,
        end_step,
        power=3,
        frequency=100,
    ):
        if int(end_step) < 0:
            raise ValueError("PolynomialDecay requires end_step >= 0.")
        if int(end_step) <= int(begin_step):
            raise ValueError("PolynomialDecay requires end_step > begin_step.")
        super().__init__(begin_step, end_step, frequency)
        self.initial_sparsity = _check_sparsity("initial_sparsity", initial_sparsity)
        self.final_sparsity = _check_sparsity("final_sparsity", final_sparsity)
        self.power = float(power)

    def target_sparsity(self, step):
        p = (step - self.begin_step) / (self.end_step - self.begin_step)
        p = min(1.0, max(0.0, p))
        return (
            self.final_sparsity
            + (self.initial_sparsity - self.final_sparsity) * (1.0 - p) ** self.power
        )

    def get_config(self):
        return {
            "initial_sparsity": self.initial_sparsity,
            "final_sparsity": self.final_sparsity,
            "begin_step": self.begin_step,
            "end_step": self.end_step,
            "power": self.power,
            "frequency": self.frequency,
        }


# -----------------------------------------------------------------------------
# Pruners
# -----------------------------------------------------------------------------
class Pruner:
    """Base class: computes a {0, 1} mask for a weight at a given sparsity."""

    def __init__(self, schedule):
        if isinstance(schedule, (float, int)) and not isinstance(schedule, bool):
            schedule = ConstantSparsity(schedule)
        elif isinstance(schedule, dict):
            schedule = keras.saving.deserialize_keras_object(schedule)
        if not isinstance(schedule, PruningSchedule):
            raise TypeError(
                "schedule must be a PruningSchedule or a float target sparsity, "
                f"got {type(schedule).__name__}."
            )
        self.schedule = schedule

    def compute_mask(self, weight, sparsity):
        raise NotImplementedError

    def get_config(self):
        return {"schedule": keras.saving.serialize_keras_object(self.schedule)}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __repr__(self):
        return f"{type(self).__name__}({self.schedule!r})"


@keras.saving.register_keras_serializable(package="qkeras")
class MagnitudePruner(Pruner):
    """Unstructured magnitude pruning: zeroes the smallest ``|w|``.

    Exactly ``round(sparsity * w.size)`` entries are masked (ties are broken
    arbitrarily), so the realised sparsity equals the target.

    Args:
      schedule: a :class:`PruningSchedule`, or a float for
        ``ConstantSparsity(schedule)``.
    """

    def compute_mask(self, weight, sparsity):
        weight = ops.convert_to_tensor(weight)
        shape = tuple(weight.shape)
        n = int(np.prod(shape))
        keep = n - int(round(float(sparsity) * n))
        keep = min(n, max(0, keep))
        if keep == n:
            return ops.ones(shape, dtype=weight.dtype)
        if keep == 0:
            return ops.zeros(shape, dtype=weight.dtype)
        flat = ops.reshape(ops.abs(weight), (-1,))
        _, idx = ops.top_k(flat, k=keep, sorted=False)
        mask = ops.scatter(
            ops.expand_dims(idx, -1),
            ops.ones((keep,), dtype=weight.dtype),
            (n,),
        )
        return ops.reshape(mask, shape)


def _deserialize_pruner(pruner):
    if pruner is None:
        return None
    if isinstance(pruner, dict):
        pruner = keras.saving.deserialize_keras_object(pruner)
    if not isinstance(pruner, Pruner):
        raise TypeError(
            "kernel_pruner must be a qkeras.pruning.Pruner, got "
            f"{type(pruner).__name__}."
        )
    return pruner


# -----------------------------------------------------------------------------
# Layer integration
# -----------------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="qkeras")
class PruningMasks(keras.layers.Layer):
    """Internal container holding the {0, 1} pruning masks of one layer."""

    def __init__(self, **kwargs):
        kwargs.setdefault("trainable", False)
        super().__init__(**kwargs)
        object.__setattr__(self, "masks", {})

    def add_mask(self, name, shape, dtype):
        mask = self.add_weight(
            name=name + MASK_SUFFIX,
            shape=shape,
            initializer="ones",
            trainable=False,
            dtype=dtype,
        )
        self.masks[name] = mask
        return mask

    def call(self, inputs):  # pragma: no cover - never called
        return inputs


class PrunableLayerMixin:
    """Adds ``kernel_pruner`` support to a Keras layer.

    Subclasses set ``_PRUNABLE_WEIGHTS`` (attribute names of the weights to
    prune), call ``_init_pruning(kernel_pruner)`` in ``__init__``,
    ``_build_pruning()`` at the end of ``build`` and wrap weights with
    ``_apply_pruning(name, weight)`` in ``call``.
    """

    _PRUNABLE_WEIGHTS = ("kernel",)

    def _init_pruning(self, kernel_pruner):
        self.kernel_pruner = _deserialize_pruner(kernel_pruner)

    def _build_pruning(self):
        if getattr(self, "kernel_pruner", None) is None:
            return
        # Masks live in a tracked, non-trainable sublayer. This leaves the
        # host layer's own variable list (and therefore the parent class'
        # save_own_variables/load_own_variables logic, e.g. keras Dense/Conv)
        # untouched, while Keras still saves/loads the masks as a child
        # object and includes them in model.non_trainable_variables (JAX state).
        holder = PruningMasks(name="pruning_masks")
        for name in self._PRUNABLE_WEIGHTS:
            weight = getattr(self, name)
            holder.add_mask(name, tuple(weight.shape), weight.dtype)
        holder.built = True
        self.pruning_mask_holder = holder

    def pruning_masks(self):
        """Dict {weight_name: mask_variable}; empty if the layer is not pruned."""
        holder = getattr(self, "pruning_mask_holder", None)
        if getattr(self, "kernel_pruner", None) is None or holder is None:
            return {}
        return dict(holder.masks)

    def _apply_pruning(self, name, weight):
        if getattr(self, "kernel_pruner", None) is None:
            return weight
        holder = getattr(self, "pruning_mask_holder", None)
        mask = None if holder is None else holder.masks.get(name)
        if mask is None:
            return weight
        return weight * ops.cast(mask, weight.dtype)

    def update_pruning(self, step, prev_step=None):
        """Recomputes masks if the schedule requires it. Eager-only.

        Returns True if the masks were updated.
        """
        masks = self.pruning_masks()
        if not masks:
            return False
        sparsity = self.kernel_pruner.schedule(step, prev_step)
        if sparsity is None:
            return False
        for name, mask in masks.items():
            weight = getattr(self, name)
            # Rank the *effective* weights: already pruned entries stay pruned.
            effective = ops.convert_to_tensor(weight) * ops.cast(mask, weight.dtype)
            new_mask = self.kernel_pruner.compute_mask(effective, sparsity)
            mask.assign(ops.cast(new_mask, mask.dtype))
            weight.assign(effective * ops.cast(new_mask, weight.dtype))
        return True

    def apply_pruning_masks(self):
        """Bakes the masks into the weights (``w <- w * mask``)."""
        for name, mask in self.pruning_masks().items():
            weight = getattr(self, name)
            weight.assign(ops.convert_to_tensor(weight) * ops.cast(mask, weight.dtype))

    def _pruning_config(self):
        return {
            "kernel_pruner": keras.saving.serialize_keras_object(self.kernel_pruner)
            if getattr(self, "kernel_pruner", None) is not None
            else None
        }


# -----------------------------------------------------------------------------
# Model-level helpers
# -----------------------------------------------------------------------------
def _iter_layers(model):
    if hasattr(model, "_flatten_layers"):
        yield from model._flatten_layers(include_self=False, recursive=True)
    else:  # pragma: no cover
        yield from model.layers


def get_prunable_layers(model):
    """Layers of ``model`` (recursively) that carry pruning masks."""
    return [
        layer
        for layer in _iter_layers(model)
        if isinstance(layer, PrunableLayerMixin) and layer.pruning_masks()
    ]


def _sync_jax_state(model):
    # JAX trainer keeps variables in a functional state during fit(); this
    # writes them back so callbacks can read/assign, and makes the next train
    # step pick up our assignments.
    sync = getattr(model, "jax_state_sync", None)
    if callable(sync):
        sync()


def get_pruning_summary(model):
    """Effective sparsity of every pruned weight.

    Returns:
      dict ``{layer_name: {weight_name: sparsity}}`` and the key
      ``"__total__"`` with the overall sparsity of all pruned weights.
    """
    summary, zeros, total = {}, 0, 0
    for layer in get_prunable_layers(model):
        entry = {}
        for name, mask in layer.pruning_masks().items():
            w = ops.convert_to_numpy(getattr(layer, name)) * ops.convert_to_numpy(mask)
            z = int(np.sum(w == 0))
            entry[name] = z / w.size
            zeros += z
            total += w.size
        summary[layer.name] = entry
    summary["__total__"] = zeros / total if total else 0.0
    return summary


def apply_pruning_masks(model):
    """Bakes all masks into the weights of ``model`` in place."""
    _sync_jax_state(model)
    for layer in get_prunable_layers(model):
        layer.apply_pruning_masks()


@keras.saving.register_keras_serializable(package="qkeras")
class UpdatePruningStep(keras.callbacks.Callback):
    """Updates pruning masks during ``fit`` according to each layer's schedule.

    Required whenever a model with ``kernel_pruner`` layers is trained (same
    role as ``tfmot.sparsity.keras.UpdatePruningStep``).

    Args:
      verbose: if truthy, print a per-layer sparsity report after each epoch.
      bake_on_train_end: if True (default), apply ``w <- w * mask`` when training
        ends, so the stored weights are exactly sparse even without
        :func:`strip_pruning`.
    """

    def __init__(self, verbose=0, bake_on_train_end=True):
        super().__init__()
        self.verbose = verbose
        self.bake_on_train_end = bake_on_train_end
        self._layers = []
        self._step = 0
        self._prev_step = 0
        self._epoch_base = 0

    def on_train_begin(self, logs=None):
        self._layers = get_prunable_layers(self.model)
        if not self._layers:
            warnings.warn(
                "UpdatePruningStep: the model has no layers with a kernel_pruner.",
                stacklevel=2,
            )
        # Variables are in sync at this point on every backend.
        self._step = int(ops.convert_to_numpy(self.model.optimizer.iterations))
        self._prev_step = self._step
        self._epoch_base = self._step

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_base = self._step

    def on_train_batch_end(self, batch, logs=None):
        # ``batch`` is the index of the last step executed in this epoch (also
        # with steps_per_execution > 1), so we can count optimizer steps without
        # reading (and on JAX: syncing) model.optimizer.iterations.
        self._step = self._epoch_base + int(batch) + 1
        prev, step = self._prev_step, self._step
        self._prev_step = step
        due = [
            layer
            for layer in self._layers
            if layer.kernel_pruner.schedule(step, prev) is not None
        ]
        if not due:
            return
        _sync_jax_state(self.model)
        for layer in due:
            layer.update_pruning(step, prev)

    def on_epoch_end(self, epoch, logs=None):
        if self.verbose and self._layers:
            _sync_jax_state(self.model)
            summary = get_pruning_summary(self.model)
            total = summary.pop("__total__")
            print(f"\nPruning summary (epoch {epoch + 1}, step {self._step}):")
            for lname, entry in summary.items():
                txt = ", ".join(f"{k}={100 * v:.1f}%" for k, v in entry.items())
                print(f"  {lname}: {txt}")
            print(f"  total sparsity of pruned weights: {100 * total:.1f}%")

    def on_train_end(self, logs=None):
        if self.bake_on_train_end and self._layers:
            apply_pruning_masks(self.model)

    def get_config(self):
        return {"verbose": self.verbose, "bake_on_train_end": self.bake_on_train_end}


# Backwards-compatible name used on the development branch.
PruningCallback = UpdatePruningStep


def _copy_weights(src_layer, dst_layer, bake_masks):
    """Copies weights between two versions of a layer, matching by name."""
    src = {v.name: v for v in src_layer.weights}
    masks = (
        src_layer.pruning_masks() if isinstance(src_layer, PrunableLayerMixin) else {}
    )
    for v in dst_layer.weights:
        if v.name not in src:
            continue  # e.g. freshly created pruning masks
        value = ops.convert_to_numpy(src[v.name])
        if bake_masks:
            # find the weight whose name matches (masks keyed by attribute name)
            for wname, mask in masks.items():
                if getattr(src_layer, wname) is src[v.name]:
                    value = value * ops.convert_to_numpy(mask)
        v.assign(value)


def _is_clonable(model):
    from keras.src.models.functional import Functional

    return isinstance(model, (keras.Sequential, Functional))


def _clone(model, clone_layer, bake_masks):
    if not _is_clonable(model):
        raise ValueError(
            "Only Sequential and functional models can be cloned. For "
            "subclassed models set `kernel_pruner` in the layer constructors "
            "and use `apply_pruning_masks(model)` after training."
        )
    _sync_jax_state(model)
    new_model = keras.models.clone_model(
        model, clone_function=clone_layer, recursive=True
    )
    old_layers = {}
    for layer in _iter_layers(model):
        old_layers.setdefault(layer.name, layer)
    for new_layer in _iter_layers(new_model):
        if isinstance(new_layer, keras.Model):
            continue  # weights are copied through its sub-layers
        old_layer = old_layers.get(new_layer.name)
        if old_layer is not None:
            _copy_weights(old_layer, new_layer, bake_masks)
    return new_model


def _clone_with_pruner(layer, pruner):
    config = layer.get_config()
    config["kernel_pruner"] = pruner
    return layer.__class__.from_config(config)


def prune_low_magnitude(model, pruner, layer_names=None):
    """Returns a copy of ``model`` with ``kernel_pruner`` set on prunable layers.

    Keras-3 counterpart of ``tfmot.sparsity.keras.prune_low_magnitude``. Only
    Sequential and functional models are supported; weights are copied.

    Args:
      model: Sequential or functional Keras model.
      pruner: a :class:`Pruner`, a :class:`PruningSchedule` or a float
        (constant sparsity). Wrapped in :class:`MagnitudePruner` if needed.
      layer_names: optional iterable of layer names to prune (default: every
        QKeras layer that supports pruning).
    """
    if not isinstance(pruner, Pruner):
        pruner = MagnitudePruner(pruner)
    names = None if layer_names is None else set(layer_names)
    found = set()

    def clone_layer(layer):
        if isinstance(layer, PrunableLayerMixin) and (
            names is None or layer.name in names
        ):
            found.add(layer.name)
            return _clone_with_pruner(layer, pruner)
        return layer.__class__.from_config(layer.get_config())

    new_model = _clone(model, clone_layer, bake_masks=False)
    if names is not None and names - found:
        raise ValueError(f"Layers not found or not prunable: {sorted(names - found)}.")
    return new_model


def strip_pruning(model):
    """Returns a copy of ``model`` without pruning masks/pruners.

    The masks are multiplied into the weights, so the result is an ordinary
    QKeras model with exactly-zero pruned weights (suitable for hls4ml).
    """

    def clone_layer(layer):
        if isinstance(layer, PrunableLayerMixin) and layer.pruning_masks():
            return _clone_with_pruner(layer, None)
        return layer.__class__.from_config(layer.get_config())

    return _clone(model, clone_layer, bake_masks=True)
