"""
Keras 3 native weight pruning utilities.

The design is based on weight transforms:

    weight
       |
       v
    Pruner
       |
       v
    masked weight

A pruner does not know about Dense/Conv/etc.
Layers simply apply transforms before using weights.
"""

import random
import keras
from keras import ops
import numpy as np


@keras.saving.register_keras_serializable(package="qkeras")
class PruningSchedule:

    def __call__(self, step):
        raise NotImplementedError

    def get_config(self):
        raise NotImplementedError


@keras.saving.register_keras_serializable(package="qkeras")
class MaskGenerator:

    def __call__(self, weights, sparsity):
        raise NotImplementedError

    def get_config(self):
        raise NotImplementedError


@keras.saving.register_keras_serializable(package="qkeras")
class MagnitudeMask(MaskGenerator):
    """
    Removes smallest magnitude weights.
    """

    def __call__(self, weights, sparsity):

        weights = ops.convert_to_tensor(weights)
        if sparsity <= 0:
            return ops.ones_like(weights)
        if sparsity >= 1:
            return ops.zeros_like(weights)

        flat = ops.reshape(
            ops.abs(weights),
            (-1,),
        )
        k = int(
            flat.shape[0]
            * float(sparsity)
        )

        if k == 0:
            return ops.ones_like(weights)

        values = np.sort(
            np.asarray(flat)
        )
        threshold = values[k]
        mask = (
            ops.abs(weights)
            > threshold
        )

        return ops.cast(
            mask,
            weights.dtype,
        )


class MagnitudePruner:

    def __init__(
        self,
        initial_sparsity=0.0,
        final_sparsity=0.8,
        begin_epoch=0,
        end_epoch=10,
        update_probability=0.1,
    ):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.begin_epoch = begin_epoch
        self.end_epoch = end_epoch
        self.update_probability = update_probability

        self.begin_step = None
        self.end_step = None

        self.mask = None

    def build(self, shape, dtype="float32"):

        self.mask = keras.Variable(
            ops.ones(shape, dtype=dtype),
            trainable=False,
            name="pruning_mask",
        )

    def __call__(self, weights, training=False):

        if self.mask is None:
            return weights

        return (
            weights
            * keras.ops.stop_gradient(self.mask)
        )

    def set_steps_per_epoch(
        self,
        steps_per_epoch,
    ):

        self.begin_step = (
            self.begin_epoch * steps_per_epoch
        )

        self.end_step = (
            self.end_epoch * steps_per_epoch
        )

    def update(self, weights, step):

        if step < self.begin_step:
            return

        if step > self.end_step:
            return

        if random.random() > self.update_probability:
            return

        sparsity = self._sparsity(step)

        mask = self._compute_mask(
            weights,
            sparsity,
        )

        self.mask.assign(mask)

    def _sparsity(self, step):

        if step >= self.end_step:
            return self.final_sparsity

        x = (
            step - self.begin_step
        ) / (
            self.end_step - self.begin_step
        )

        x = max(0.0, min(1.0, x))

        return (
            self.final_sparsity
            +
            (
                self.initial_sparsity
                -
                self.final_sparsity
            )
            *
            (1 - x) ** 3
        )

    def sparsity(self):
        if self.mask is None:
            return 0.0

        return float(
            np.mean(
                self.mask.numpy() == 0
            )
        )

    def _compute_mask(self, weights, sparsity):

        flat = ops.reshape(
            ops.abs(weights),
            (-1,),
        )

        n = flat.shape[0]

        keep = int(
            n * (1.0 - sparsity)
        )

        values, indices = ops.top_k(
            flat,
            k=keep,
        )
        threshold = values[-1]

        return ops.cast(
            ops.abs(weights) >= threshold,
            weights.dtype,
        )


@keras.saving.register_keras_serializable(package="qkeras")
class TransformPipeline(keras.layers.Layer):
    """
    Compose multiple weight transforms.
    Example:
        kernel = pipeline(kernel)
    """

    def __init__(self, transforms, **kwargs):
        super().__init__(**kwargs)
        self.transforms = transforms

    def build(self, shape):
        for t in self.transforms:
            t.build(shape)

    def call(self, weights, training=False):

        for t in self.transforms:
            weights = t(
                weights,
                training=training,
            )

        return weights

    def update(self, weights, step):

        for t in self.transforms:
            if hasattr(t, "update"):
                t.update(
                    weights,
                    step,
                )


@keras.saving.register_keras_serializable(package="qkeras")
class PruningCallback(keras.callbacks.Callback):

    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose

    def on_train_begin(self, logs=None):

        steps = self.params["steps"]

        for layer in self.model.layers:

            if hasattr(layer, "kernel_pruner"):

                if layer.kernel_pruner:
                    layer.kernel_pruner.set_steps_per_epoch(
                        steps
                    )

    def on_train_batch_end(self, batch, logs=None):

        step = self.model.optimizer.iterations
        for layer in self.model.layers:
            if hasattr(
                layer,
                "update_pruning",
            ):
                layer.update_pruning(step)

    def on_epoch_end(self, epoch, logs=None):
        if self.verbose:
            self._report()

    def _report(self):

        print(
            f"\nPruning report"
        )

        total_weights = 0
        total_zero = 0

        for layer in self.model.layers:

            if not hasattr(layer, "kernel_pruner"):
                continue

            pruner = layer.kernel_pruner

            if pruner is None:
                continue

            if pruner.mask is None:
                continue

            mask = np.asarray(
                pruner.mask
            )

            zeros = np.sum(mask == 0)
            count = mask.size

            total_zero += zeros
            total_weights += count

            print(
                f"  {layer.name}: "
                f"{zeros}/{count} "
                f"({100.0 * zeros / count:.2f}%) pruned"
            )

        if total_weights:
            print(
                f"  Total sparsity: "
                f"{100.0 * total_zero / total_weights:.2f}%"
            )

    def get_config(self):
        return {
            **super().get_config(),
            "pruning_layers": self.pruning_layers
        }
