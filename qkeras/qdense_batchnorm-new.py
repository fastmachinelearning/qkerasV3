# Copyright 2020 Google LLC
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fold batchnormalization with previous QDense layers (JAX-friendly)."""

import keras
import keras.ops.numpy as knp
from keras import layers
from keras import ops as Kops

from .ops_portable import bias_add_portable
from .qlayers import QDense
from .quantizers import *


class QDenseBatchnorm(QDense):
    """Quantized Dense fused with BatchNorm."""

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        kernel_quantizer=None,
        bias_quantizer=None,
        kernel_range=None,
        bias_range=None,
        # batchnorm params
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        trainable=True,
        # other params
        ema_freeze_delay=None,
        folding_mode="ema_stats_folding",
        **kwargs
    ):
        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            kernel_quantizer=kernel_quantizer,
            bias_quantizer=bias_quantizer,
            kernel_range=kernel_range,
            bias_range=bias_range,
            **kwargs
        )

        # BatchNorm sub-layer (Keras handles moving-stats updates in a JAX-safe way)
        self.batchnorm = layers.BatchNormalization(
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            trainable=trainable,
        )

        self.ema_freeze_delay = ema_freeze_delay
        assert folding_mode in ["ema_stats_folding", "batch_stats_folding"]
        self.folding_mode = folding_mode

    def build(self, input_shape):
        super().build(input_shape)
        # Build BN with the Dense output shape
        bn_input_shape = tuple([None] * (len(input_shape) - 1) + [self.units])
        self.batchnorm.build(bn_input_shape)

        # Keep non-donated shadow copies for inference math
        self._gamma_ref = keras.Variable(
            self.batchnorm.gamma_initializer((self.units,)), trainable=False, dtype="float32"
        )
        self._beta_ref = keras.Variable(
            self.batchnorm.beta_initializer((self.units,)), trainable=False, dtype="float32"
        )
        self._moving_mean_ref = keras.Variable(
            self.batchnorm.moving_mean_initializer((self.units,)), trainable=False, dtype="float32"
        )
        self._moving_var_ref = keras.Variable(
            self.batchnorm.moving_variance_initializer((self.units,)), trainable=False, dtype="float32"
        )

    def _sync_bn_refs(self):
        """Copy BN weights into non-donated refs for safe use in JAX call()."""
        self._gamma_ref.assign(self.batchnorm.gamma)
        self._beta_ref.assign(self.batchnorm.beta)
        self._moving_mean_ref.assign(self.batchnorm.moving_mean)
        self._moving_var_ref.assign(self.batchnorm.moving_variance)

    def _compute_batch_stats(self, x, axes):
        """Backend-portable mean/var with keepdims matching BN axis handling."""
        if isinstance(axes, int):
            axes = [axes]
        ndims = len(x.shape)
        red_axes = tuple(i for i in range(ndims) if i not in axes)
        # knp.var uses population variance by default; match TF BN impl (unbiased=False).
        mean = knp.mean(x, axis=red_axes, keepdims=(len(axes) > 1))
        # Add keepdims to match mean shape for broadcasting
        var = knp.var(x, axis=red_axes, keepdims=(len(axes) > 1))
        return mean, var

    def _maybe_quantize(self, tensor, quantizer_internal):
        return quantizer_internal(tensor) if quantizer_internal is not None else tensor

    def call(self, inputs, training=None, step=None):
        """Forward pass.

        Args:
          inputs: (..., in_features)
          training: bool or boolean tensor.
          step: optional int/0-D tensor. If provided and `ema_freeze_delay` >= 0,
                BN will run in training mode only while `step <= ema_freeze_delay`.
                This keeps the function pure (no in-call state mutation).
        """
        training = Kops.cast(False if training is None else training, dtype=bool)

        # Decide if BN should run in "training" mode
        if (self.ema_freeze_delay is None) or (self.ema_freeze_delay < 0) or (step is None):
            bn_training = training
        else:
            # use provided step input to freeze after delay
            bn_training = Kops.logical_and(training, Kops.less_equal(step, self.ema_freeze_delay))

        kernel = self.kernel
        bias = self.bias if self.use_bias else 0

        # --- read BN parameters *before* calling the layer ---
        self._sync_bn_refs()
        gamma = self._gamma_ref
        beta = self._beta_ref
        moving_mean = self._moving_mean_ref
        moving_variance = self._moving_var_ref

        # Unfolded QDense matmul (+ optional bias)
        qdense_outputs = keras.ops.dot(inputs, kernel)
        if self.use_bias:
            qdense_outputs = bias_add_portable(qdense_outputs, bias, data_format="channels_last")

        # Let BN update its moving stats when appropriate (we ignore its output here).
        def _bn_update():
            # Run BN so moving stats update in training; ignore its output value.
            return self.batchnorm(qdense_outputs, training=True)

        def _bn_skip():
            # No-op path for inference.
            return qdense_outputs

        if isinstance(bn_training, bool):
            if bn_training:
                _ = self.batchnorm(qdense_outputs, training=True)
        else:
            _ = Kops.cond(bn_training, _bn_update, _bn_skip)

        # Compute batch stats for folding (pure; no side effects)
        mean, variance = self._compute_batch_stats(
            Kops.cast(qdense_outputs, self.batchnorm.compute_dtype),  # match BN dtype
            axes=self.batchnorm.axis
        )

        # Folding logic
        if self.folding_mode == "batch_stats_folding":
            if isinstance(bn_training, bool):
                if bn_training:
                    new_mean, new_var = mean, variance
                else:
                    new_mean, new_var = moving_mean, moving_variance
            else:
                new_mean = Kops.where(bn_training, mean, moving_mean)
                new_var  = Kops.where(bn_training, variance, moving_variance)
            inv = 1.0 / Kops.sqrt(new_var + eps)
            if gamma is not None:
                inv = inv * gamma
            folded_bias = inv * (bias - new_mean) + beta

        elif self.folding_mode == "ema_stats_folding":
            mv_inv = 1.0 / Kops.sqrt(moving_variance + eps)
            batch_inv = 1.0 / Kops.sqrt(variance + eps)
            if gamma is not None:
                mv_inv = mv_inv * gamma
                batch_inv = batch_inv * gamma
            folded_bias = Kops.where(
                bn_training,
                batch_inv * (bias - mean) + beta,
                mv_inv * (bias - moving_mean) + beta,
            )
            inv = mv_inv  # weights always folded with moving stats in this mode

        else:
            raise ValueError(f"Unknown folding_mode: {self.folding_mode}")

        # Fold kernel, then quantize (kernel/bias quantizers are applied AFTER folding)
        folded_kernel = inv * kernel
        q_folded_kernel = self._maybe_quantize(folded_kernel, self.kernel_quantizer_internal)
        q_folded_bias = self._maybe_quantize(folded_bias, self.bias_quantizer_internal)

        # Recompute output with folded, quantized params
        y = keras.ops.dot(inputs, q_folded_kernel)

        if (self.folding_mode == "ema_stats_folding"):
            # Apply correction during training phase to mimic batch-stat behavior pre-freeze
            # (matches original logic; no stateful ops).
            y_corr = Kops.where(
                bn_training,
                Kops.sqrt(moving_variance + eps) * (1.0 / Kops.sqrt(variance + eps)),
                1.0,
            )
            y = y * y_corr

        y = bias_add_portable(y, q_folded_bias, data_format="channels_last")

        if self.activation is not None:
            y = self.activation(y)
        return y

    def get_config(self):
        base_config = super().get_config()
        bn_config = self.batchnorm.get_config()
        config = {
            "ema_freeze_delay": self.ema_freeze_delay,
            "folding_mode": self.folding_mode,
        }
        name = base_config["name"]
        out_config = dict(list(base_config.items()) + list(bn_config.items()) + list(config.items()))
        out_config["name"] = name
        return out_config

    def get_quantization_config(self):
        return {
            "depthwise_quantizer": str(self.depthwise_quantizer_internal),
            "bias_quantizer": str(self.bias_quantizer_internal),
            "activation": str(self.activation),
        }

    def get_quantizers(self):
        return self.quantizers

    def get_folded_weights(self):
        """Return EMA-folded (kernel, bias) using moving stats only (no batch stats)."""
        kernel = self.kernel
        bias = self.bias if self.use_bias else 0
        gamma = self.batchnorm.gamma
        beta = self.batchnorm.beta
        moving_mean = self.batchnorm.moving_mean
        moving_variance = self.batchnorm.moving_variance
        inv = 1.0 / Kops.sqrt(moving_variance + self.batchnorm.epsilon)
        if gamma is not None:
            inv = inv * gamma
        folded_kernel = inv * kernel
        folded_bias = inv * (bias - moving_mean) + beta
        return [folded_kernel, folded_bias]
