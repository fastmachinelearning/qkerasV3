# Copyright 2020 Google LLC
#
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
"""Fold batchnormalization with previous QDepthwiseConv2D layers."""

import keras
import keras.ops.numpy as knp
from keras import layers
from keras import ops as Kops

from .ops_portable import bias_add_portable
from .qconvolutional import QDepthwiseConv2D
from .quantizers import *


@register_keras_serializable(package="qkerasV3")
class QDepthwiseConv2DBatchnorm(QDepthwiseConv2D):
    """Fold batchnormalization with a previous QDepthwiseConv2d layer."""

    def __init__(
        self,
        # QDepthwiseConv2d params
        kernel_size,
        strides=(1, 1),
        padding="VALID",
        depth_multiplier=1,
        data_format=None,
        activation=None,
        use_bias=True,
        depthwise_initializer="he_normal",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        dilation_rate=(1, 1),
        depthwise_quantizer=None,
        bias_quantizer=None,
        depthwise_range=None,
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
        **kwargs,
    ):
        """A composite layer that folds depthwiseconv2d and batch normalization.

        The first group of parameters correponds to the initialization parameters
          of a QDepthwiseConv2d layer. check qkerasV3.qconvolutional.QDepthwiseConv2D
          for details.

        The 2nd group of parameters corresponds to the initialization parameters
          of a BatchNormalization layer. Check keras.layers.normalization.BatchNorma
          lizationBase for details.

        The 3rd group of parameters corresponds to the initialization parameters
          specific to this class.

          ema_freeze_delay: int or None. number of steps before batch normalization
            mv_mean and mv_variance will be frozen and used in the folded layer.
          folding_mode: string
            "ema_stats_folding": mimic tflite which uses the ema statistics to
              fold the kernel to suppress quantization induced jitter then performs
              the correction to have a similar effect of using the current batch
              statistics.
            "batch_stats_folding": use batch mean and variance to fold kernel first;
              after enough training steps switch to moving_mean and moving_variance
              for kernel folding.
        """
        kwargs.pop("synchronized", None)
        # intialization the QDepthwiseConv2d part of the composite layer
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            dilation_rate=dilation_rate,
            depthwise_quantizer=depthwise_quantizer,
            bias_quantizer=bias_quantizer,
            depthwise_range=depthwise_range,
            bias_range=bias_range,
            **kwargs,
        )

        # initialization of batchnorm part of the composite layer
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
        super(QDepthwiseConv2DBatchnorm, self).build(input_shape)

        if self.data_format == "channels_last":
            channel_axis = -1
        else:
            channel_axis = 1

        in_channels = int(input_shape[channel_axis])
        out_channels = in_channels * self.depth_multiplier

        if self.data_format == "channels_last":
            bn_input_shape = (input_shape[0], 1, 1, out_channels)
        else:  # channels_first
            bn_input_shape = (input_shape[0], out_channels, 1, 1)

        self.batchnorm.build(bn_input_shape)

        # If start training from scratch, self._iteration (i.e., training_steps)
        # is initialized with -1. When loading ckpt, it can load the number of
        # training steps that have been previously trainied.
        # TODO(lishanok): develop a way to count iterations outside layer
        self._iteration = keras.Variable(
            -1, trainable=False, name="iteration", dtype="int64"
        )

    def call(self, inputs, training=False):
        # default
        if training is None:
            training = False

        # Determine whether BatchNormalization layers should run in training mode.
        if (self.ema_freeze_delay is None) or (self.ema_freeze_delay < 0):
            bn_training = Kops.cast(training, dtype=bool)
        else:
            bn_training = Kops.logical_and(
                training, knp.less_equal(self._iteration, self.ema_freeze_delay)
            )

        depthwise_kernel = self.depthwise_kernel

        # Depthwise Convolution
        conv_outputs = keras.ops.depthwise_conv(
            inputs,
            depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

        if self.use_bias:
            bias = self.bias
            conv_outputs = bias_add_portable(
                conv_outputs, bias, data_format=self.data_format
            )
        else:
            bias = 0

        # Perform a forward pass through BatchNormalization once to ensure that
        # the moving statistics (mean and variance) are updated if in training mode.
        def call_batchnorm(conv_outputs, bn_training):
            return self.batchnorm(conv_outputs, training=Kops.cast(bn_training, bool))
        _ = call_batchnorm(conv_outputs, bn_training)

        self._iteration.assign_add(
            keras.ops.where(
                Kops.cast(training, bool),
                knp.array(1, dtype="int64"),
                knp.array(0, dtype="int64")
            )
        )

        bn_shape = conv_outputs.shape
        ndims = len(bn_shape)
        axes = self.batchnorm.axis
        if isinstance(axes, int):
            axes = [axes]

        reduction_axes = [i for i in range(ndims) if i not in axes]
        keep_dims = len(axes) > 1

        mean, variance = keras.ops.moments(conv_outputs, reduction_axes, keepdims=keep_dims)

        gamma = self.batchnorm.gamma
        beta = self.batchnorm.beta
        moving_mean = self.batchnorm.moving_mean
        moving_variance = self.batchnorm.moving_variance

        if self.folding_mode not in ["batch_stats_folding", "ema_stats_folding"]:
            raise ValueError(f"mode {self.folding_mode} not supported!")

        mv_inv = keras.ops.rsqrt(moving_variance + self.batchnorm.epsilon)
        batch_inv = keras.ops.rsqrt(variance + self.batchnorm.epsilon)

        if gamma is not None:
            mv_inv *= gamma
            batch_inv *= gamma

        folded_bias = keras.ops.where(
            Kops.cast(bn_training, bool),
            batch_inv * (bias - mean) + beta,
            mv_inv * (bias - moving_mean) + beta
        )

        if self.folding_mode == "batch_stats_folding":
            inv = keras.ops.where(
                Kops.cast(bn_training, bool),
                batch_inv,
                mv_inv
            )
        elif self.folding_mode == "ema_stats_folding":
            inv = mv_inv

        depthwise_weights_shape = [
            depthwise_kernel.shape[2],
            depthwise_kernel.shape[3],
        ]
        inv = knp.reshape(inv, depthwise_weights_shape)

        folded_depthwise_kernel = inv * depthwise_kernel

        if self.depthwise_quantizer is not None:
            q_folded_depthwise_kernel = self.depthwise_quantizer_internal(
                folded_depthwise_kernel
            )
        else:
            q_folded_depthwise_kernel = folded_depthwise_kernel

        if self.bias_quantizer is not None:
            q_folded_bias = self.bias_quantizer_internal(folded_bias)
        else:
            q_folded_bias = folded_bias

        applied_kernel = q_folded_depthwise_kernel
        applied_bias = q_folded_bias

        folded_outputs = keras.ops.depthwise_conv(
            inputs,
            applied_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

        if training is True and self.folding_mode == "ema_stats_folding":
            y_corr = keras.ops.where(
                Kops.cast(bn_training, bool),
                knp.sqrt(moving_variance + self.batchnorm.epsilon) * keras.ops.rsqrt(variance + self.batchnorm.epsilon),
                1
            )
            folded_outputs = folded_outputs * y_corr

        folded_outputs = bias_add_portable(
            folded_outputs, applied_bias, data_format=self.data_format
        )

        if self.activation is not None:
            return self.activation(folded_outputs)

        return folded_outputs

    def get_config(self):
        base_config = super().get_config()
        bn_config = self.batchnorm.get_config()
        config = {
            "ema_freeze_delay": self.ema_freeze_delay,
            "folding_mode": self.folding_mode,
        }
        name = base_config["name"]
        out_config = dict(
            list(base_config.items()) + list(bn_config.items()) + list(config.items())
        )

        # names from different config override each other; use the base layer name
        # as the this layer's config name
        out_config["name"] = name
        return out_config

    def get_quantization_config(self):
        return {
            "depthwise_quantizer": str(self.depthwise_quantizer_internal),
            "bias_quantizer": str(self.bias_quantizer_internal),
            "activation": str(self.activation),
            "filters": str(self.filters),
        }

    def get_quantizers(self):
        return self.quantizers

    def get_folded_weights(self):
        """Function to get the batchnorm folded weights.

        This function converts the weights by folding batchnorm parameters into
        the weight of QDepthwiseConv2d. The high-level equation:

        W_fold = gamma * W / sqrt(variance + epsilon)
        bias_fold = gamma * (bias - moving_mean) / sqrt(variance + epsilon) + beta
        """

        depthwise_kernel = self.depthwise_kernel

        if self.use_bias:
            bias = self.bias
        else:
            bias = 0

        # get Batchnorm stats
        gamma = self.batchnorm.gamma
        beta = self.batchnorm.beta
        moving_mean = self.batchnorm.moving_mean
        moving_variance = self.batchnorm.moving_variance

        # get the inversion factor so that we replace division by multiplication
        inv = 1 / keras.ops.sqrt(moving_variance + self.batchnorm.epsilon)
        if gamma is not None:
            inv *= gamma
        # fold bias with bn stats
        folded_bias = inv * (bias - moving_mean) + beta

        # for DepthwiseConv2D inv needs to be broadcasted to the last 2 dimensions
        # of the kernels
        depthwise_weights_shape = [depthwise_kernel.shape[2], depthwise_kernel.shape[3]]
        inv = keras.ops.reshape(inv, depthwise_weights_shape)
        # wrap conv kernel with bn parameters
        folded_depthwise_kernel = inv * depthwise_kernel

        return [folded_depthwise_kernel, folded_bias]

    def save_own_variables(self, store):
        super().save_own_variables(store)
        # Stores the value of the variable upon saving
        store["iteration"] = keras.ops.convert_to_numpy(self._iteration)

    def load_own_variables(self, store):
        super().load_own_variables(store)
        # Assigns the value of the variable upon loading
        self._iteration.assign(store["iteration"])
