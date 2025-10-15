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
"""Fold batchnormalization with previous QDense layers."""

import keras
import keras.ops.numpy as knp
from keras import layers
from keras import ops as Kops

from .ops_portable import bias_add_portable
from .qlayers import QDense
from .quantizers import *


class QDenseBatchnorm(QDense):
    """Implements a quantized Dense layer fused with Batchnorm."""

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

        """A composite layer that folds QDense and batch normalization.

        The first group of parameters correponds to the initialization parameters
          of a QDense layer. check qkerasV3.qlayers.QDense
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
        super(QDenseBatchnorm, self).__init__(
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
        super(QDenseBatchnorm, self).build(input_shape)

        bn_input_shape = tuple([None] * (len(input_shape) - 1) + [self.units])
        self.batchnorm.build(bn_input_shape)

        # self._iteration (i.e., training_steps) is initialized with -1. When
        # loading ckpt, it can load the number of training steps that have been
        # previously trainied. If start training from scratch.
        # TODO(lishanok): develop a way to count iterations outside layer
        self._iteration = keras.Variable(
            -1, trainable=False, name="iteration", dtype="int64"
        )

    def call(self, inputs, training=None):
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

        kernel = self.kernel

        # execute qdense output
        qdense_outputs = keras.ops.dot(
            inputs, kernel
        )

        if self.use_bias:
            bias = self.bias
            qdense_outputs = bias_add_portable(
                qdense_outputs, bias, data_format="channels_last"
            )
        else:
            bias = 0

        # TODO(makoeppel): the following code is hacky:
        # since self._iteration is counted inside the layer
        # `bn_training` will be always a tensor in graph mode,
        # which can not be passed like training=bn_training.
        # Therefore, we have to call self.batchnorm with `training`
        # first to perform a forward pass and then use `bn_training`
        # later in keras.ops.where to assign the correct values.
        gamma_prev = self.batchnorm.gamma
        beta_prev = self.batchnorm.beta
        mm_prev = self.batchnorm.moving_mean
        mv_prev = self.batchnorm.moving_variance

        _ = self.batchnorm(qdense_outputs, training=training)

        gamma = Kops.where(bn_training, self.batchnorm.gamma, gamma_prev)
        beta = Kops.where(bn_training, self.batchnorm.beta, beta_prev)
        moving_mean = Kops.where(bn_training, self.batchnorm.moving_mean, mm_prev)
        moving_variance = Kops.where(bn_training, self.batchnorm.moving_variance, mv_prev)

        self.batchnorm.gamma.assign(gamma)
        self.batchnorm.beta.assign(beta)
        self.batchnorm.moving_mean.assign(moving_mean)
        self.batchnorm.moving_variance.assign(moving_variance)

        self._iteration.assign_add(
            keras.ops.where(
                Kops.cast(training, bool),
                knp.array(1, dtype="int64"),
                knp.array(0, dtype="int64")
            )
        )

        # calculate mean and variance from current batch
        bn_shape = qdense_outputs.shape
        ndims = len(bn_shape)
        axes = self.batchnorm.axis
        if isinstance(axes, int):
            axes = [axes]

        reduction_axes = [i for i in range(ndims) if i not in axes]
        keep_dims = len(axes) > 1

        mean, variance = keras.ops.moments(  # pylint: disable=protected-access
            keras.ops.cast(qdense_outputs, self.batchnorm.compute_dtype),  # pylint: disable=protected-access
            reduction_axes,
            keepdims=keep_dims,
        )

        if self.folding_mode == "batch_stats_folding":
            # using batch mean and variance in the initial training stage
            # after sufficient training, switch to moving mean and variance
            new_mean = keras.ops.where(
                Kops.cast(bn_training, bool), mean, moving_mean
            )
            new_variance = keras.ops.where(
                Kops.cast(bn_training, bool), variance, moving_variance
            )

            # get the inversion factor so that we replace division by multiplication
            inv = 1 / keras.ops.sqrt(new_variance + self.batchnorm.epsilon)
            if gamma is not None:
                inv *= gamma
            # fold bias with bn stats
            folded_bias = inv * (bias - new_mean) + beta

        elif self.folding_mode == "ema_stats_folding":
            # We always scale the weights with a correction factor to the long term
            # statistics prior to quantization. This ensures that there is no jitter
            # in the quantized weights due to batch to batch variation. During the
            # initial phase of training, we undo the scaling of the weights so that
            # outputs are identical to regular batch normalization. We also modify
            # the bias terms correspondingly. After sufficient training, switch from
            # using batch statistics to long term moving averages for batch
            # normalization.

            # use batch stats for calcuating bias before bn freeze, and use moving
            # stats after bn freeze
            mv_inv = 1 / keras.ops.sqrt(moving_variance + self.batchnorm.epsilon)
            batch_inv = 1 / keras.ops.sqrt(variance + self.batchnorm.epsilon)

            if gamma is not None:
                mv_inv *= gamma
                batch_inv *= gamma
            folded_bias = keras.ops.where(
                Kops.cast(bn_training, bool),
                batch_inv * (bias - mean) + beta,
                mv_inv * (bias - moving_mean) + beta,
            )
            # moving stats is always used to fold kernel in tflite; before bn freeze
            # an additional correction factor will be applied to the conv2d output
            inv = mv_inv
        else:
            assert ValueError

        # wrap conv kernel with bn parameters
        folded_kernel = inv * kernel
        # quantize the folded kernel
        if self.kernel_quantizer is not None:
            q_folded_kernel = self.kernel_quantizer_internal(folded_kernel)
        else:
            q_folded_kernel = folded_kernel

        # If loaded from a ckpt, bias_quantizer is the ckpt value
        # Else if bias_quantizer not specified, bias
        #   quantizer is None and we need to calculate bias quantizer
        #   type according to accumulator type. User can call
        #   bn_folding_utils.populate_bias_quantizer_from_accumulator(
        #      model, input_quantizer_list]) to populate such bias quantizer.
        if self.bias_quantizer_internal is not None:
            q_folded_bias = self.bias_quantizer_internal(folded_bias)
        else:
            q_folded_bias = folded_bias

        applied_kernel = q_folded_kernel
        applied_bias = q_folded_bias

        # calculate qdense output using the quantized folded kernel
        folded_outputs = keras.ops.dot(inputs, applied_kernel)

        if training and self.folding_mode == "ema_stats_folding":
            batch_inv = 1 / keras.ops.sqrt(variance + self.batchnorm.epsilon)
            y_corr = keras.ops.where(
                Kops.cast(bn_training, bool),
                keras.ops.sqrt(moving_variance + self.batchnorm.epsilon) * (1 / keras.ops.sqrt(variance + self.batchnorm.epsilon)),
                1.0
            )
            folded_outputs = folded_outputs * y_corr

        folded_outputs = bias_add_portable(
            folded_outputs, applied_bias, data_format="channels_last"
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
        }

    def get_quantizers(self):
        return self.quantizers

    def get_folded_weights(self):
        """Function to get the batchnorm folded weights.
        This function converts the weights by folding batchnorm parameters into
        the weight of QDense. The high-level equation:
        W_fold = gamma * W / sqrt(variance + epsilon)
        bias_fold = gamma * (bias - moving_mean) / sqrt(variance + epsilon) + beta
        """

        kernel = self.kernel
        if self.use_bias:
            bias = self.bias
        else:
            bias = 0

        # get batchnorm weights and moving stats
        gamma = self.batchnorm.gamma
        beta = self.batchnorm.beta
        moving_mean = self.batchnorm.moving_mean
        moving_variance = self.batchnorm.moving_variance

        # get the inversion factor so that we replace division by multiplication
        inv = 1 / keras.ops.sqrt(moving_variance + self.batchnorm.epsilon)
        if gamma is not None:
            inv *= gamma

        # wrap conv kernel and bias with bn parameters
        folded_kernel = inv * kernel
        folded_bias = inv * (bias - moving_mean) + beta

        return [folded_kernel, folded_bias]

    def save_own_variables(self, store):
        super().save_own_variables(store)
        # Stores the value of the variable upon saving
        store["iteration"] = keras.ops.convert_to_numpy(self._iteration)

    def load_own_variables(self, store):
        super().load_own_variables(store)
        # Assigns the value of the variable upon loading
        self._iteration.assign(store["iteration"])
