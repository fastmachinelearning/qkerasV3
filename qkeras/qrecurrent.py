# Copyright 2020 Google LLC
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
"""Quantized recurrent layers for Keras 3 / qkerasV3."""

import keras
from keras import activations, constraints, initializers, layers, regularizers
from keras.saving import register_keras_serializable, serialize_keras_object

from .ops_portable import is_nested
from .qlayers import get_auto_range_constraint_initializer
from .quantizers import get_quantizer


ops = keras.ops


def _serialize_quantizer(quantizer):
    if quantizer is None:
        return None
    return serialize_keras_object(quantizer)


def _dot(x, kernel):
    return ops.matmul(x, kernel)


def _bias_add(x, bias):
    return x + bias


def _get_dropout_mask(cell, inputs, count):
    if 0.0 < cell.dropout < 1.0:
        mask = cell.get_dropout_mask(inputs)
        if isinstance(mask, (list, tuple)):
            return list(mask)
        return [mask for _ in range(count)]
    return [None for _ in range(count)]


def _get_recurrent_dropout_mask(cell, state, count):
    if 0.0 < cell.recurrent_dropout < 1.0:
        mask = cell.get_recurrent_dropout_mask(state)
        if isinstance(mask, (list, tuple)):
            return list(mask)
        return [mask for _ in range(count)]
    return [None for _ in range(count)]


@register_keras_serializable(package="qkeras")
class QSimpleRNNCell(layers.SimpleRNNCell):
    """Quantized SimpleRNN cell."""

    def __init__(
        self,
        units,
        activation="quantized_tanh",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        kernel_quantizer=None,
        recurrent_quantizer=None,
        bias_quantizer=None,
        state_quantizer=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        **kwargs,
    ):
        self.kernel_quantizer = kernel_quantizer
        self.recurrent_quantizer = recurrent_quantizer
        self.bias_quantizer = bias_quantizer
        self.state_quantizer = state_quantizer

        self.kernel_quantizer_internal = get_quantizer(kernel_quantizer)
        self.recurrent_quantizer_internal = get_quantizer(recurrent_quantizer)
        self.bias_quantizer_internal = get_quantizer(bias_quantizer)
        self.state_quantizer_internal = get_quantizer(state_quantizer)

        self.quantizers = [
            self.kernel_quantizer_internal,
            self.recurrent_quantizer_internal,
            self.bias_quantizer_internal,
            self.state_quantizer_internal,
        ]

        for quantizer in [
            self.kernel_quantizer_internal,
            self.recurrent_quantizer_internal,
        ]:
            if hasattr(quantizer, "_set_trainable_parameter"):
                quantizer._set_trainable_parameter()

        kernel_constraint, kernel_initializer = get_auto_range_constraint_initializer(
            self.kernel_quantizer_internal, kernel_constraint, kernel_initializer
        )
        recurrent_constraint, recurrent_initializer = get_auto_range_constraint_initializer(
            self.recurrent_quantizer_internal,
            recurrent_constraint,
            recurrent_initializer,
        )
        if use_bias:
            bias_constraint, bias_initializer = get_auto_range_constraint_initializer(
                self.bias_quantizer_internal, bias_constraint, bias_initializer
            )

        super().__init__(
            units=units,
            activation=get_quantizer(activation) if activation is not None else None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            seed=seed,
            **kwargs,
        )

    def call(self, inputs, states, training=False):
        prev_output = states[0] if is_nested(states) else states

        if self.state_quantizer:
            prev_output = self.state_quantizer_internal(prev_output)

        dp_mask = _get_dropout_mask(self, inputs, 1) if training else [None]
        rec_dp_mask = _get_recurrent_dropout_mask(self, prev_output, 1) if training else [None]

        quantized_kernel = (
            self.kernel_quantizer_internal(self.kernel)
            if self.kernel_quantizer
            else self.kernel
        )
        quantized_recurrent = (
            self.recurrent_quantizer_internal(self.recurrent_kernel)
            if self.recurrent_quantizer
            else self.recurrent_kernel
        )

        inputs_i = inputs * dp_mask[0] if dp_mask[0] is not None else inputs
        prev_output_i = (
            prev_output * rec_dp_mask[0]
            if rec_dp_mask[0] is not None
            else prev_output
        )

        h = _dot(inputs_i, quantized_kernel)
        if self.bias is not None:
            quantized_bias = (
                self.bias_quantizer_internal(self.bias) if self.bias_quantizer else self.bias
            )
            h = _bias_add(h, quantized_bias)

        output = h + _dot(prev_output_i, quantized_recurrent)
        if self.activation is not None:
            output = self.activation(output)
        return output, [output]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_quantizer": _serialize_quantizer(self.kernel_quantizer_internal),
                "recurrent_quantizer": _serialize_quantizer(
                    self.recurrent_quantizer_internal
                ),
                "bias_quantizer": _serialize_quantizer(self.bias_quantizer_internal),
                "state_quantizer": _serialize_quantizer(self.state_quantizer_internal),
            }
        )
        return config


@register_keras_serializable(package="qkeras")
class QSimpleRNN(layers.RNN):
    """Quantized SimpleRNN layer."""

    def __init__(self, units, activity_regularizer=None, **kwargs):
        rnn_kwargs = _pop_rnn_kwargs(kwargs)
        cell = QSimpleRNNCell(
            units,
            **kwargs,
        )
        super().__init__(cell, **rnn_kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [layers.InputSpec(ndim=3)]

    def call(self, sequences, initial_state=None, mask=None, training=False):
        return super().call(
            sequences,
            initial_state=initial_state,
            mask=mask,
            training=training,
        )

    def compute_output_shape(self, inputs_shape):
        return super().compute_output_shape(inputs_shape)

    def get_quantizers(self):
        return self.cell.quantizers

    def get_prunable_weights(self):
        return [self.cell.kernel, self.cell.recurrent_kernel]

    def get_quantization_config(self):
        return {
            "kernel_quantizer": str(self.cell.kernel_quantizer_internal),
            "recurrent_quantizer": str(self.cell.recurrent_quantizer_internal),
            "bias_quantizer": str(self.cell.bias_quantizer_internal),
            "state_quantizer": str(self.cell.state_quantizer_internal),
            "activation": str(self.cell.activation),
        }

    def get_config(self):
        base_config = super().get_config()
        base_config.pop("cell", None)
        base_config.update(_cell_config(self.cell))
        base_config["activity_regularizer"] = regularizers.serialize(
            self.activity_regularizer
        )
        return base_config

    @classmethod
    def from_config(cls, config):
        config.pop("implementation", None)
        return cls(**config)


@register_keras_serializable(package="qkeras")
class QLSTMCell(layers.LSTMCell):
    """Quantized LSTM cell."""

    def __init__(
        self,
        units,
        activation="quantized_tanh",
        recurrent_activation="hard_sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        kernel_quantizer=None,
        recurrent_quantizer=None,
        bias_quantizer=None,
        state_quantizer=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        implementation=1,
        seed=None,
        **kwargs,
    ):
        implementation = 1 if implementation == 0 else implementation
        self.kernel_quantizer = kernel_quantizer
        self.recurrent_quantizer = recurrent_quantizer
        self.bias_quantizer = bias_quantizer
        self.state_quantizer = state_quantizer

        self.kernel_quantizer_internal = get_quantizer(kernel_quantizer)
        self.recurrent_quantizer_internal = get_quantizer(recurrent_quantizer)
        self.bias_quantizer_internal = get_quantizer(bias_quantizer)
        self.state_quantizer_internal = get_quantizer(state_quantizer)
        self.quantizers = [
            self.kernel_quantizer_internal,
            self.recurrent_quantizer_internal,
            self.bias_quantizer_internal,
            self.state_quantizer_internal,
        ]

        for quantizer in [
            self.kernel_quantizer_internal,
            self.recurrent_quantizer_internal,
        ]:
            if hasattr(quantizer, "_set_trainable_parameter"):
                quantizer._set_trainable_parameter()

        kernel_constraint, kernel_initializer = get_auto_range_constraint_initializer(
            self.kernel_quantizer_internal, kernel_constraint, kernel_initializer
        )
        recurrent_constraint, recurrent_initializer = get_auto_range_constraint_initializer(
            self.recurrent_quantizer_internal,
            recurrent_constraint,
            recurrent_initializer,
        )
        if use_bias:
            bias_constraint, bias_initializer = get_auto_range_constraint_initializer(
                self.bias_quantizer_internal, bias_constraint, bias_initializer
            )

        super().__init__(
            units=units,
            activation=get_quantizer(activation) if activation is not None else None,
            recurrent_activation=(
                get_quantizer(recurrent_activation)
                if recurrent_activation is not None
                else None
            ),
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            seed=seed,
            **kwargs,
        )

        self.implementation = implementation

    def _compute_carry_and_output(self, x, h_tm1, c_tm1, quantized_recurrent):
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        i = self.recurrent_activation(
            x_i + _dot(h_tm1_i, quantized_recurrent[:, : self.units])
        )
        f = self.recurrent_activation(
            x_f + _dot(h_tm1_f, quantized_recurrent[:, self.units : self.units * 2])
        )
        c = f * c_tm1 + i * self.activation(
            x_c + _dot(h_tm1_c, quantized_recurrent[:, self.units * 2 : self.units * 3])
        )
        o = self.recurrent_activation(
            x_o + _dot(h_tm1_o, quantized_recurrent[:, self.units * 3 :])
        )
        return c, o

    def _compute_carry_and_output_fused(self, z, c_tm1):
        z0, z1, z2, z3 = z
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)
        return c, o

    def call(self, inputs, states, training=False):
        h_tm1 = states[0]
        c_tm1 = states[1]

        if self.state_quantizer:
            h_tm1 = self.state_quantizer_internal(h_tm1)
            c_tm1 = self.state_quantizer_internal(c_tm1)

        dp_mask = _get_dropout_mask(self, inputs, 4) if training else [None] * 4
        rec_dp_mask = (
            _get_recurrent_dropout_mask(self, h_tm1, 4) if training else [None] * 4
        )

        quantized_kernel = (
            self.kernel_quantizer_internal(self.kernel)
            if self.kernel_quantizer
            else self.kernel
        )
        quantized_recurrent = (
            self.recurrent_quantizer_internal(self.recurrent_kernel)
            if self.recurrent_quantizer
            else self.recurrent_kernel
        )
        quantized_bias = None
        if self.use_bias:
            quantized_bias = (
                self.bias_quantizer_internal(self.bias) if self.bias_quantizer else self.bias
            )

        if self.implementation == 1:
            inputs_i = inputs * dp_mask[0] if dp_mask[0] is not None else inputs
            inputs_f = inputs * dp_mask[1] if dp_mask[1] is not None else inputs
            inputs_c = inputs * dp_mask[2] if dp_mask[2] is not None else inputs
            inputs_o = inputs * dp_mask[3] if dp_mask[3] is not None else inputs

            k_i, k_f, k_c, k_o = ops.split(quantized_kernel, 4, axis=1)
            x_i = _dot(inputs_i, k_i)
            x_f = _dot(inputs_f, k_f)
            x_c = _dot(inputs_c, k_c)
            x_o = _dot(inputs_o, k_o)

            if self.use_bias:
                b_i, b_f, b_c, b_o = ops.split(quantized_bias, 4, axis=0)
                x_i = _bias_add(x_i, b_i)
                x_f = _bias_add(x_f, b_f)
                x_c = _bias_add(x_c, b_c)
                x_o = _bias_add(x_o, b_o)

            h_tm1_i = h_tm1 * rec_dp_mask[0] if rec_dp_mask[0] is not None else h_tm1
            h_tm1_f = h_tm1 * rec_dp_mask[1] if rec_dp_mask[1] is not None else h_tm1
            h_tm1_c = h_tm1 * rec_dp_mask[2] if rec_dp_mask[2] is not None else h_tm1
            h_tm1_o = h_tm1 * rec_dp_mask[3] if rec_dp_mask[3] is not None else h_tm1

            c, o = self._compute_carry_and_output(
                (x_i, x_f, x_c, x_o),
                (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o),
                c_tm1,
                quantized_recurrent,
            )
        else:
            inputs_i = inputs * dp_mask[0] if dp_mask[0] is not None else inputs
            h_i = h_tm1 * rec_dp_mask[0] if rec_dp_mask[0] is not None else h_tm1
            z = _dot(inputs_i, quantized_kernel) + _dot(h_i, quantized_recurrent)
            if self.use_bias:
                z = _bias_add(z, quantized_bias)
            z = ops.split(z, 4, axis=1)
            c, o = self._compute_carry_and_output_fused(z, c_tm1)

        h = o * self.activation(c)
        return h, [h, c]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "implementation": self.implementation,
                "kernel_quantizer": _serialize_quantizer(self.kernel_quantizer_internal),
                "recurrent_quantizer": _serialize_quantizer(
                    self.recurrent_quantizer_internal
                ),
                "bias_quantizer": _serialize_quantizer(self.bias_quantizer_internal),
                "state_quantizer": _serialize_quantizer(self.state_quantizer_internal),
            }
        )
        return config


@register_keras_serializable(package="qkeras")
class QLSTM(layers.RNN):
    """Quantized LSTM layer."""

    def __init__(self, units, activity_regularizer=None, **kwargs):
        rnn_kwargs = _pop_rnn_kwargs(kwargs)
        cell = QLSTMCell(
            units,
            **kwargs,
        )
        super().__init__(cell, **rnn_kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [layers.InputSpec(ndim=3)]

    def call(self, sequences, initial_state=None, mask=None, training=False):
        return super().call(
            sequences,
            initial_state=initial_state,
            mask=mask,
            training=training,
        )

    def compute_output_shape(self, inputs_shape):
        return super().compute_output_shape(inputs_shape)

    def get_quantizers(self):
        return self.cell.quantizers

    def get_prunable_weights(self):
        return [self.cell.kernel, self.cell.recurrent_kernel]

    def get_quantization_config(self):
        return {
            "kernel_quantizer": str(self.cell.kernel_quantizer_internal),
            "recurrent_quantizer": str(self.cell.recurrent_quantizer_internal),
            "bias_quantizer": str(self.cell.bias_quantizer_internal),
            "state_quantizer": str(self.cell.state_quantizer_internal),
            "activation": str(self.cell.activation),
            "recurrent_activation": str(self.cell.recurrent_activation),
        }

    def get_config(self):
        base_config = super().get_config()
        base_config.pop("cell", None)
        base_config.update(_cell_config(self.cell))
        base_config["activity_regularizer"] = regularizers.serialize(
            self.activity_regularizer
        )
        return base_config

    @classmethod
    def from_config(cls, config):
        if config.get("implementation") == 0:
            config["implementation"] = 1
        return cls(**config)


@register_keras_serializable(package="qkeras")
class QGRUCell(layers.GRUCell):
    """Quantized GRU cell."""

    def __init__(
        self,
        units,
        activation="quantized_tanh",
        recurrent_activation="hard_sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        kernel_quantizer=None,
        recurrent_quantizer=None,
        bias_quantizer=None,
        state_quantizer=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        implementation=1,
        reset_after=False,
        seed=None,
        **kwargs,
    ):
        implementation = 1 if implementation == 0 else implementation
        self.kernel_quantizer = kernel_quantizer
        self.recurrent_quantizer = recurrent_quantizer
        self.bias_quantizer = bias_quantizer
        self.state_quantizer = state_quantizer

        self.kernel_quantizer_internal = get_quantizer(kernel_quantizer)
        self.recurrent_quantizer_internal = get_quantizer(recurrent_quantizer)
        self.bias_quantizer_internal = get_quantizer(bias_quantizer)
        self.state_quantizer_internal = get_quantizer(state_quantizer)
        self.quantizers = [
            self.kernel_quantizer_internal,
            self.recurrent_quantizer_internal,
            self.bias_quantizer_internal,
            self.state_quantizer_internal,
        ]

        for quantizer in [
            self.kernel_quantizer_internal,
            self.recurrent_quantizer_internal,
        ]:
            if hasattr(quantizer, "_set_trainable_parameter"):
                quantizer._set_trainable_parameter()

        kernel_constraint, kernel_initializer = get_auto_range_constraint_initializer(
            self.kernel_quantizer_internal, kernel_constraint, kernel_initializer
        )
        recurrent_constraint, recurrent_initializer = get_auto_range_constraint_initializer(
            self.recurrent_quantizer_internal,
            recurrent_constraint,
            recurrent_initializer,
        )
        if use_bias:
            bias_constraint, bias_initializer = get_auto_range_constraint_initializer(
                self.bias_quantizer_internal, bias_constraint, bias_initializer
            )

        super().__init__(
            units=units,
            activation=get_quantizer(activation) if activation is not None else None,
            recurrent_activation=(
                get_quantizer(recurrent_activation)
                if recurrent_activation is not None
                else None
            ),
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            reset_after=reset_after,
            seed=seed,
            **kwargs,
        )

        self.implementation = implementation

    def call(self, inputs, states, training=False):
        h_tm1 = states[0] if is_nested(states) else states
        if self.state_quantizer:
            h_tm1 = self.state_quantizer_internal(h_tm1)

        dp_mask = _get_dropout_mask(self, inputs, 3) if training else [None] * 3
        rec_dp_mask = (
            _get_recurrent_dropout_mask(self, h_tm1, 3) if training else [None] * 3
        )

        quantized_kernel = (
            self.kernel_quantizer_internal(self.kernel)
            if self.kernel_quantizer
            else self.kernel
        )
        quantized_recurrent = (
            self.recurrent_quantizer_internal(self.recurrent_kernel)
            if self.recurrent_quantizer
            else self.recurrent_kernel
        )

        if self.use_bias:
            quantized_bias = (
                self.bias_quantizer_internal(self.bias) if self.bias_quantizer else self.bias
            )
            if self.reset_after:
                input_bias, recurrent_bias = ops.unstack(quantized_bias)
            else:
                input_bias, recurrent_bias = quantized_bias, None
        else:
            input_bias = recurrent_bias = None

        if self.implementation == 1:
            inputs_z = inputs * dp_mask[0] if dp_mask[0] is not None else inputs
            inputs_r = inputs * dp_mask[1] if dp_mask[1] is not None else inputs
            inputs_h = inputs * dp_mask[2] if dp_mask[2] is not None else inputs

            x_z = _dot(inputs_z, quantized_kernel[:, : self.units])
            x_r = _dot(inputs_r, quantized_kernel[:, self.units : self.units * 2])
            x_h = _dot(inputs_h, quantized_kernel[:, self.units * 2 :])

            if self.use_bias:
                x_z = _bias_add(x_z, input_bias[: self.units])
                x_r = _bias_add(x_r, input_bias[self.units : self.units * 2])
                x_h = _bias_add(x_h, input_bias[self.units * 2 :])

            h_tm1_z = h_tm1 * rec_dp_mask[0] if rec_dp_mask[0] is not None else h_tm1
            h_tm1_r = h_tm1 * rec_dp_mask[1] if rec_dp_mask[1] is not None else h_tm1
            h_tm1_h = h_tm1 * rec_dp_mask[2] if rec_dp_mask[2] is not None else h_tm1

            recurrent_z = _dot(h_tm1_z, quantized_recurrent[:, : self.units])
            recurrent_r = _dot(
                h_tm1_r, quantized_recurrent[:, self.units : self.units * 2]
            )

            if self.reset_after and self.use_bias:
                recurrent_z = _bias_add(recurrent_z, recurrent_bias[: self.units])
                recurrent_r = _bias_add(
                    recurrent_r, recurrent_bias[self.units : self.units * 2]
                )

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            if self.reset_after:
                recurrent_h = _dot(h_tm1_h, quantized_recurrent[:, self.units * 2 :])
                if self.use_bias:
                    recurrent_h = _bias_add(recurrent_h, recurrent_bias[self.units * 2 :])
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = _dot(r * h_tm1_h, quantized_recurrent[:, self.units * 2 :])

            hh = self.activation(x_h + recurrent_h)
        else:
            inputs_i = inputs * dp_mask[0] if dp_mask[0] is not None else inputs
            h_i = h_tm1 * rec_dp_mask[0] if rec_dp_mask[0] is not None else h_tm1

            matrix_x = _dot(inputs_i, quantized_kernel)
            if self.use_bias:
                matrix_x = _bias_add(matrix_x, input_bias)
            x_z, x_r, x_h = ops.split(matrix_x, 3, axis=-1)

            if self.reset_after:
                matrix_inner = _dot(h_i, quantized_recurrent)
                if self.use_bias:
                    matrix_inner = _bias_add(matrix_inner, recurrent_bias)
            else:
                matrix_inner = _dot(h_i, quantized_recurrent[:, : 2 * self.units])

            recurrent_z, recurrent_r, recurrent_h = ops.split(matrix_inner, 3, axis=-1)

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            if self.reset_after:
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = _dot(r * h_i, quantized_recurrent[:, 2 * self.units :])

            hh = self.activation(x_h + recurrent_h)

        h = z * h_tm1 + (1.0 - z) * hh
        return h, [h]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "implementation": self.implementation,
                "kernel_quantizer": _serialize_quantizer(self.kernel_quantizer_internal),
                "recurrent_quantizer": _serialize_quantizer(
                    self.recurrent_quantizer_internal
                ),
                "bias_quantizer": _serialize_quantizer(self.bias_quantizer_internal),
                "state_quantizer": _serialize_quantizer(self.state_quantizer_internal),
            }
        )
        return config


@register_keras_serializable(package="qkeras")
class QGRU(layers.RNN):
    """Quantized GRU layer."""

    def __init__(self, units, activity_regularizer=None, **kwargs):
        rnn_kwargs = _pop_rnn_kwargs(kwargs)
        cell = QGRUCell(
            units,
            **kwargs,
        )
        super().__init__(cell, **rnn_kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [layers.InputSpec(ndim=3)]

    def call(self, sequences, initial_state=None, mask=None, training=False):
        return super().call(
            sequences,
            initial_state=initial_state,
            mask=mask,
            training=training,
        )

    def compute_output_shape(self, inputs_shape):
        return super().compute_output_shape(inputs_shape)

    def get_quantizers(self):
        return self.cell.quantizers

    def get_prunable_weights(self):
        return [self.cell.kernel, self.cell.recurrent_kernel]

    def get_quantization_config(self):
        return {
            "kernel_quantizer": str(self.cell.kernel_quantizer_internal),
            "recurrent_quantizer": str(self.cell.recurrent_quantizer_internal),
            "bias_quantizer": str(self.cell.bias_quantizer_internal),
            "state_quantizer": str(self.cell.state_quantizer_internal),
            "activation": str(self.cell.activation),
            "recurrent_activation": str(self.cell.recurrent_activation),
        }

    def get_config(self):
        base_config = super().get_config()
        base_config.pop("cell", None)
        base_config.update(_cell_config(self.cell))
        base_config["activity_regularizer"] = regularizers.serialize(
            self.activity_regularizer
        )
        return base_config

    @classmethod
    def from_config(cls, config):
        if config.get("implementation") == 0:
            config["implementation"] = 1
        return cls(**config)


@register_keras_serializable(package="qkeras")
class QBidirectional(layers.Bidirectional):
    """Quantized bidirectional wrapper."""

    def get_quantizers(self):
        return self.forward_layer.get_quantizers() + self.backward_layer.get_quantizers()

    @property
    def activation(self):
        return self.forward_layer.activation

    def get_quantization_config(self):
        return {
            "layer": self.forward_layer.get_quantization_config(),
            "backward_layer": self.backward_layer.get_quantization_config(),
        }


def _pop_rnn_kwargs(kwargs):
    rnn_keys = [
        "return_sequences",
        "return_state",
        "go_backwards",
        "stateful",
        "unroll",
        "zero_output_for_mask",
    ]
    rnn_kwargs = {key: kwargs.pop(key) for key in rnn_keys if key in kwargs}
    kwargs.pop("enable_caching_device", None)
    return rnn_kwargs


def _cell_config(cell):
    config = cell.get_config()
    config.pop("name", None)
    config.pop("trainable", None)
    config.pop("dtype", None)
    return config


# Backward-compatible layer properties. Kept outside class bodies to reduce
# duplication and keep qkeras/hls4ml-style accessors working.
def _delegate_property(name):
    return property(lambda self: getattr(self.cell, name))


for _cls in [QSimpleRNN, QLSTM, QGRU]:
    for _name in [
        "units",
        "activation",
        "use_bias",
        "kernel_initializer",
        "recurrent_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "recurrent_regularizer",
        "bias_regularizer",
        "kernel_constraint",
        "recurrent_constraint",
        "bias_constraint",
        "kernel_quantizer_internal",
        "recurrent_quantizer_internal",
        "bias_quantizer_internal",
        "state_quantizer_internal",
        "kernel_quantizer",
        "recurrent_quantizer",
        "bias_quantizer",
        "state_quantizer",
        "dropout",
        "recurrent_dropout",
    ]:
        setattr(_cls, _name, _delegate_property(_name))

for _cls in [QLSTM, QGRU]:
    setattr(_cls, "recurrent_activation", _delegate_property("recurrent_activation"))
    setattr(_cls, "implementation", _delegate_property("implementation"))

setattr(QLSTM, "unit_forget_bias", _delegate_property("unit_forget_bias"))
setattr(QGRU, "reset_after", _delegate_property("reset_after"))
