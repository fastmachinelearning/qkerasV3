"""Tests for qkerasV3 recurrent layers.

These tests are intentionally backend-light and should run with Keras 3 on the
TensorFlow backend. They cover construction, forward passes, serialization, and
regressions that commonly break during Keras 2 -> Keras 3 migrations.
"""

import numpy as np
import pytest

import keras

from qkeras.qrecurrent import QBidirectional, QGRU, QGRUCell, QLSTM, QSimpleRNN


pytestmark = pytest.mark.filterwarnings(
    "ignore:.*The structure of `inputs` doesn't match.*:UserWarning"
)


RNN_LAYER_CASES = [
    (QSimpleRNN, {"units": 4}),
    (QLSTM, {"units": 4}),
    (QGRU, {"units": 4}),
]


@pytest.mark.parametrize("layer_cls,kwargs", RNN_LAYER_CASES)
def test_quantized_rnn_layer_forward_pass(layer_cls, kwargs):
    layer = layer_cls(
        **kwargs,
        kernel_quantizer="quantized_bits(4,0,1)",
        recurrent_quantizer="quantized_bits(4,0,1)",
        bias_quantizer="quantized_bits(4,0,1)",
        state_quantizer="quantized_bits(4,0,1)",
        return_sequences=True,
    )

    x = keras.random.normal((2, 5, 3), seed=1337)
    y = layer(x, training=False)

    assert tuple(y.shape) == (2, 5, 4)
    assert len(layer.get_quantizers()) == 4
    assert len(layer.get_prunable_weights()) == 2


@pytest.mark.parametrize("layer_cls,kwargs", RNN_LAYER_CASES)
def test_quantized_rnn_layer_config_round_trip(layer_cls, kwargs):
    layer = layer_cls(
        **kwargs,
        activation="quantized_tanh",
        kernel_quantizer="quantized_bits(3,0,1)",
        recurrent_quantizer="quantized_bits(3,0,1)",
        bias_quantizer="quantized_bits(3,0,1)",
        state_quantizer="quantized_bits(3,0,1)",
        return_sequences=True,
        return_state=True,
        go_backwards=True,
        name=f"test_{layer_cls.__name__.lower()}",
    )

    config = layer.get_config()
    restored = layer_cls.from_config(config)

    assert restored.units == layer.units
    assert restored.return_sequences is True
    assert restored.return_state is True
    assert restored.go_backwards is True
    assert restored.kernel_quantizer_internal is not None
    assert restored.recurrent_quantizer_internal is not None
    assert restored.bias_quantizer_internal is not None
    assert restored.state_quantizer_internal is not None


@pytest.mark.parametrize("layer_cls,kwargs", RNN_LAYER_CASES)
def test_quantized_rnn_model_save_load_round_trip(tmp_path, layer_cls, kwargs):
    inputs = keras.Input(shape=(5, 3))
    outputs = layer_cls(
        **kwargs,
        kernel_quantizer="quantized_bits(4,0,1)",
        recurrent_quantizer="quantized_bits(4,0,1)",
        bias_quantizer="quantized_bits(4,0,1)",
        state_quantizer="quantized_bits(4,0,1)",
    )(inputs)
    model = keras.Model(inputs, outputs)

    x = np.random.default_rng(123).normal(size=(2, 5, 3)).astype("float32")
    before = model.predict(x, verbose=0)

    path = tmp_path / f"{layer_cls.__name__}.keras"
    model.save(path)
    restored = keras.models.load_model(path)
    after = restored.predict(x, verbose=0)

    np.testing.assert_allclose(before, after, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("layer_cls", [QLSTM, QGRU])
def test_implementation_zero_is_upgraded_for_legacy_configs(layer_cls):
    layer = layer_cls(
        4,
        implementation=0,
        kernel_quantizer="quantized_bits(3,0,1)",
        recurrent_quantizer="quantized_bits(3,0,1)",
    )

    assert layer.implementation == 1

    config = layer.get_config()
    config["implementation"] = 0
    restored = layer_cls.from_config(config)

    assert restored.implementation == 1


def test_lstm_respects_unit_forget_bias_false():
    layer = QLSTM(4, unit_forget_bias=False)

    assert layer.unit_forget_bias is False
    assert layer.get_config()["unit_forget_bias"] is False


def test_bidirectional_returns_forward_and_backward_quantizers():
    layer = QBidirectional(
        QGRU(
            4,
            kernel_quantizer="quantized_bits(3,0,1)",
            recurrent_quantizer="quantized_bits(3,0,1)",
            bias_quantizer="quantized_bits(3,0,1)",
            state_quantizer="quantized_bits(3,0,1)",
        )
    )

    x = keras.random.normal((2, 5, 3), seed=2024)
    y = layer(x)

    assert tuple(y.shape) == (2, 8)
    assert len(layer.get_quantizers()) == 8
    assert "layer" in layer.get_quantization_config()
    assert "backward_layer" in layer.get_quantization_config()


def test_gru_cell_uses_recurrent_kernel_when_recurrent_quantizer_is_absent():
    """Regression test for the accidental `else: self.kernel` bug.

    The recurrent kernel has shape `(units, 3 * units)`, while the input kernel
    has shape `(input_dim, 3 * units)`. With `input_dim != units`, using
    `self.kernel` in the recurrent path causes a matmul shape error.
    """

    cell = QGRUCell(
        4,
        kernel_quantizer="quantized_bits(4,0,1)",
        recurrent_quantizer=None,
        bias_quantizer=None,
        state_quantizer=None,
        reset_after=False,
    )

    inputs = keras.random.normal((2, 3), seed=7)
    previous_state = keras.random.normal((2, 4), seed=8)

    output, states = cell(inputs, [previous_state], training=False)

    assert tuple(output.shape) == (2, 4)
    assert len(states) == 1
    assert tuple(states[0].shape) == (2, 4)


def test_dropout_training_call_does_not_crash():
    layer = QSimpleRNN(
        4,
        dropout=0.25,
        recurrent_dropout=0.25,
        kernel_quantizer="quantized_bits(4,0,1)",
        recurrent_quantizer="quantized_bits(4,0,1)",
        return_sequences=True,
    )

    x = keras.random.normal((2, 5, 3), seed=9)
    y = layer(x, training=True)

    assert tuple(y.shape) == (2, 5, 4)
