# Copyright 2019 Google LLC
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
"""Test layers from qconvolutional.py."""

import os
import tempfile

import keras
import keras.ops.numpy as knp
import numpy as np
import pytest
from keras.backend import clear_session
from keras.layers import Activation, Flatten, Input
from keras.models import Model
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from qkerasV3 import (
    QActivation,
    QConv1D,
    QConv2D,
    QConv2DTranspose,
    QDense,
    QSeparableConv1D,
    QSeparableConv2D,
    binary,
    extract_model_operations,
    print_qstats,
    quantized_bits,
    quantized_relu,
    ternary,
)
from qkerasV3.utils import (
    load_qmodel,
    model_save_quantized_weights,
    quantized_model_from_json,
)

# set random seed
keras.utils.set_random_seed(812)


def test_qnetwork():

    x = x_in = Input((28, 28, 1), name="input")

    x = QSeparableConv2D(
        32,
        (2, 2),
        strides=(2, 2),
        depthwise_quantizer=binary(alpha=1.0),
        pointwise_quantizer=quantized_bits(4, 0, 1, alpha=1.0),
        activation=quantized_bits(6, 2, 1, alpha=1.0),
        bias_quantizer=quantized_bits(4, 0, 1),
        name="conv2d_0_m",
    )(x)

    x = QActivation("quantized_relu(6,2,1)", name="act0_m")(x)

    x = QConv2D(
        64,
        (3, 3),
        strides=(2, 2),
        kernel_quantizer=ternary(alpha=1.0),
        bias_quantizer=quantized_bits(4, 0, 1),
        activation=quantized_relu(6, 3, 1),
        name="conv2d_1_m",
    )(x)

    x = QConv2D(
        64,
        (2, 2),
        strides=(2, 2),
        kernel_quantizer=quantized_bits(6, 2, 1, alpha=1.0),
        bias_quantizer=quantized_bits(4, 0, 1),
        name="conv2d_2_m",
    )(x)

    x = QActivation("quantized_relu(6,4,1)", name="act2_m")(x)

    x = Flatten(name="flatten")(x)

    x = QDense(
        10,
        kernel_quantizer=quantized_bits(6, 2, 1, alpha=1.0),
        bias_quantizer=quantized_bits(4, 0, 1),
        name="dense",
    )(x)

    x = Activation("softmax", name="softmax")(x)

    model = Model(inputs=[x_in], outputs=[x])

    # Reload model from JSON to check serialization/deserialization
    json_string = model.to_json()
    clear_session()
    model = quantized_model_from_json(json_string)

    np.random.seed(42)
    for layer in model.layers:
        all_weights = []
        for i, weights in enumerate(layer.get_weights()):
            input_shape = layer.input.shape
            if hasattr(input_shape, "as_list"):
                input_shape = input_shape.as_list()
            input_size = knp.prod(input_shape[1:]) if input_shape is not None else 1

            if len(layer.get_weights()) == 3 and i > 0:
                input_size = input_size // knp.prod(layer.kernel_size)

            shape = weights.shape
            assert input_size > 0, f"input size for {layer.name} {i}"
            all_weights.append(
                10.0 * np.random.normal(0.0, knp.sqrt(2 / input_size), shape)
            )

        if all_weights:
            layer.set_weights(all_weights)

    model_save_quantized_weights(model)

    all_weights = []
    for layer in model.layers:
        for weights in layer.get_weights():
            all_weights.append(knp.sum(weights))
    all_weights = knp.array(all_weights)

    expected_weights = knp.array(
        [2.0, -6.75, -0.625, -2.0, -0.25, -56.0, 1.125, -1.625, -1.125]
    )
    assert knp.size(all_weights) == knp.size(expected_weights)
    assert knp.all(all_weights == expected_weights)

    expected_output = knp.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype="float16",
    )

    inputs = 2 * np.random.rand(10, 28, 28, 1)
    actual_output = model.predict(inputs, verbose=0)
    assert_allclose(actual_output, expected_output, rtol=1e-2, atol=1e-3)


def test_sequential_qnetwork():
    inputs = keras.Input(shape=(28, 28, 1), name="input")
    x = QConv2D(
        32,
        (2, 2),
        strides=(2, 2),
        kernel_quantizer=quantized_bits(4, 0, 1),
        bias_quantizer=quantized_bits(4, 0, 1),
        name="conv2d_0_m",
    )(inputs)
    x = QActivation(quantized_relu(4, 0), name="act0_m")(x)
    x = QConv2D(
        64,
        (3, 3),
        strides=(2, 2),
        kernel_quantizer=quantized_bits(4, 0, 1),
        bias_quantizer=quantized_bits(4, 0, 1),
        name="conv2d_1_m",
    )(x)
    x = QActivation(quantized_relu(4, 0), name="act1_m")(x)
    x = QConv2D(
        64,
        (2, 2),
        strides=(2, 2),
        kernel_quantizer=quantized_bits(4, 0, 1),
        bias_quantizer=quantized_bits(4, 0, 1),
        name="conv2d_2_m",
    )(x)
    x = QActivation(quantized_relu(4, 0), name="act2_m")(x)
    x = Flatten()(x)
    x = QDense(
        10,
        kernel_quantizer=quantized_bits(4, 0, 1),
        bias_quantizer=quantized_bits(4, 0, 1),
        name="dense",
    )(x)
    outputs = Activation("softmax", name="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Check that all model operation were found correctly
    model_ops = extract_model_operations(model)
    for layer in model_ops.keys():
        assert model_ops[layer]["type"][0] != "null"
    return model


@pytest.mark.parametrize("layer_cls", ["QConv1D", "QSeparableConv1D"])
def test_qconv1d(layer_cls):
    np.random.seed(33)

    if layer_cls == "QConv1D":
        x = Input(
            (
                4,
                4,
            )
        )
        y = QConv1D(
            2,
            1,
            kernel_quantizer=quantized_bits(6, 2, 1, alpha=1.0),
            bias_quantizer=quantized_bits(4, 0, 1),
            name="qconv1d",
        )(x)
        model = Model(inputs=x, outputs=y)
    else:
        x = Input(
            (
                4,
                4,
            )
        )
        y = QSeparableConv1D(
            2,
            2,
            depthwise_quantizer=quantized_bits(6, 2, 1, alpha=1.0),
            pointwise_quantizer=quantized_bits(4, 0, 1, alpha=1.0),
            bias_quantizer=quantized_bits(4, 0, 1),
            name="qconv1d",
        )(x)
        model = Model(inputs=x, outputs=y)

    # Extract model operations
    model_ops = extract_model_operations(model)

    # Check the input layer model operation was found correctly
    assert model_ops["qconv1d"]["type"][0] != "null"

    # Assertion about the number of operations for this (Separable)Conv1D layer
    if layer_cls == "QConv1D":
        assert model_ops["qconv1d"]["number_of_operations"] == 32
    else:
        assert model_ops["qconv1d"]["number_of_operations"] == 30

    # Print qstats to make sure it works with Conv1D layer
    print_qstats(model)

    # reload the model to ensure saving/loading works
    # json_string = model.to_json()
    # clear_session()
    # model = quantized_model_from_json(json_string)

    for layer in model.layers:
        all_weights = []
        for i, weights in enumerate(layer.get_weights()):
            input_size = knp.prod(layer.input.shape[1:])
            if input_size is None:
                input_size = 10 * 10
            shape = weights.shape
            assert input_size > 0, f"input size for {layer.name} {i}"
            all_weights.append(
                10.0 * np.random.normal(0.0, knp.sqrt(2 / input_size), shape)
            )
        if all_weights:
            layer.set_weights(all_weights)
    # Save the model as an keras file using Keras's model.save()
    fd, fname = tempfile.mkstemp(".keras")
    model.save(fname)
    del model  # Delete the existing model

    # Return a compiled model identical to the previous one
    model = load_qmodel(fname)

    # Clean the created keras file after loading the model
    os.close(fd)
    os.remove(fname)

    # apply quantizer to weights
    model_save_quantized_weights(model)

    inputs = np.random.rand(2, 4, 4)
    p = keras.ops.cast(model.predict(inputs), dtype="float16")
    if layer_cls == "QConv1D":
        y = knp.array(
            [
                [[-2.441, 3.816], [-3.807, -1.426], [-2.684, -1.317], [-1.659, 0.9834]],
                [[-4.99, 1.139], [-2.559, -1.216], [-2.285, 1.905], [-2.652, -0.467]],
            ], dtype="float16")
    else:
        y = knp.array(
            [
                [[-2.275, -3.178], [-0.4358, -3.262], [1.987, 0.3987]],
                [[-0.01251, -0.376], [0.3928, -1.328], [-1.243, -2.43]],
            ], dtype="float16")
    assert_allclose(p, y, rtol=1e-4)


def test_qconv2dtranspose():
    x = Input(
        (
            4,
            4,
            1,
        )
    )
    y = QConv2DTranspose(
        1,
        kernel_size=(3, 3),
        kernel_quantizer=binary(),
        bias_quantizer=binary(),
        name="conv2d_tran",
    )(x)
    model = Model(inputs=x, outputs=y)
    data = knp.ones(shape=(1, 4, 4, 1))
    kernel = knp.ones(shape=(3, 3, 1, 1))
    bias = knp.ones(shape=(1,))
    model.get_layer("conv2d_tran").set_weights([kernel, bias])
    actual_output = model.predict(data)
    expected_output = (
        np.array(
            [
                [2.0, 3.0, 4.0, 4.0, 3.0, 2.0],
                [3.0, 5.0, 7.0, 7.0, 5.0, 3.0],
                [4.0, 7.0, 10.0, 10.0, 7.0, 4.0],
                [4.0, 7.0, 10.0, 10.0, 7.0, 4.0],
                [3.0, 5.0, 7.0, 7.0, 5.0, 3.0],
                [2.0, 3.0, 4.0, 4.0, 3.0, 2.0],
            ], dtype="float16"
        )
        .reshape((1, 6, 6, 1))
    )
    assert_allclose(actual_output, expected_output, rtol=1e-4)


def test_masked_qconv2d_creates_correct_parameters():
    mask = mask = knp.ones((5, 5), dtype=float)
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(10, 10, 1)))
    model.add(QConv2D(mask=mask, filters=1, kernel_size=(5, 5), use_bias=False))

    # There should be no non-trainable params.
    assert_equal(len(model.non_trainable_weights), 0)

    # Validate number of trainable params. This should be equal to one (5,5)
    # kernel.
    assert_equal(len(model.trainable_weights), 1)
    num_trainable_params = knp.prod(model.trainable_weights[0].shape)
    assert_equal(num_trainable_params.numpy(), 25)


def test_qconv2d_masks_weights():
    # Create an arbitrary mask.
    mask = knp.array(
        [
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0],
        ], dtype=float
    )
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(5, 5, 1)))
    model.add(QConv2D(mask=mask, filters=1, kernel_size=(5, 5), use_bias=False))

    # Set the weights to be all ones.
    model.layers[0].set_weights([knp.ones((5, 5, 1, 1), dtype=float)])

    # Run inference on a all ones input.
    output = model.predict(knp.ones((1, 5, 5, 1)))
    # Output should just be summation of number of ones in the mask.
    assert_array_equal(output, knp.array([[[[11.0]]]], dtype=float))


def test_masked_qconv2d_load_restore_works():
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(10, 10, 1)))
    model.add(
        QConv2D(
            mask=knp.ones((5, 5), dtype=float),
            filters=1,
            kernel_size=(5, 5),
            use_bias=False,
        )
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model.keras")
        # Can save the model.
        model.save(model_path)

        # Can load the model.
        custom_objects = {
            "QConv2D": QConv2D,
        }
        loaded_model = keras.models.load_model(
            model_path, custom_objects=custom_objects
        )

        assert_array_equal(
            model.layers[0].weights[0], loaded_model.layers[0].weights[0]
        )


def test_qconv2d_groups_works():
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(10, 10, 10)))
    model.add(
        QConv2D(
            filters=6,
            kernel_size=(1, 1),
            use_bias=True,
            groups=2,
        )
    )
    # Validate number of trainable params.
    assert_equal(len(model.trainable_weights), 2)
    num_trainable_params = knp.prod(model.trainable_weights[0].shape) + knp.prod(
        model.trainable_weights[1].shape
    )
    expected_trainable_params = 36  # (5*3)*2 + 6
    assert_equal(num_trainable_params.numpy(), expected_trainable_params)


if __name__ == "__main__":
    pytest.main([__file__])
