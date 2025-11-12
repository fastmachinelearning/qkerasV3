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
"""Tests layers from folded_layers.py."""

import os
import tempfile

import jax
import keras
import keras.ops.numpy as knp
import numpy as np
import pytest
from keras import layers
from keras.backend import clear_session
from keras.models import Model
from keras.utils import to_categorical
from numpy.testing import assert_allclose, assert_equal, assert_raises

from qkeras import (
    QActivation,
    QConv2D,
    QConv2DBatchnorm,
    QDense,
    QDepthwiseConv2D,
    QDepthwiseConv2DBatchnorm,
    bn_folding_utils,
    QDenseBatchnorm,
)
from qkeras.utils import (
    _add_supported_quantized_objects,
    load_qmodel,
    quantized_model_from_json,
)

# set random seed
keras.utils.set_random_seed(812)


def get_qconv2d_model(input_shape, kernel_size, kernel_quantizer=None):
    num_class = 2

    x = x_in = layers.Input(input_shape, name="input")

    x = QConv2D(
        filters=2,
        kernel_size=kernel_size,
        strides=(4, 4),
        kernel_initializer="ones",
        bias_initializer="zeros",
        use_bias=False,
        kernel_quantizer=kernel_quantizer,
        bias_quantizer=None,
        name="conv2d",
    )(x)

    x = layers.BatchNormalization(
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
        name="bn",
    )(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(
        num_class, use_bias=False, kernel_initializer="ones", name="dense"
    )(x)
    x = layers.Activation("softmax", name="softmax")(x)
    model = Model(inputs=[x_in], outputs=[x])
    return model


def get_qconv2d_batchnorm_model(
    input_shape, kernel_size, folding_mode, kernel_quantizer=None
):
    num_class = 2

    x = x_in = layers.Input(input_shape, name="input")
    x = QConv2DBatchnorm(
        filters=2,
        kernel_size=kernel_size,
        strides=(4, 4),
        kernel_initializer="ones",
        bias_initializer="zeros",
        use_bias=False,
        kernel_quantizer=kernel_quantizer,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        folding_mode=folding_mode,
        name="foldconv2d",
    )(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(
        num_class, use_bias=False, kernel_initializer="ones", name="dense"
    )(x)
    x = layers.Activation("softmax", name="softmax")(x)
    model = Model(inputs=[x_in], outputs=[x])
    return model


def get_conv2d_models_with_one_layer(kernel_quantizer, folding_mode, ema_freeze_delay):
    x_shape = (2, 2, 1)
    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)

    # define a model with seperate conv2d and bn layers
    x = x_in = layers.Input(x_shape, name="input")
    x = QConv2D(
        filters=2,
        kernel_size=(2, 2),
        strides=(4, 4),
        kernel_initializer="ones",
        bias_initializer="zeros",
        use_bias=False,
        kernel_quantizer=kernel_quantizer,
        bias_quantizer=None,
        name="conv2d",
    )(x)
    x = layers.BatchNormalization(
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
        name="bn",
    )(x)
    unfold_model = Model(inputs=[x_in], outputs=[x])
    unfold_model.compile(loss=loss_fn, optimizer=optimizer, metrics=["acc"])

    x = x_in = layers.Input(x_shape, name="input")
    x = QConv2DBatchnorm(
        filters=2,
        kernel_size=(2, 2),
        strides=(4, 4),
        kernel_initializer="ones",
        bias_initializer="zeros",
        use_bias=False,
        kernel_quantizer=kernel_quantizer,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        folding_mode=folding_mode,
        ema_freeze_delay=ema_freeze_delay,
        name="foldconv2d",
    )(x)
    fold_model = Model(inputs=[x_in], outputs=[x])
    fold_model.compile(loss=loss_fn, optimizer=optimizer, metrics=["acc"])

    return (unfold_model, fold_model)


def get_dense_models_with_one_layer(kernel_quantizer, folding_mode, ema_freeze_delay):

    x_shape = (4,)
    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)

    # define a model with seperate conv2d and bn layers
    x = x_in = layers.Input(x_shape, name="input")
    x = QDense(
        2,
        kernel_initializer="ones",
        bias_initializer="zeros", use_bias=False,
        kernel_quantizer=kernel_quantizer, bias_quantizer=None,
        name="conv2d")(x)
    x = layers.BatchNormalization(
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
        name="bn")(x)
    unfold_model = Model(inputs=[x_in], outputs=[x])
    unfold_model.compile(loss=loss_fn, optimizer=optimizer, metrics=["acc"])

    x = x_in = layers.Input(x_shape, name="input")
    x = QDenseBatchnorm(
        2,
        kernel_initializer="ones",
        bias_initializer="zeros",
        use_bias=False,
        kernel_quantizer=kernel_quantizer,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        folding_mode=folding_mode,
        ema_freeze_delay=ema_freeze_delay,
        name="foldconv2d"
    )(x)
    fold_model = Model(inputs=[x_in], outputs=[x])
    fold_model.compile(loss=loss_fn, optimizer=optimizer, metrics=["acc"])

    return (unfold_model, fold_model)


def get_debug_model(model):
    layer_output_list = []
    for layer in model.layers:
        if layer.__class__.__name__ not in ["Flatten", "InputLayer"]:
            layer_output_list.append(layer.output)

    debug_model = Model(inputs=model.inputs, outputs=layer_output_list)
    return debug_model


def generate_dataset(
    train_size=10, input_shape=(3, 3, 1), num_class=2, output_shape=None
):
    """create a toy dataset"""

    x_train = np.random.rand(*(train_size,) + input_shape)

    if output_shape:
        y_train = np.random.random_sample((train_size,) + output_shape)
    else:
        y_train = np.random.randint(num_class, size=train_size)
        y_train = to_categorical(y_train, num_class)

    return x_train, y_train


def run_training(
    model, x, y, epochs, loss_fn, loss_metric, optimizer, do_preditc=False
):
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=[loss_metric])
    # TODO: fix eager mode
    if os.environ["KERAS_BACKEND"] == "jax":
        with jax.disable_jit():
            model.fit(x, y, epochs=epochs, shuffle=False)
    else:
        model.fit(x, y, epochs=epochs, shuffle=False)
    if do_preditc: return model.predict(x)


def test_unfold_model():
    """Test if unfold_model works properly.

    Convert a folded model to a normal model. The kernel/bias weight in
    the normal model should be the same as the folded kernel/bias in the folded
    model. Test if the function can convert both sequential and non-sequantial
    models properly.
    """

    x_shape = (2, 2, 1)
    kernel_quantizer = "quantized_bits(4, 0, 1)"
    folding_mode = "batch_stats_folding"
    ema_freeze_delay = 10
    kernel = knp.array([[[[1.0, 1.0]], [[1.0, 0.0]]], [[[1.0, 1.0]], [[0.0, 1.0]]]])
    gamma = knp.array([2.0, 1.0])
    beta = knp.array([0.0, 1.0])
    moving_mean = knp.array([1.0, 1.0])
    moving_variance = knp.array([1.0, 2.0])
    iteration = knp.array(-1)

    def _get_sequantial_folded_model(x_shape):
        x = x_in = layers.Input(x_shape, name="input")
        x = QConv2DBatchnorm(
            filters=2,
            kernel_size=(2, 2),
            strides=(2, 2),
            kernel_initializer="ones",
            bias_initializer="zeros",
            use_bias=False,
            kernel_quantizer=kernel_quantizer,
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones",
            folding_mode=folding_mode,
            ema_freeze_delay=ema_freeze_delay,
            name="foldconv2d",
        )(x)
        x = QDepthwiseConv2DBatchnorm(
            kernel_size=(2, 2),
            strides=(1, 1),
            use_bias=False,
            depthwise_quantizer=kernel_quantizer,
            folding_mode=folding_mode,
            ema_freeze_delay=ema_freeze_delay,
            name="folddepthwiseconv2d",
        )(x)
        model = Model(inputs=[x_in], outputs=[x])
        model.layers[1].set_weights(
            [kernel, iteration, gamma, beta, moving_mean, moving_variance]
        )

        return model

    def _get_nonseq_folded_model(x_shape):
        x = x_in = layers.Input(x_shape, name="input")
        x1 = layers.Conv2D(
            filters=1, kernel_size=(1, 1), strides=(1, 1), name="conv2d_1"
        )(x)
        x2 = layers.Conv2D(
            filters=1, kernel_size=(1, 1), strides=(1, 1), name="conv2d_2"
        )(x)
        x = layers.Maximum()([x1, x2])
        x = QConv2DBatchnorm(
            filters=2,
            kernel_size=(2, 2),
            strides=(4, 4),
            kernel_initializer="ones",
            bias_initializer="zeros",
            use_bias=False,
            kernel_quantizer=kernel_quantizer,
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones",
            folding_mode=folding_mode,
            ema_freeze_delay=ema_freeze_delay,
            name="foldconv2d",
        )(x)
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(2, use_bias=False, kernel_initializer="ones", name="dense")(x)
        model = Model(inputs=[x_in], outputs=[x])
        model.layers[4].set_weights(
            [kernel, iteration, gamma, beta, moving_mean, moving_variance]
        )
        return model

    seq_model = _get_sequantial_folded_model((4, 4, 1))
    nonseq_model = _get_nonseq_folded_model(x_shape)

    for model in [nonseq_model, seq_model]:
        # preparing data for testing if model prediction matches

        output_shape = model.output_shape[1:]
        input_shape = model.input_shape[1:]
        x, y = generate_dataset(
            train_size=10,
            input_shape=input_shape,
            output_shape=output_shape,
        )

        # convert model with folded layers to a model with coresspoinding QConv2D
        # or QDepthwiseConv2D layers
        cvt_model = bn_folding_utils.unfold_model(model)

        for layer_type in ["QConv2DBatchnorm", "QDepthwiseConv2DBatchnorm"]:
            weight1 = None
            weight2 = None
            for layer in model.layers:
                if layer.__class__.__name__ == layer_type:
                    weight1 = layer.get_folded_weights()
                    break

            for layer in cvt_model.layers:
                if layer.__class__.__name__ == layer_type[:-9]:
                    weight2 = layer.get_weights()
                    break

            # test if the corresponding layers have identical weights
            if weight1 and weight2:
                assert_equal(weight1[0], weight2[0])
                assert_equal(weight1[1], weight2[1])

        # test if the predictions of the two models are identical
        # TODO: fix eager mode
        if os.environ["KERAS_BACKEND"] == "jax":
            with jax.disable_jit():
                pred1 = model.predict(x)
        else:
            pred1 = model.predict(x)
        pred2 = cvt_model.predict(x)
        assert_equal(pred1, pred2)


def test_loading():
    """Test to load model using different approahches."""

    loss_fn = keras.losses.MeanSquaredError()
    loss_metric = 'MAE'
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)
    x_shape = (2, 2, 1)

    custom_objects = {}
    _add_supported_quantized_objects(custom_objects)

    x, y = generate_dataset(
        train_size=1,
        input_shape=x_shape,
        num_class=2
    )

    model_fold = get_qconv2d_batchnorm_model(
        input_shape=x_shape, kernel_size=(2, 2), folding_mode="ema_stats_folding"
    )
    run_training(
        model_fold, x, y, 10, loss_fn, loss_metric, optimizer, do_preditc=False
    )

    # test load model from json to ensure saving/loading model architecture works
    json_string = model_fold.to_json()
    clear_session()
    model_from_json = quantized_model_from_json(json_string)
    assert json_string == model_from_json.to_json()

    # test reload model from hdf5 files to ensure saving/loading works
    _, fname = tempfile.mkstemp(".keras")
    model_fold.save(fname)
    model_loaded = load_qmodel(fname)
    weight1 = model_fold.layers[1].get_folded_weights()
    weight2 = model_loaded.layers[1].get_folded_weights()
    assert_equal(keras.ops.convert_to_numpy(weight1[0]), keras.ops.convert_to_numpy(weight2[0]))
    assert_equal(keras.ops.convert_to_numpy(weight1[1]), keras.ops.convert_to_numpy(weight2[1]))

    # test convert a folded model to a normal model for zpm
    # the kernel/bias weight in the normal model should be the same as the folded
    # kernel/bias in the folded model
    normal_model = bn_folding_utils.unfold_model(model_fold)
    weight2 = normal_model.layers[1].get_weights()

    assert_equal(keras.ops.convert_to_numpy(weight1[0]), weight2[0])
    assert_equal(keras.ops.convert_to_numpy(weight1[1]), weight2[1])


@pytest.mark.parametrize("model_name", ["conv2d", "dense"])
def test_same_training_and_prediction(model_name):
    """test if fold/unfold layer has the same training and prediction output."""

    epochs = 5
    loss_fn = keras.losses.MeanSquaredError()
    loss_metric = 'MAE'

    if model_name == "conv2d":
        x_shape = (2, 2, 1)
        kernel = np.array([[[[1., 1.]], [[1., 0.]]], [[[1., 1.]], [[0., 1.]]]])
    elif model_name == "dense":
        x_shape = (4,)
        kernel = np.array([[1., 1.], [1., 0.], [1., 1.], [0., 1.]])
    gamma = knp.array([2.0, 1.0])
    beta = knp.array([0.0, 1.0])
    moving_mean = knp.array([1.0, 1.0])
    moving_variance = knp.array([1.0, 2.0])
    iteration = knp.array(-1)

    x, y = generate_dataset(
        train_size=10,
        input_shape=x_shape,
        num_class=2
    )

    if model_name == "conv2d":
        (unfold_model, fold_model_batch) = get_conv2d_models_with_one_layer(
            kernel_quantizer=None, folding_mode="batch_stats_folding",
            ema_freeze_delay=10)
        (_, fold_model_ema) = get_conv2d_models_with_one_layer(
            kernel_quantizer=None, folding_mode="ema_stats_folding",
            ema_freeze_delay=10)
    elif model_name == "dense":
        (unfold_model, fold_model_batch) = get_dense_models_with_one_layer(
            kernel_quantizer=None, folding_mode="batch_stats_folding",
            ema_freeze_delay=10)
        (_, fold_model_ema) = get_dense_models_with_one_layer(
            kernel_quantizer=None, folding_mode="ema_stats_folding",
            ema_freeze_delay=10)

    unfold_model.layers[1].set_weights([kernel])
    unfold_model.layers[2].set_weights([gamma, beta, moving_mean, moving_variance])
    fold_model_batch.layers[1].set_weights(
        [kernel, iteration, gamma, beta, moving_mean, moving_variance]
    )
    fold_model_ema.layers[1].set_weights(
        [kernel, iteration, gamma, beta, moving_mean, moving_variance]
    )

    # check if prediction is the same
    y1 = unfold_model.predict(x)
    # TODO: fix eager mode
    if os.environ["KERAS_BACKEND"] == "jax":
        with jax.disable_jit():
            y2_batch = fold_model_batch.predict(x)
            y2_ema = fold_model_ema.predict(x)
    else:
        y2_batch = fold_model_batch.predict(x)
        y2_ema = fold_model_ema.predict(x)
    assert_allclose(y1, y2_batch, rtol=1e-2)
    assert_allclose(y1, y2_ema, rtol=1e-2)

    # check if training for a number of epochs, and before bn freeeze, models
    # reached the same point
    y1 = run_training(
        model=unfold_model,
        x=x,
        y=y,
        epochs=epochs,
        loss_fn=loss_fn,
        loss_metric=loss_metric,
        optimizer=keras.optimizers.SGD(learning_rate=1e-3),
        do_preditc=True
    )

    y2_batch = run_training(
        model=fold_model_batch,
        x=x,
        y=y,
        epochs=epochs,
        loss_fn=loss_fn,
        loss_metric=loss_metric,
        optimizer=keras.optimizers.SGD(learning_rate=1e-3),
        do_preditc=True
    )

    y2_ema = run_training(
        model=fold_model_ema,
        x=x,
        y=y,
        epochs=epochs,
        loss_fn=loss_fn,
        loss_metric=loss_metric,
        optimizer=keras.optimizers.SGD(learning_rate=1e-3),
        do_preditc=True
    )
    assert_allclose(y1, y2_batch, atol=1e-1)
    assert_allclose(y1, y2_ema, atol=1e-1)

    # check if training for long enough (after bn freezes), unfold model and fold
    # models should be different, but the two folding modes should be the same
    epochs = 5
    iteration = knp.array(8)
    if model_name == "conv2d":
        (unfold_model, fold_model_batch) = get_conv2d_models_with_one_layer(
            kernel_quantizer=None, folding_mode="batch_stats_folding",
            ema_freeze_delay=10)
        (_, fold_model_ema) = get_conv2d_models_with_one_layer(
            kernel_quantizer=None, folding_mode="ema_stats_folding",
            ema_freeze_delay=10)
    elif model_name == "dense":
        (unfold_model, fold_model_batch) = get_dense_models_with_one_layer(
            kernel_quantizer=None, folding_mode="batch_stats_folding",
            ema_freeze_delay=10)
        (_, fold_model_ema) = get_dense_models_with_one_layer(
            kernel_quantizer=None, folding_mode="ema_stats_folding",
            ema_freeze_delay=10)

    unfold_model.layers[1].set_weights([kernel])
    unfold_model.layers[2].set_weights([gamma, beta, moving_mean, moving_variance])
    fold_model_batch.layers[1].set_weights(
        [kernel, iteration, gamma, beta, moving_mean, moving_variance]
    )
    fold_model_ema.layers[1].set_weights(
        [kernel, iteration, gamma, beta, moving_mean, moving_variance]
    )

    y1 = run_training(
        model=unfold_model,
        x=x,
        y=y,
        epochs=epochs,
        loss_fn=loss_fn,
        loss_metric=loss_metric,
        optimizer=keras.optimizers.SGD(learning_rate=1e-3),
        do_preditc=True
    )

    y2_batch = run_training(
        model=fold_model_batch,
        x=x,
        y=y,
        epochs=epochs,
        loss_fn=loss_fn,
        loss_metric=loss_metric,
        optimizer=keras.optimizers.SGD(learning_rate=1e-3),
        do_preditc=True
    )

    y2_ema = run_training(
        model=fold_model_ema,
        x=x,
        y=y,
        epochs=epochs,
        loss_fn=loss_fn,
        loss_metric=loss_metric,
        optimizer=keras.optimizers.SGD(learning_rate=1e-3),
        do_preditc=True
    )
    assert_raises(AssertionError, assert_allclose, y1, y2_batch, rtol=1e-4)
    assert_allclose(y2_batch, y2_ema, rtol=1e-4)

    # test QDepthwiseConv2DBatchnorm layers
    def _get_models(
        x_shape, num_class, depthwise_quantizer, folding_mode, ema_freeze_delay
    ):
        x = x_in = layers.Input(x_shape, name="input")
        x = QDepthwiseConv2DBatchnorm(
            kernel_size=(2, 2),
            strides=(2, 2),
            depth_multiplier=1,
            depthwise_initializer="ones",
            bias_initializer="zeros",
            use_bias=False,
            depthwise_quantizer=depthwise_quantizer,
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones",
            folding_mode=folding_mode,
            ema_freeze_delay=ema_freeze_delay,
            name="fold_depthwiseconv2d",
        )(x)
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(
            num_class, use_bias=False, kernel_initializer="ones", name="dense"
        )(x)
        x = layers.Activation("softmax", name="softmax")(x)
        fold_model = Model(inputs=[x_in], outputs=[x])

        x = x_in = layers.Input(x_shape, name="input")
        x = QDepthwiseConv2D(
            kernel_size=(2, 2),
            strides=(2, 2),
            depth_multiplier=1,
            depthwise_initializer="ones",
            bias_initializer="zeros",
            use_bias=False,
            depthwise_quantizer=depthwise_quantizer,
            name="depthwiseconv2d",
        )(x)
        x = layers.BatchNormalization(
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones",
            name="bn",
        )(x)
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(
            num_class, use_bias=False, kernel_initializer="ones", name="dense"
        )(x)
        x = layers.Activation("softmax", name="softmax")(x)
        model = Model(inputs=[x_in], outputs=[x])

        return (model, fold_model)

    input_shape = (4, 4, 1)
    num_class = 2
    depthwise_quantizer = None
    folding_mode = "ema_stats_folding"
    ema_freeze_delay = 10

    # weights
    depthwise_kernel = knp.array([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]])
    gamma = knp.array([2])
    beta = knp.array([0])
    moving_mean = knp.array([4.0])
    moving_variance = knp.array([2.0])
    iteration = knp.array(2)
    folded_depthwise_kernel_quantized = knp.array(
        [[[[1.4138602]], [[0.0]]], [[[0.0]], [[1.4138602]]]]
    )
    folded_bias_quantized = knp.array([-5.655441])
    dense_weight = knp.array([[1.0, 0], [0, 0], [0, 0], [0, 0]])

    # generate dataset
    x, y = generate_dataset(
        train_size=3,
        input_shape=input_shape,
        num_class=2
    )

    # define models, one with folded layer and one without
    (model, fold_model) = _get_models(
        input_shape,
        num_class=num_class,
        depthwise_quantizer=depthwise_quantizer,
        folding_mode=folding_mode,
        ema_freeze_delay=ema_freeze_delay,
    )

    # set weights
    fold_model.layers[1].set_weights(
        [depthwise_kernel, iteration, gamma, beta, moving_mean, moving_variance]
    )
    fold_model.layers[3].set_weights([dense_weight])

    model.layers[1].set_weights([depthwise_kernel])
    model.layers[2].set_weights([gamma, beta, moving_mean, moving_variance])
    model.layers[4].set_weights([dense_weight])

    # perform training
    epochs = 5
    loss_fn = keras.losses.MeanSquaredError()
    loss_metric = 'MAE'

    pred1 = run_training(
        model=model,
        x=x,
        y=y,
        epochs=epochs,
        loss_fn=loss_fn,
        loss_metric=loss_metric,
        optimizer=keras.optimizers.SGD(learning_rate=1e-3),
        do_preditc=True
    )
    pred2 = run_training(
        model=fold_model,
        x=x,
        y=y,
        epochs=epochs,
        loss_fn=loss_fn,
        loss_metric=loss_metric,
        optimizer=keras.optimizers.SGD(learning_rate=1e-3),
        do_preditc=True
    )

    # before bn freezes, the two models should reach the same point
    assert_allclose(pred1, pred2, rtol=1e-4)

    # after bn freezes, the two models will not reach the same
    iteration = knp.array(12)
    epochs = 5
    ema_freeze_delay = 10
    (model, fold_model) = _get_models(
        input_shape,
        num_class=num_class,
        depthwise_quantizer=depthwise_quantizer,
        folding_mode=folding_mode,
        ema_freeze_delay=ema_freeze_delay,
    )
    fold_model.layers[1].set_weights(
        [depthwise_kernel, iteration, gamma, beta, moving_mean, moving_variance]
    )
    fold_model.layers[3].set_weights([dense_weight])
    model.layers[1].set_weights([depthwise_kernel])
    model.layers[2].set_weights([gamma, beta, moving_mean, moving_variance])
    model.layers[4].set_weights([dense_weight])

    pred1 = run_training(
        model=model,
        x=x,
        y=y,
        epochs=epochs,
        loss_fn=loss_fn,
        loss_metric=loss_metric,
        optimizer=keras.optimizers.SGD(learning_rate=1e-3),
        do_preditc=True
    )
    pred2 = run_training(
        model=fold_model,
        x=x,
        y=y,
        epochs=epochs,
        loss_fn=loss_fn,
        loss_metric=loss_metric,
        optimizer=keras.optimizers.SGD(learning_rate=1e-3),
        do_preditc=True
    )

    assert_raises(AssertionError, assert_allclose, pred1, pred2, rtol=1e-4)


def test_populate_bias_quantizer_from_accumulator():
    """Test populate_bias_quantizer_from_accumulator function.

    Define a qkeras model with a QConv2DBatchnorm layer. Set bias quantizer in the
    layer as None. Call populate_bias_quantizer_from_accumulator function
    to automatically generate bias quantizer type from the MAC accumulator type.
    Set the bias quantizer accordingly in the model.

    Call populate_bias_quantizer_from_accumulator again in this model. This time
    since bias quantizer is already set, populate_bias_quantizer_from_accumulator
    function should not change the bias quantizer.
    """

    x_shape = (2, 2, 1)

    # get a qkeras model with QConv2DBatchnorm layer. Set bias quantizer in the
    # layer as None.
    x = x_in = layers.Input(x_shape, name="input")
    x1 = QConv2D(
        filters=1,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        kernel_quantizer="quantized_bits(4, 0, 1)",
        name="conv2d_1",
    )(x)
    x2 = QConv2D(
        filters=1,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        kernel_quantizer="quantized_bits(4, 0, 1)",
        name="conv2d_2",
    )(x)
    x = layers.Maximum()([x1, x2])
    x = QActivation("quantized_relu(4, 1)")(x)
    x = QConv2DBatchnorm(
        filters=2,
        kernel_size=(2, 2),
        strides=(4, 4),
        kernel_initializer="ones",
        bias_initializer="zeros",
        use_bias=False,
        kernel_quantizer="quantized_bits(4, 0, 1)",
        bias_quantizer=None,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        folding_mode="batch_stats_folding",
        ema_freeze_delay=10,
        name="foldconv2d",
    )(x)
    x1 = x
    x2 = layers.Flatten(name="flatten")(x)
    x2 = QDense(
        2,
        use_bias=False,
        kernel_initializer="ones",
        kernel_quantizer="quantized_bits(6, 2, 1)",
        name="dense",
    )(x2)
    model = Model(inputs=[x_in], outputs=[x1, x2])
    assert_equal(model.layers[5].get_quantizers()[1], None)

    # Call populate_bias_quantizer_from_accumulator function
    # to automatically generate bias quantizer from the MAC accumulator type.
    _ = bn_folding_utils.populate_bias_quantizer_from_accumulator(
        model, ["quantized_bits(8, 0, 1)"]
    )
    q = model.layers[5].get_quantizers()[1]
    assert_equal(q.__str__(), "quantized_bits(10,3,1)")

    # Call populate_bias_quantizer_from_accumulator function again
    # bias quantizer should not change
    _ = bn_folding_utils.populate_bias_quantizer_from_accumulator(
        model, ["quantized_bits(8, 0, 1)"]
    )
    q = model.layers[5].get_quantizers()[1]
    assert_equal(q.__str__(), "quantized_bits(10,3,1)")


if __name__ == "__main__":
    pytest.main([__file__])
