import os
import numpy as np
import pytest

# Force TF backend for Keras 3
os.environ.setdefault("KERAS_BACKEND", "tensorflow")


def _set_deterministic_seed(seed: int = 123):
    import tensorflow as tf

    np.random.seed(seed)
    #tf.random.set_seed(seed)


def _make_qdense_model():
    import tensorflow as tf
    from tensorflow import keras
    from qkeras import QDense, quantized_bits, QActivation

    inputs = keras.Input(shape=(16,), name="x")

    x = QDense(
        8,
        name="qdense_0",
        kernel_quantizer=quantized_bits(8, 0, 1),
        bias_quantizer=quantized_bits(8, 0, 1),
        activation="relu",   # keep conversion stable
    )(inputs)

    x = QActivation(activation="quantized_sigmoid")(x)

    outputs = QDense(
        4,
        name="qdense_1",
        kernel_quantizer=quantized_bits(8, 0, 1),
        bias_quantizer=quantized_bits(8, 0, 1),
        activation=None,     # logits (avoid softmax codegen issues)
    )(x)

    model = keras.Model(inputs, outputs, name="qdense_model")
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    )
    for layer in model.layers:
        if hasattr(layer, "activation"):
            print(layer, layer.activation)
    return model


def _make_qconv1d_model():
    import tensorflow as tf
    from tensorflow import keras
    from qkeras import QConv1D, QDense, quantized_bits

    inputs = keras.Input(shape=(32, 1), name="x")

    x = QConv1D(
        4,
        kernel_size=3,
        padding="same",
        name="qconv1d_0",
        kernel_quantizer=quantized_bits(6, 0, 1),
        bias_quantizer=quantized_bits(6, 0, 1),
        activation="relu",  # stable + supported
    )(inputs)

    x = keras.layers.Flatten(name="flatten")(x)

    outputs = QDense(
        3,
        name="qdense_out",
        kernel_quantizer=quantized_bits(6, 0, 1),
        bias_quantizer=quantized_bits(6, 0, 1),
        activation=None,    # logits
    )(x)

    model = keras.Model(inputs, outputs, name="qconv1d_model")
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    )
    return model


def _convert_compile_and_predict(model, out_dir, x_np: np.ndarray):
    import hls4ml

    config = hls4ml.utils.config_from_keras_model(model, granularity="name")
    config["Model"]["ReuseFactor"] = 1
    config["Model"]["Strategy"] = "Latency"

    # Important: ensure we generate a C-sim path that can run predict()
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(out_dir),
        backend="Vivado",
    )

    hls_model.compile()

    # Keras prediction (float)
    y_keras = model(x_np, training=False).numpy()

    # HLS prediction (C-sim)
    y_hls = hls_model.predict(x_np)

    return y_keras, y_hls


@pytest.mark.parametrize(
    "make_model,input_shape,atol,rtol",
    [
        (_make_qdense_model, (5, 16), 5e-2, 5e-2),
        (_make_qconv1d_model, (5, 32, 1), 1e-1, 1e-1),
    ],
)
def test_hls4ml_predictions_close(tmp_path, make_model, input_shape, atol, rtol):
    _set_deterministic_seed(123)

    model = make_model()

    # Use a modest range to avoid overflow with fixed-point
    x = np.random.uniform(-0.5, 0.5, size=input_shape).astype(np.float32)

    y_keras, y_hls = _convert_compile_and_predict(model, tmp_path / model.name, x)

    assert y_keras.shape == y_hls.shape

    # Compare within tolerances: fixed-point + quantization introduces error
    np.testing.assert_allclose(y_hls, y_keras, atol=atol, rtol=rtol)


if __name__ == "__main__":
    from pathlib import Path

    out = Path("hls_manual_runs")
    out.mkdir(exist_ok=True)

    model = _make_qdense_model()
    x = np.random.uniform(-0.5, 0.5, size=(5, 16)).astype(np.float32)
    y_keras, y_hls = _convert_compile_and_predict(model, out / model.name, x)

    assert y_keras.shape == y_hls.shape

    # Compare within tolerances: fixed-point + quantization introduces error
    np.testing.assert_allclose(y_hls, y_keras, atol=1e-1, rtol=1e-1)

    print("max abs diff:", np.max(np.abs(y_keras - y_hls)))
