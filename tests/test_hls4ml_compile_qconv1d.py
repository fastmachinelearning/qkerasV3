
import os
import re
import shutil
from dataclasses import dataclass

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "tensorflow")


@pytest.fixture(scope="session", autouse=True)
def require_vitis():
    if (
        shutil.which("vitis_hls") is None
        and shutil.which("vivado_hls") is None
    ):
        pytest.skip("Vitis/Vivado HLS is not installed")


@dataclass(frozen=True)
class QConv1DCase:
    input_len: int
    in_ch: int
    filters: int
    kernel_size: int
    padding: str
    stride: int
    kernel_bits: int
    bias_bits: int
    use_bias: bool
    conv_activation: str | None
    qactivation: str | None
    dense_units: int
    atol: float
    rtol: float

    @property
    def id(self) -> str:
        raw = "_".join(
            [
                "qconv1d",
                f"len{self.input_len}",
                f"ch{self.in_ch}",
                f"f{self.filters}",
                f"k{self.kernel_size}",
                self.padding,
                f"s{self.stride}",
                f"kb{self.kernel_bits}",
                f"bb{self.bias_bits}",
                f"bias{int(self.use_bias)}",
                f"conv_{self.conv_activation}",
                f"qact_{self.qactivation}",
                f"dense{self.dense_units}",
            ]
        )
        return re.sub(r"[^a-zA-Z0-9_]+", "_", raw).strip("_").lower()


def _quantizer(bits: int):
    from qkeras import quantized_bits

    return quantized_bits(bits, 0, 1)


def _build_model(case: QConv1DCase):
    import tensorflow as tf
    from tensorflow import keras
    from qkeras import QActivation, QConv1D, QDense

    inputs = keras.Input(shape=(case.input_len, case.in_ch), name="x")

    x = QConv1D(
        case.filters,
        kernel_size=case.kernel_size,
        strides=case.stride,
        padding=case.padding,
        name="qconv1d_0",
        kernel_quantizer=_quantizer(case.kernel_bits),
        bias_quantizer=_quantizer(case.bias_bits) if case.use_bias else None,
        use_bias=case.use_bias,
        activation=case.conv_activation,
    )(inputs)

    if case.qactivation is not None:
        x = QActivation(case.qactivation, name="q_activation")(x)

    x = keras.layers.Flatten(name="flatten")(x)

    outputs = QDense(
        case.dense_units,
        name="qdense_out",
        kernel_quantizer=_quantizer(case.kernel_bits),
        bias_quantizer=_quantizer(case.bias_bits) if case.use_bias else None,
        use_bias=case.use_bias,
        activation=None,
    )(x)

    model = keras.Model(inputs, outputs, name=case.id)
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

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(out_dir),
        backend="Vivado",
    )
    hls_model.compile()

    y_keras = model(x_np, training=False).numpy()
    y_hls = hls_model.predict(x_np)
    return y_keras, y_hls


def _convert_only(model, out_dir):
    import hls4ml

    config = hls4ml.utils.config_from_keras_model(model, granularity="name")
    config["Model"]["ReuseFactor"] = 1
    config["Model"]["Strategy"] = "Latency"

    return hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(out_dir),
        backend="Vivado",
    )


def _assert_outputs_close(y_keras, y_hls, *, atol, rtol):
    assert y_keras.shape == y_hls.shape, (y_keras.shape, y_hls.shape)
    try:
        np.testing.assert_allclose(y_hls, y_keras, atol=atol, rtol=rtol)
    except AssertionError as exc:
        diff = np.abs(y_hls - y_keras)
        raise AssertionError(
            f"{exc}\n"
            f"max abs diff: {diff.max()}\n"
            f"mean abs diff: {diff.mean()}\n"
            f"keras range: [{y_keras.min()}, {y_keras.max()}]\n"
            f"hls range:   [{y_hls.min()}, {y_hls.max()}]"
        ) from exc


ALL_CASES = [
    QConv1DCase(16, 1, 2, 1, "valid", 1, 4, 4, True, None, None, 3, 1.8e-1, 1.8e-1),
    QConv1DCase(16, 1, 4, 3, "valid", 1, 6, 6, True, "relu", None, 3, 1.2e-1, 1.2e-1),
    QConv1DCase(32, 1, 4, 3, "same", 1, 6, 6, True, "relu", "quantized_sigmoid", 3, 1.5e-1, 1.5e-1),
    QConv1DCase(32, 2, 2, 1, "same", 1, 8, 8, False, None, "quantized_tanh", 5, 1.0e-1, 1.0e-1),
    QConv1DCase(16, 2, 4, 3, "valid", 1, 8, 6, True, "relu", None, 5, 9.0e-2, 9.0e-2),
    QConv1DCase(32, 2, 2, 3, "same", 1, 4, 8, False, None, "quantized_sigmoid", 3, 2.1e-1, 2.1e-1),
]

NUMERIC_CASES = [
    pytest.param(case, id=case.id, marks=pytest.mark.numeric_close)
    for case in ALL_CASES
]

CONVERSION_ONLY_CASES = [
    pytest.param(
        case,
        id=case.id,
        marks=pytest.mark.conversion_only,
    )
    for case in [
        QConv1DCase(16, 1, 2, 1, "valid", 1, 4, 8, True, "relu", None, 3, 0.0, 0.0),
        QConv1DCase(32, 1, 4, 3, "same", 1, 8, 4, False, None, "quantized_tanh", 5, 0.0, 0.0),
        QConv1DCase(16, 2, 2, 3, "valid", 1, 6, 6, True, None, "quantized_sigmoid", 3, 0.0, 0.0),
        QConv1DCase(32, 2, 4, 1, "same", 1, 8, 8, True, "relu", None, 5, 0.0, 0.0),
    ]
]


@pytest.fixture(autouse=True)
def deterministic_seed():
    import tensorflow as tf

    np.random.seed(123)
    tf.random.set_seed(123)


@pytest.fixture
def x_conv(case: QConv1DCase):
    return np.random.uniform(-0.2, 0.2, size=(5, case.input_len, case.in_ch)).astype(np.float32)


@pytest.fixture
def model(case: QConv1DCase):
    return _build_model(case)


@pytest.mark.parametrize("case", CONVERSION_ONLY_CASES)
def test_qconv1d_converts(case, model, tmp_path):
    hls_model = _convert_only(model, tmp_path / case.id)
    assert hls_model is not None


@pytest.mark.parametrize("case", NUMERIC_CASES)
def test_qconv1d_predictions_close(case, model, x_conv, tmp_path):
    y_keras, y_hls = _convert_compile_and_predict(model, tmp_path / case.id, x_conv)
    _assert_outputs_close(y_keras, y_hls, atol=case.atol, rtol=case.rtol)
