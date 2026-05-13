
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
class QDenseCase:
    input_dim: int
    hidden_units: int
    output_units: int
    kernel_bits: int
    bias_bits: int
    use_bias: bool
    dense_activation: str | None
    qactivation: str | None
    final_activation: str | None
    atol: float
    rtol: float

    @property
    def id(self) -> str:
        raw = "_".join(
            [
                "qdense",
                f"in{self.input_dim}",
                f"h{self.hidden_units}",
                f"out{self.output_units}",
                f"kb{self.kernel_bits}",
                f"bb{self.bias_bits}",
                f"bias{int(self.use_bias)}",
                f"dense_{self.dense_activation}",
                f"qact_{self.qactivation}",
                f"final_{self.final_activation}",
            ]
        )
        return re.sub(r"[^a-zA-Z0-9_]+", "_", raw).strip("_").lower()


def _quantizer(bits: int):
    from qkeras import quantized_bits

    return quantized_bits(bits, 0, 1)


def _build_model(case: QDenseCase):
    import tensorflow as tf
    from tensorflow import keras
    from qkeras import QActivation, QDense

    inputs = keras.Input(shape=(case.input_dim,), name="x")

    x = QDense(
        case.hidden_units,
        name="qdense_0",
        kernel_quantizer=_quantizer(case.kernel_bits),
        bias_quantizer=_quantizer(case.bias_bits) if case.use_bias else None,
        use_bias=case.use_bias,
        activation=case.dense_activation,
    )(inputs)

    if case.qactivation is not None:
        x = QActivation(case.qactivation, name="q_activation")(x)

    outputs = QDense(
        case.output_units,
        name="qdense_1",
        kernel_quantizer=_quantizer(case.kernel_bits),
        bias_quantizer=_quantizer(case.bias_bits) if case.use_bias else None,
        use_bias=case.use_bias,
        activation=case.final_activation,
    )(x)

    model = keras.Model(inputs, outputs, name=case.id)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=(case.final_activation is None)),
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
    QDenseCase(8, 4, 3, 4, 4, True, None, None, None, 1.2e-1, 1.2e-1),
    QDenseCase(8, 4, 3, 6, 6, True, "relu", None, None, 8.0e-2, 8.0e-2),
    QDenseCase(16, 8, 4, 8, 8, True, "relu", "quantized_sigmoid", None, 6.0e-2, 6.0e-2),
    QDenseCase(16, 8, 4, 6, 6, False, None, "quantized_tanh", None, 9.0e-2, 9.0e-2),
    QDenseCase(16, 4, 3, 4, 6, True, "relu", "quantized_relu(4,0)", None, 1.4e-1, 1.4e-1),
    QDenseCase(8, 8, 4, 8, 4, False, None, "quantized_sigmoid", None, 7.0e-2, 7.0e-2),
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
        QDenseCase(8, 4, 3, 4, 8, True, "relu", None, None, 0.0, 0.0),
        QDenseCase(16, 8, 4, 8, 4, False, None, "quantized_tanh", None, 0.0, 0.0),
        QDenseCase(16, 4, 4, 6, 8, True, "relu", "quantized_sigmoid", None, 0.0, 0.0),
        QDenseCase(8, 8, 3, 8, 6, False, None, "quantized_relu(4,0)", None, 0.0, 0.0),
    ]
]


@pytest.fixture(autouse=True)
def deterministic_seed():
    import tensorflow as tf

    np.random.seed(123)
    tf.random.set_seed(123)


@pytest.fixture
def x_dense(case: QDenseCase):
    return np.random.uniform(-0.25, 0.25, size=(5, case.input_dim)).astype(np.float32)


@pytest.fixture
def model(case: QDenseCase):
    return _build_model(case)


@pytest.mark.parametrize("case", CONVERSION_ONLY_CASES)
def test_qdense_converts(case, model, tmp_path):
    hls_model = _convert_only(model, tmp_path / case.id)
    assert hls_model is not None


@pytest.mark.parametrize("case", NUMERIC_CASES)
def test_qdense_predictions_close(case, model, x_dense, tmp_path):
    y_keras, y_hls = _convert_compile_and_predict(model, tmp_path / case.id, x_dense)
    _assert_outputs_close(y_keras, y_hls, atol=case.atol, rtol=case.rtol)
