"""Tests for the Keras-3 native pruning API (qkeras.pruning).

Run for each backend, e.g. ``KERAS_BACKEND=jax pytest tests/pruning_test.py``.
"""

import numpy as np
import pytest
import keras
from keras import ops

from qkeras import (
    QConv1D,
    QConv1DTranspose,
    QConv2D,
    QConv2DBatchnorm,
    QConv2DTranspose,
    QDense,
    QDenseBatchnorm,
    QDepthwiseConv2D,
    QDepthwiseConv2DBatchnorm,
    QSeparableConv1D,
    QSeparableConv2D,
)
from qkeras.pruning import (
    ConstantSparsity,
    MagnitudePruner,
    PolynomialDecay,
    PruningCallback,
    UpdatePruningStep,
    apply_pruning_masks,
    get_prunable_layers,
    get_pruning_summary,
    prune_low_magnitude,
    strip_pruning,
)
from qkeras.utils import load_qmodel


def _np(x):
    return ops.convert_to_numpy(x)


# ----------------------------------------------------------------- schedules
def test_constant_sparsity_update_points():
    s = ConstantSparsity(0.5, begin_step=10, end_step=50, frequency=20)
    due = [t for t in range(0, 80) if s(t) is not None]
    assert due == [10, 30, 50]
    assert s(30) == 0.5


def test_schedule_window_for_steps_per_execution():
    s = ConstantSparsity(0.5, begin_step=10, end_step=-1, frequency=20)
    assert s(12, prev_step=8) == 0.5  # 10 is in (8, 12]
    assert s(29, prev_step=12) is None
    assert s(33, prev_step=29) == 0.5  # 30 in (29, 33]


def test_polynomial_decay_values():
    s = PolynomialDecay(0.1, 0.8, begin_step=0, end_step=100, power=3, frequency=10)
    assert s.target_sparsity(0) == pytest.approx(0.1)
    assert s.target_sparsity(50) == pytest.approx(0.8 + (0.1 - 0.8) * 0.5**3)
    assert s.target_sparsity(100) == pytest.approx(0.8)
    assert s.target_sparsity(1000) == pytest.approx(0.8)
    # last update exactly at end_step even if not on the frequency grid
    s2 = PolynomialDecay(0.0, 0.8, begin_step=0, end_step=95, frequency=10)
    assert s2(95) == pytest.approx(0.8)
    assert s2(100) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(target_sparsity=1.0),
        dict(target_sparsity=0.5, frequency=0),
        dict(target_sparsity=0.5, begin_step=10, end_step=5),
    ],
)
def test_schedule_validation(kwargs):
    with pytest.raises(ValueError):
        ConstantSparsity(**kwargs)


def test_pruner_accepts_float_and_rejects_garbage():
    assert isinstance(MagnitudePruner(0.3).schedule, ConstantSparsity)
    with pytest.raises(TypeError):
        MagnitudePruner("0.3")
    with pytest.raises(TypeError):
        QDense(4, kernel_pruner=0.5)


# ------------------------------------------------------------------- masks
@pytest.mark.parametrize("sparsity", [0.0, 0.25, 0.5, 0.9, 0.999])
def test_magnitude_mask_exact_sparsity(sparsity):
    rng = np.random.default_rng(0)
    w = rng.normal(size=(17, 13)).astype("float32")
    m = _np(MagnitudePruner(0.5).compute_mask(w, sparsity))
    assert set(np.unique(m)) <= {0.0, 1.0}
    assert int((m == 0).sum()) == int(round(sparsity * w.size))
    if (m == 0).any() and (m == 1).any():
        # every kept weight is at least as large as every pruned one
        assert np.abs(w[m == 1]).min() >= np.abs(w[m == 0]).max()


def test_magnitude_mask_removes_small_weights():
    w = np.array([[10.0, 0.01], [0.02, 20.0]], "float32")
    m = _np(MagnitudePruner(0.5).compute_mask(w, 0.5))
    np.testing.assert_array_equal(m, [[1, 0], [0, 1]])


# ------------------------------------------------------------------ layers
LAYER_CASES = [
    (lambda p: QDense(6, kernel_pruner=p), (5, 8), ["kernel"]),
    (
        lambda p: QDense(6, kernel_quantizer="quantized_bits(4,0,1)", kernel_pruner=p),
        (5, 8),
        ["kernel"],
    ),
    (lambda p: QConv1D(4, 3, kernel_pruner=p), (2, 10, 3), ["kernel"]),
    (lambda p: QConv2D(4, 3, kernel_pruner=p), (2, 8, 8, 3), ["kernel"]),
    (
        lambda p: QConv1DTranspose(4, 3, kernel_pruner=p),
        (2, 10, 3),
        ["kernel"],
    ),
    (
        lambda p: QConv2DTranspose(4, 3, kernel_pruner=p),
        (2, 6, 6, 3),
        ["kernel"],
    ),
    (
        lambda p: QSeparableConv1D(4, 3, kernel_pruner=p),
        (2, 10, 3),
        ["depthwise_kernel", "pointwise_kernel"],
    ),
    (
        lambda p: QSeparableConv2D(4, 3, kernel_pruner=p),
        (2, 8, 8, 3),
        ["depthwise_kernel", "pointwise_kernel"],
    ),
    (
        lambda p: QDepthwiseConv2D(3, kernel_pruner=p),
        (2, 8, 8, 3),
        ["depthwise_kernel"],
    ),
    (lambda p: QDenseBatchnorm(6, kernel_pruner=p), (5, 8), ["kernel"]),
    (lambda p: QConv2DBatchnorm(4, 3, kernel_pruner=p), (2, 8, 8, 3), ["kernel"]),
    (
        lambda p: QDepthwiseConv2DBatchnorm(3, kernel_pruner=p),
        (2, 8, 8, 3),
        ["depthwise_kernel"],
    ),
]
LAYER_IDS = [
    "QDense",
    "QDense-quant",
    "QConv1D",
    "QConv2D",
    "QConv1DTranspose",
    "QConv2DTranspose",
    "QSeparableConv1D",
    "QSeparableConv2D",
    "QDepthwiseConv2D",
    "QDenseBatchnorm",
    "QConv2DBatchnorm",
    "QDepthwiseConv2DBatchnorm",
]


@pytest.mark.parametrize("make,shape,names", LAYER_CASES, ids=LAYER_IDS)
def test_layer_masks_and_forward(make, shape, names, request):
    if keras.backend.backend() == "torch" and "QDenseBatchnorm" in request.node.name:
        pytest.xfail("QDenseBatchnorm is broken on torch independently of pruning")
    x = np.random.default_rng(1).normal(size=shape).astype("float32")

    plain = make(None)
    plain(x)
    n_plain = len(plain.weights)
    assert get_prunable_layers(keras.Sequential([keras.Input(shape[1:]), plain])) == []

    layer = make(MagnitudePruner(0.5))
    layer(x)
    # exactly one extra, non-trainable weight per prunable kernel
    assert len(layer.weights) == n_plain + len(names)
    assert sorted(layer.pruning_masks()) == sorted(names)
    for m in layer.pruning_masks().values():
        assert not m.trainable

    # copy weights so both layers are identical, then zero some mask entries
    for name in names:
        getattr(layer, name).assign(getattr(plain, name))
    for v in plain.weights:
        for u in layer.weights:
            if u.name == v.name:
                u.assign(v)
    np.testing.assert_allclose(_np(layer(x)), _np(plain(x)), rtol=1e-5, atol=1e-5)

    rng = np.random.default_rng(2)
    for name, m in layer.pruning_masks().items():
        new = (rng.random(m.shape) > 0.5).astype("float32")
        m.assign(new)
        getattr(plain, name).assign(_np(getattr(plain, name)) * new)
    np.testing.assert_allclose(_np(layer(x)), _np(plain(x)), rtol=1e-5, atol=1e-5)


def test_mask_applied_before_quantization():
    layer = QDense(
        4, kernel_quantizer="quantized_bits(4,0,1)", kernel_pruner=MagnitudePruner(0.5)
    )
    x = np.eye(8, dtype="float32")
    layer(x)
    mask = np.zeros((8, 4), "float32")
    mask[::2] = 1
    layer.pruning_masks()["kernel"].assign(mask)
    y = _np(layer(x)) - _np(layer.bias)  # rows of the effective kernel
    assert np.all(y[1::2] == 0)


# ---------------------------------------------------------------- training
def _data(n=512, d=16, classes=4, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype("float32")
    w = rng.normal(size=(d, classes))
    y = keras.utils.to_categorical(np.argmax(x @ w, 1), classes)
    return x, y


def _model(pruner, quant=True):
    kq = "quantized_bits(8,0,1)" if quant else None
    inp = keras.Input((16,))
    h = QDense(32, kernel_quantizer=kq, kernel_pruner=pruner, name="d0")(inp)
    h = keras.layers.Activation("relu")(h)
    out = QDense(4, kernel_quantizer=kq, kernel_pruner=pruner, name="d1")(h)
    return keras.Model(inp, keras.layers.Activation("softmax")(out))


@pytest.mark.parametrize("steps_per_execution", [1, 3])
def test_fit_reaches_final_sparsity(steps_per_execution):
    if steps_per_execution > 1 and keras.backend.backend() == "torch":
        pytest.skip("steps_per_execution > 1 is not supported by the torch backend")
    keras.utils.set_random_seed(0)
    x, y = _data()
    pruner = MagnitudePruner(
        PolynomialDecay(0.0, 0.8, begin_step=4, end_step=40, frequency=4)
    )
    model = _model(pruner)
    model.compile(
        "adam", "categorical_crossentropy", steps_per_execution=steps_per_execution
    )
    model.fit(
        x, y, batch_size=32, epochs=4, verbose=0, callbacks=[UpdatePruningStep()]
    )  # 64 steps
    summary = get_pruning_summary(model)
    assert summary["d0"]["kernel"] == pytest.approx(0.8, abs=1e-2)
    assert summary["d1"]["kernel"] == pytest.approx(0.8, abs=1e-2)
    # baked: the stored kernel itself is sparse
    for name in ["d0", "d1"]:
        k = _np(model.get_layer(name).kernel)
        assert np.mean(k == 0) == pytest.approx(0.8, abs=1e-2)


def test_mask_is_active_inside_compiled_train_step():
    """Pruned weights must receive zero gradient during fit().

    With plain SGD (no momentum/decay) a pruned weight that was zeroed at mask
    update time stays exactly zero iff the mask is used inside the compiled
    train step *and* the callback's assignments survive (JAX state sync).
    """
    keras.utils.set_random_seed(0)
    x, y = _data()
    pruner = MagnitudePruner(ConstantSparsity(0.7, begin_step=2, frequency=10_000))
    model = _model(pruner, quant=False)
    model.compile(keras.optimizers.SGD(0.1), "categorical_crossentropy")
    model.fit(
        x,
        y,
        batch_size=32,
        epochs=2,
        verbose=0,
        callbacks=[UpdatePruningStep(bake_on_train_end=False)],
    )
    for name in ["d0", "d1"]:
        layer = model.get_layer(name)
        m = _np(layer.pruning_masks()["kernel"])
        k = _np(layer.kernel)
        assert np.mean(m == 0) == pytest.approx(0.7, abs=1e-2)
        assert np.all(k[m == 0] == 0)
        assert np.all(k[m == 1] != 0)


def test_pretrain_prune_finetune_keeps_accuracy():
    """Standard workflow: pretrain dense, prune to 50 %, fine-tune.

    Gradual magnitude pruning + fine-tuning must recover the dense accuracy
    and beat one-shot magnitude pruning without fine-tuning.
    """
    x, y = _data(n=2048)
    keras.utils.set_random_seed(0)
    base = _model(None)
    base.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    base.fit(x, y, batch_size=32, epochs=15, verbose=0)
    acc_dense = base.evaluate(x, y, verbose=0)[1]

    oneshot = keras.models.clone_model(base)
    oneshot.set_weights(base.get_weights())
    oneshot.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    for layer in oneshot.layers:
        if hasattr(layer, "kernel"):
            k = _np(layer.kernel)
            layer.kernel.assign(k * (np.abs(k) > np.quantile(np.abs(k), 0.5)))
    acc_oneshot = oneshot.evaluate(x, y, verbose=0)[1]

    pruned = prune_low_magnitude(
        base, PolynomialDecay(0.0, 0.5, begin_step=0, end_step=320, frequency=32)
    )
    pruned.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    pruned.fit(
        x, y, batch_size=32, epochs=10, verbose=0, callbacks=[UpdatePruningStep()]
    )
    acc_pruned = pruned.evaluate(x, y, verbose=0)[1]

    assert get_pruning_summary(pruned)["__total__"] == pytest.approx(0.5, abs=1e-2)
    assert acc_pruned >= acc_dense - 0.02
    assert acc_pruned > acc_oneshot


def test_backward_compatible_callback_name():
    assert PruningCallback is UpdatePruningStep


def test_callback_warns_without_prunable_layers():
    x, y = _data(n=64)
    model = _model(None)
    model.compile("adam", "categorical_crossentropy")
    with pytest.warns(UserWarning, match="no layers with a kernel_pruner"):
        model.fit(x, y, epochs=1, verbose=0, callbacks=[UpdatePruningStep()])


def test_nested_model_is_pruned():
    keras.utils.set_random_seed(0)
    x, y = _data()
    pruner = MagnitudePruner(ConstantSparsity(0.6, begin_step=1, frequency=5))
    inner = keras.Sequential(
        [keras.Input((16,)), QDense(8, kernel_pruner=pruner, name="inner_d")],
        name="inner",
    )
    inp = keras.Input((16,))
    out = keras.layers.Dense(4, activation="softmax")(inner(inp))
    model = keras.Model(inp, out)
    model.compile("adam", "categorical_crossentropy")
    model.fit(x, y, epochs=1, verbose=0, callbacks=[UpdatePruningStep()])
    assert get_pruning_summary(model)["inner_d"]["kernel"] == pytest.approx(
        0.6, abs=1e-2
    )


# ------------------------------------------------------- saving / cloning
def _trained_pruned_model():
    keras.utils.set_random_seed(0)
    x, y = _data()
    pruner = MagnitudePruner(
        PolynomialDecay(0.0, 0.75, begin_step=0, end_step=16, frequency=4)
    )
    model = _model(pruner)
    model.compile("adam", "categorical_crossentropy")
    model.fit(
        x,
        y,
        batch_size=32,
        epochs=2,
        verbose=0,
        callbacks=[UpdatePruningStep(bake_on_train_end=False)],
    )
    return model, x


def test_save_and_load_keras_format(tmp_path):
    model, x = _trained_pruned_model()
    path = str(tmp_path / "pruned.keras")
    model.save(path)
    for loaded in (keras.models.load_model(path), load_qmodel(path)):
        layer = loaded.get_layer("d0")
        assert isinstance(layer.kernel_pruner, MagnitudePruner)
        assert isinstance(layer.kernel_pruner.schedule, PolynomialDecay)
        np.testing.assert_array_equal(
            _np(layer.pruning_masks()["kernel"]),
            _np(model.get_layer("d0").pruning_masks()["kernel"]),
        )
        np.testing.assert_allclose(
            loaded.predict(x, verbose=0), model.predict(x, verbose=0), atol=1e-6
        )


def test_save_and_load_weights_h5(tmp_path):
    model, x = _trained_pruned_model()
    path = str(tmp_path / "pruned.weights.h5")
    model.save_weights(path)
    fresh = _model(MagnitudePruner(0.75))
    fresh.load_weights(path)
    np.testing.assert_allclose(
        fresh.predict(x, verbose=0), model.predict(x, verbose=0), atol=1e-6
    )


def test_strip_pruning():
    model, x = _trained_pruned_model()
    stripped = strip_pruning(model)
    assert get_prunable_layers(stripped) == []
    for name in ["d0", "d1"]:
        layer = stripped.get_layer(name)
        assert layer.kernel_pruner is None
        assert [v.name for v in layer.weights] == ["kernel", "bias"]
        assert not hasattr(layer, "pruning_mask_holder")
        assert np.mean(_np(layer.kernel) == 0) == pytest.approx(0.75, abs=1e-2)
        assert layer.get_config()["kernel_pruner"] is None
    np.testing.assert_allclose(
        stripped.predict(x, verbose=0), model.predict(x, verbose=0), atol=1e-6
    )


def test_apply_pruning_masks_in_place():
    model, x = _trained_pruned_model()
    before = model.predict(x, verbose=0)
    apply_pruning_masks(model)
    k = _np(model.get_layer("d0").kernel)
    m = _np(model.get_layer("d0").pruning_masks()["kernel"])
    assert np.all(k[m == 0] == 0)
    np.testing.assert_allclose(model.predict(x, verbose=0), before, atol=1e-6)


@pytest.mark.parametrize("kind", ["functional", "sequential"])
def test_prune_low_magnitude(kind):
    keras.utils.set_random_seed(0)
    x, y = _data()
    if kind == "functional":
        base = _model(None)
    else:
        base = keras.Sequential(
            [
                keras.Input((16,)),
                QDense(32, kernel_quantizer="quantized_bits(8,0,1)", name="d0"),
                keras.layers.Activation("relu"),
                QDense(4, name="d1"),
                keras.layers.Activation("softmax"),
            ]
        )
    pruner = PolynomialDecay(0.0, 0.5, begin_step=0, end_step=20, frequency=5)
    pruned = prune_low_magnitude(base, pruner)
    assert {layer.name for layer in get_prunable_layers(pruned)} == {"d0", "d1"}
    # weights are copied, masks start at ones -> identical outputs
    np.testing.assert_allclose(
        pruned.predict(x, verbose=0), base.predict(x, verbose=0), atol=1e-6
    )
    pruned.compile("adam", "categorical_crossentropy")
    pruned.fit(x, y, epochs=2, verbose=0, callbacks=[UpdatePruningStep()])
    assert get_pruning_summary(pruned)["__total__"] == pytest.approx(0.5, abs=1e-2)


def test_prune_low_magnitude_layer_subset_and_errors():
    base = _model(None)
    pruned = prune_low_magnitude(base, 0.5, layer_names=["d1"])
    assert [layer.name for layer in get_prunable_layers(pruned)] == ["d1"]
    with pytest.raises(ValueError, match="not found or not prunable"):
        prune_low_magnitude(base, 0.5, layer_names=["nope"])

    class Sub(keras.Model):
        def __init__(self):
            super().__init__()
            self.d = QDense(2)

        def call(self, x):
            return self.d(x)

    sub = Sub()
    sub(np.zeros((1, 3), "float32"))
    with pytest.raises(ValueError, match="Sequential and functional"):
        prune_low_magnitude(sub, 0.5)
