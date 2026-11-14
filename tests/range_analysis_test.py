"""Tests for qkeras.range_analysis (range-driven bit-width adaptation)."""

import keras
import numpy as np
import pytest

from qkeras import (
    QActivation,
    QConv2D,
    QDense,
    RangeAnalysisCallback,
    analyze_ranges,
    optimize_bitwidths,
    quantized_bits,
    quantized_relu,
)
from qkeras.qlayers import Clip
from qkeras.range_analysis import fixed_point_format, random_calibration_data


def _mlp(act_bits=(10, 0), relu=(10, 6)):
    inp = keras.Input((16,))
    x = QDense(
        32,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        activation=quantized_bits(*act_bits),
        name="d1",
    )(inp)
    x = QActivation(quantized_relu(*relu), name="a1")(x)
    x = QDense(
        4,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        name="d2",
    )(x)
    return keras.Model(inp, x)


def _data(n=1024, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 4, (n, 16)).astype("float32")


def _entry(report, layer, role):
    return next(e for e in report.entries if e.layer == layer and e.role == role)


@pytest.mark.parametrize(
    "q",
    [
        quantized_bits(8, 0),
        quantized_bits(8, 3),
        quantized_bits(6, -2),
        quantized_bits(8, 2, keep_negative=False),
        quantized_bits(8, 2, symmetric=1),
        quantized_relu(8, 0),
        quantized_relu(8, 4),
        quantized_relu(6, -1),
    ],
)
def test_fixed_point_format_matches_quantizer(q):
    fmt = fixed_point_format(q)
    x = np.linspace(-1000, 1000, 400001).astype("float32")
    y = keras.ops.convert_to_numpy(q(x))
    assert y.max() == pytest.approx(fmt["max"])
    assert y.min() == pytest.approx(fmt["min"])
    steps = np.diff(np.unique(y))
    assert steps.min() == pytest.approx(fmt["lsb"])


def test_unsupported_quantizers_are_reported_not_changed():
    inp = keras.Input((8,))
    x = QDense(4, kernel_quantizer=quantized_bits(6, 0), name="d")(inp)  # auto_po2
    m = keras.Model(inp, x)
    rep = optimize_bitwidths(m, np.ones((64, 8), "float32") * 3, verbose=0)
    e = _entry(rep, "d", "kernel")
    assert not e.supported and "auto_po2" in e.reason
    assert m.get_layer("d").kernel_quantizer_internal.bits == 6


def test_grows_to_minimal_integer_without_saturation():
    m = _mlp()
    x = _data()
    rep = optimize_bitwidths(m, x, verbose=0)
    e = _entry(rep, "d1", "activation")
    q = m.get_layer("d1").activation
    peak = max(e.stats.high, -e.stats.low)
    assert q.integer > 0 and e.saturated_fraction == 0.0
    # minimality: one integer bit less would not hold the peak
    assert peak > 2.0 ** (q.integer - 1)
    # keep_fractional: fractional bits unchanged (10 - 1 - 0 = 9)
    assert q.bits - 1 - q.integer == 9
    # downstream QActivation sized on the *grown* range (fixed-point iteration)
    a = m.get_layer("a1").activation
    assert 2.0**a.integer >= e.stats.high
    assert _entry(rep, "a1", "activation").saturated_fraction == 0.0


def test_keep_total_policy_keeps_bits():
    m = _mlp()
    optimize_bitwidths(m, _data(), policy="keep_total", verbose=0)
    assert m.get_layer("d1").activation.bits == 10
    assert m.get_layer("a1").activation.bits == 10
    assert m.get_layer("d1").activation.integer > 0


def test_dry_run_and_analyze_do_not_modify():
    m = _mlp()
    before = str(m.get_layer("d1").activation), str(m.get_layer("a1").activation)
    analyze_ranges(m, _data())
    rep = optimize_bitwidths(m, _data(), dry_run=True, verbose=0)
    assert rep.changes()
    after = str(m.get_layer("d1").activation), str(m.get_layer("a1").activation)
    assert before == after


def test_random_calibration_data_shape_and_range():
    m = _mlp()
    x = random_calibration_data(m, 100, low=-2, high=3)
    assert x.shape == (100, 16) and x.min() >= -2 and x.max() <= 3
    rep = optimize_bitwidths(m, None, random_low=0, random_high=4, verbose=0)
    assert _entry(rep, "d1", "activation").applied


def test_compiled_train_step_is_retraced():
    """After adaptation the cached train function must use the new widths."""
    m = _mlp()
    m.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0), loss="mse")
    x = _data(256)
    y = np.zeros((256, 4), "float32")
    loss_old = float(m.train_on_batch(x, y))  # builds and caches train_function
    optimize_bitwidths(m, x, verbose=0)
    loss_new_eval = float(m.evaluate(x, y, batch_size=256, verbose=0))
    loss_new_train = float(m.train_on_batch(x, y))
    assert loss_new_train == pytest.approx(loss_new_eval, rel=1e-4)
    assert abs(loss_new_train - loss_old) > 1e-3 * max(1.0, abs(loss_old))


def test_weight_clip_constraint_follows_quantizer():
    m = _mlp()
    d1 = m.get_layer("d1")
    k = d1.get_weights()[0]
    k[0, 0] = 3.0  # outside the initial [-1, 1] clip
    d1.set_weights([k, d1.get_weights()[1]])
    optimize_bitwidths(m, _data(), verbose=0)
    q = d1.kernel_quantizer_internal
    assert q.integer == 2
    assert isinstance(d1.kernel_constraint, Clip)
    assert d1.kernel_constraint.max_value == pytest.approx(4.0)


def test_include_exclude_and_layer_opt_out():
    m = _mlp()
    m.get_layer("a1").range_analysis = False
    rep = optimize_bitwidths(m, _data(), exclude=["d2"], verbose=0)
    assert {e.layer for e in rep.entries} == {"d1"}
    assert str(m.get_layer("a1").activation) == "quantized_relu(10,6)"


def test_serialization_roundtrip_keeps_new_widths():
    m = _mlp()
    optimize_bitwidths(m, _data(), verbose=0)
    m2 = keras.Model.from_config(m.get_config())
    for name in ("d1", "a1"):
        assert str(m2.get_layer(name).activation) == str(m.get_layer(name).activation)


def test_callback_toggle_window_and_patience():
    x = _data(512)
    y = np.zeros((512, 4), "float32")

    m = _mlp()
    m.compile("adam", "mse")
    cb = RangeAnalysisCallback(x, enabled=False, verbose=0)
    m.fit(x, y, epochs=2, batch_size=128, verbose=0, callbacks=[cb])
    assert cb.history == []

    cb.enabled = True
    cb.start_epoch, cb.stop_epoch = 1, 3
    h = m.fit(x, y, epochs=4, batch_size=128, verbose=0, callbacks=[cb])
    assert [r.epoch for r in cb.history] == [1, 2]
    assert "range_analysis_bit_delta" in h.history

    # oversized relu (10,6) must not shrink before `patience` checks agree
    m = _mlp(act_bits=(14, 4))
    m.compile("adam", "mse")
    cb = RangeAnalysisCallback(x, patience=3, adjust_weights=False, verbose=0)
    m.fit(x, y, epochs=2, batch_size=128, verbose=0, callbacks=[cb])
    assert m.get_layer("a1").activation.integer == 6
    m.fit(x, y, epochs=1, batch_size=128, verbose=0, callbacks=[cb])
    assert m.get_layer("a1").activation.integer < 6


def test_callback_grows_during_fit():
    x = _data(512)
    y = (x[:, :4] * 0.5).astype("float32")
    m = _mlp()
    m.compile("adam", "mse")
    cb = RangeAnalysisCallback("random", random_low=0, random_high=4, verbose=0)
    m.fit(x, y, epochs=1, batch_size=128, verbose=0, callbacks=[cb])
    assert m.get_layer("d1").activation.integer > 0
    rep = analyze_ranges(m, x)
    assert _entry(rep, "d1", "activation").saturated_fraction == 0.0


def test_conv_model():
    inp = keras.Input((8, 8, 2))
    x = QConv2D(
        4,
        3,
        kernel_quantizer=quantized_bits(6, 0),
        bias_quantizer=quantized_bits(6, 0),
        activation=quantized_relu(8, 0),
        name="c",
    )(inp)
    x = keras.layers.Flatten()(x)
    x = QDense(2, kernel_quantizer=quantized_bits(6, 0, alpha=1), name="d")(x)
    m = keras.Model(inp, x)
    xs = np.random.default_rng(0).uniform(0, 8, (128, 8, 8, 2)).astype("float32")
    rep = optimize_bitwidths(m, xs, verbose=0)
    assert _entry(rep, "c", "activation").new_integer > 0
    assert not _entry(rep, "c", "kernel").supported  # auto_po2
    assert "c" in rep.layer_ranges and rep.layer_ranges["c"]["output"].valid
    assert rep.hls4ml_precisions()["c"]["activation"].startswith("ufixed<")
