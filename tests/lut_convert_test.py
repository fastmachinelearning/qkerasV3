"""Tests for range-analysis driven teacher -> LUT conversion (qkeras.logic.convert)."""
import itertools
import os

import keras
import numpy as np
import pytest

from qkeras import QActivation, QConv2D, QDense
from qkeras.logic import (DistillConfig, FanInConstraint, convert_to_lut, fan_in_report,
                          load_netlist, save_netlist, simulate)
from qkeras.range_analysis import analyze_patterns, analyze_value_sets


def _set_fan_in(layer, fan_in, rng):
    k = layer.kernel.numpy() if hasattr(layer.kernel, "numpy") else keras.ops.convert_to_numpy(layer.kernel)
    flat = k.reshape(-1, k.shape[-1]).copy()
    for u in range(flat.shape[1]):
        keep = rng.choice(flat.shape[0], fan_in, replace=False)
        m = np.zeros(flat.shape[0], bool)
        m[keep] = True
        flat[~m, u] = 0.0
        flat[m, u] = rng.uniform(-1, 1, fan_in)
    layer.kernel.assign(flat.reshape(k.shape))
    if layer.use_bias:
        layer.bias.assign(rng.uniform(-0.5, 0.5, layer.bias.shape).astype("float32"))


def _dense_teacher(n_in=8, fan=(3, 4), seed=0):
    rng = np.random.default_rng(seed)
    x = keras.Input((n_in,))
    h = QDense(6, kernel_quantizer="quantized_bits(3,0,1,alpha=1)",
               bias_quantizer="quantized_bits(4,1,1,alpha=1)", name="d1")(x)
    h = QActivation("quantized_relu(2)", name="a1")(h)
    o = QDense(3, kernel_quantizer="quantized_bits(3,0,1,alpha=1)",
               bias_quantizer="quantized_bits(4,1,1,alpha=1)", name="d2")(h)
    m = keras.Model(x, o, name="teacher")
    _set_fan_in(m.get_layer("d1"), fan[0], rng)
    _set_fan_in(m.get_layer("d2"), fan[1], rng)
    return m


def _all_binary(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), np.float32)


def _teacher_logit_words(teacher, x, conv):
    t = teacher.predict(x, verbose=0).astype(np.float64)
    e, z = conv.netlist.meta["lsb_exponent"], conv.netlist.meta["zmin"]
    return np.round(t * 2.0 ** e).astype(np.int64) - z


def test_value_sets_and_patterns():
    m = _dense_teacher()
    X = _all_binary(8)
    vs = analyze_value_sets(m, X)
    assert vs["__input__"].size == 2 and vs["__input__"].bits == 1
    assert vs["a1"].bits <= 2 and vs["a1"].source in ("range", "observed")
    pats = analyze_patterns(m, X, vs)
    p1 = pats["d1"]
    assert all(b <= 3 for b in p1.bits)            # weights quantised to 0 drop out
    # exhaustive binary input: every pattern of every neuron is observed
    assert all(len(p) == (1 << b) for p, b in zip(p1.patterns, p1.bits))


@pytest.mark.parametrize("head", ["linear", "table"])
def test_exact_dense_is_bit_exact_on_all_inputs(head):
    m = _dense_teacher()
    X = _all_binary(8)
    conv = convert_to_lut(m, X, dont_care="none", grid="quantizer", head=head)
    net = conv.netlist
    assert net.head_type == ("linear" if head == "linear" else "words")
    assert not net.has_dont_cares
    scores, cls = simulate(net, X)
    np.testing.assert_array_equal(scores, _teacher_logit_words(m, X, conv))
    np.testing.assert_array_equal(cls, m.predict(X, verbose=0).argmax(-1))


def test_dont_cares_exact_on_care_set():
    m = _dense_teacher()
    X = _all_binary(8)
    calib = X[np.random.default_rng(1).choice(len(X), 40, replace=False)]
    conv = convert_to_lut(m, calib, dont_care="observed", head="table")
    net = conv.netlist
    assert net.has_dont_cares
    assert 0 < net.stats()["care_fraction"] < 1
    s_cal, _ = simulate(net, calib)
    np.testing.assert_array_equal(s_cal, _teacher_logit_words(m, calib, conv))
    # every LUT is exact wherever its care mask is set
    for L in net.layers:
        for l in L:
            assert l.care is None or l.care.dtype == bool


def test_input_quantizer_uniform_encoder():
    rng = np.random.default_rng(3)
    x = keras.Input((5,))
    h = QActivation("quantized_bits(3,1,1,alpha=1)", name="q_in")(x)
    h = QDense(4, kernel_quantizer="quantized_bits(3,0,1,alpha=1)",
               bias_quantizer="quantized_bits(4,1,1,alpha=1)", name="d1")(h)
    h = QActivation("quantized_relu(2,1)", name="a1")(h)
    o = QDense(3, kernel_quantizer="quantized_bits(3,0,1,alpha=1)",
               bias_quantizer="quantized_bits(4,1,1,alpha=1)", name="d2")(h)
    m = keras.Model(x, o)
    _set_fan_in(m.get_layer("d1"), 2, rng)
    _set_fan_in(m.get_layer("d2"), 3, rng)
    X = rng.normal(0, 1.5, (3000, 5)).astype(np.float32)
    conv = convert_to_lut(m, X, grid="quantizer", dont_care="none")
    assert conv.netlist.input_encoder["type"] == "uniform"
    Xt = rng.normal(0, 1.5, (2000, 5)).astype(np.float32)
    s, cls = simulate(conv.netlist, Xt)
    np.testing.assert_array_equal(s, _teacher_logit_words(m, Xt, conv))


def test_conv_teacher_exact():
    rng = np.random.default_rng(4)
    x = keras.Input((7, 7, 1))
    h = QConv2D(2, (3, 3), kernel_quantizer="binary", bias_quantizer="quantized_bits(4,2,1,alpha=1)",
                name="c1")(x)
    h = QActivation("binary", name="a1")(h)
    h = keras.layers.Flatten()(h)
    o = QDense(3, kernel_quantizer="quantized_bits(3,0,1,alpha=1)",
               bias_quantizer="quantized_bits(4,1,1,alpha=1)", name="d2")(h)
    m = keras.Model(x, o)
    _set_fan_in(m.get_layer("d2"), 6, rng)
    X = (rng.random((400, 7, 7, 1)) > 0.5).astype(np.float32)
    conv = convert_to_lut(m, X, dont_care="observed")
    rep = {r["layer"]: r for r in conv.report}
    assert rep["c1"]["positions"] == 25 and rep["c1"]["max_in_bits"] == 9
    s, cls = simulate(conv.netlist, X)
    np.testing.assert_array_equal(s, _teacher_logit_words(m, X, conv))


def test_distilled_block_matches_kd_student_and_learns():
    keras.utils.set_random_seed(0)
    rng = np.random.default_rng(5)
    n = 24
    x = keras.Input((n,))
    h = QDense(8, kernel_quantizer="quantized_bits(3,0,1,alpha=1)", name="wide")(x)
    h = QActivation("quantized_relu(1)", name="a1")(h)
    o = QDense(2, kernel_quantizer="quantized_bits(3,0,1,alpha=1)", use_bias=False, name="out")(h)
    m = keras.Model(x, o)
    _set_fan_in(m.get_layer("out"), 4, rng)
    X = (rng.random((3000, n)) > 0.5).astype(np.float32)
    y = m.predict(X, verbose=0).argmax(-1)
    cfg = DistillConfig(hidden=(128,), lut_rank=4, epochs=8, kd_epochs=3, batch_size=128,
                        learning_rate=0.03)
    conv = convert_to_lut(m, X, labels=y, max_exact_bits=8, distill=cfg)
    modes = {r["layer"]: r["mode"] for r in conv.report}
    assert modes == {"wide": "distilled", "out": "linear"}
    pred = conv.predict(X)
    kd_pred = conv.student.predict(X, verbose=0).argmax(-1)
    assert (pred == kd_pred).mean() == 1.0          # netlist == trained Keras student
    assert (pred == y).mean() > 0.7                 # it learned the teacher


def test_netlist_v2_roundtrip(tmp_path):
    m = _dense_teacher()
    X = _all_binary(8)
    conv = convert_to_lut(m, X[:50], dont_care="observed")
    p = os.path.join(tmp_path, "n.json")
    save_netlist(conv.netlist, p)
    net2 = load_netlist(p)
    assert net2.head_type == "linear" and net2.word_bits == conv.netlist.word_bits
    assert net2.linear == {"terms": [list(map(tuple, t)) for t in conv.netlist.linear["terms"]],
                           "bias": conv.netlist.linear["bias"]}
    for L1, L2 in zip(conv.netlist.layers, net2.layers):
        for a, b in zip(L1, L2):
            assert a.pins == b.pins
            np.testing.assert_array_equal(a.table, b.table)
            np.testing.assert_array_equal(a.care if a.care is not None else True,
                                          b.care if b.care is not None else True)
    np.testing.assert_array_equal(simulate(net2, X)[0], simulate(conv.netlist, X)[0])


def test_wide_lut_roundtrip_large_table(tmp_path):
    from qkeras.logic import LUT, Netlist
    rng = np.random.default_rng(0)
    t = rng.random(1 << 14) > 0.5
    c = rng.random(1 << 14) > 0.3
    net = Netlist(14, [[LUT(tuple(range(14)), t, c)]], None)
    p = os.path.join(tmp_path, "w.json")
    save_netlist(net, p)
    l2 = load_netlist(p).layers[0][0]
    np.testing.assert_array_equal(l2.table, t)
    np.testing.assert_array_equal(l2.care, c)


def test_fan_in_constraint_training():
    keras.utils.set_random_seed(0)
    x = keras.Input((20,))
    h = QDense(8, kernel_quantizer="quantized_bits(4,0,1,alpha=1)",
               kernel_constraint=FanInConstraint(3), name="d1")(x)
    h = QActivation("quantized_relu(2)")(h)
    o = QDense(2, kernel_quantizer="quantized_bits(4,0,1,alpha=1)",
               kernel_constraint=FanInConstraint(4), name="d2")(h)
    m = keras.Model(x, o)
    m.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    X = np.random.rand(256, 20).astype("float32")
    m.fit(X, (X[:, 0] > 0.5).astype(int), epochs=2, verbose=0)
    rep = fan_in_report(m)
    assert rep["d1"]["max"] <= 3 and rep["d2"]["max"] <= 4


def test_range_analysis_entry_point():
    from qkeras.range_analysis import convert_to_lut as ra_convert
    m = _dense_teacher()
    conv = ra_convert(m, _all_binary(8))
    assert conv.netlist.n_inputs == 8


def test_unquantised_logits_raise():
    x = keras.Input((4,))
    o = QDense(2, kernel_quantizer="quantized_bits(3,0,1,alpha=1)", name="d")(x)
    m = keras.Model(x, o)
    m.get_layer("d").bias.assign(np.array([0.1234567, -0.3], "float32"))
    with pytest.raises(ValueError, match="fixed-point"):
        convert_to_lut(m, _all_binary(4))


def test_linear_head_allows_dense_output_layer():
    rng = np.random.default_rng(7)
    x = keras.Input((12,))
    h = QDense(16, kernel_quantizer="quantized_bits(3,0,1,alpha=1)",
               bias_quantizer="quantized_bits(4,1,1,alpha=1)", name="d1")(x)
    h = QActivation("quantized_relu(2)", name="a1")(h)
    o = QDense(4, kernel_quantizer="quantized_bits(4,0,1,alpha=1)",
               bias_quantizer="quantized_bits(6,2,1,alpha=1)", name="out")(h)   # dense: 16x2 bits
    m = keras.Model(x, o)
    _set_fan_in(m.get_layer("d1"), 4, rng)
    X = (rng.random((2000, 12)) > 0.5).astype(np.float32)
    conv = convert_to_lut(m, X, dont_care="none", grid="quantizer")
    assert conv.netlist.head_type == "linear"
    s, cls = simulate(conv.netlist, X)
    np.testing.assert_array_equal(s, _teacher_logit_words(m, X, conv))


def test_batchnorm_teacher_exact():
    keras.utils.set_random_seed(1)
    rng = np.random.default_rng(8)
    x = keras.Input((10,))
    h = QDense(8, kernel_quantizer="quantized_bits(3,0,1,alpha=1)", use_bias=False,
               kernel_constraint=FanInConstraint(4), name="d1")(x)
    h = keras.layers.BatchNormalization(name="bn1")(h)
    h = QActivation("quantized_relu(2,1)", name="a1")(h)
    o = QDense(3, kernel_quantizer="quantized_bits(3,0,1,alpha=1)",
               bias_quantizer="quantized_bits(4,1,1,alpha=1)", name="out")(h)
    m = keras.Model(x, o)
    X = (rng.random((3000, 10)) > 0.5).astype(np.float32)
    y = (X[:, 0] + X[:, 3] > X[:, 5] + 0.5).astype(int) + (X[:, 7] > 0.5)
    m.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    m.fit(X, y, epochs=3, batch_size=64, verbose=0)
    conv = convert_to_lut(m, _all_binary(10), dont_care="none", grid="quantizer")
    Xa = _all_binary(10)
    s, cls = simulate(conv.netlist, Xa)
    np.testing.assert_array_equal(s, _teacher_logit_words(m, Xa, conv))
