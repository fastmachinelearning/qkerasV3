"""Thermometer encoder + materialize_inputs: exactness of the hardware view."""
import numpy as np
import keras
import pytest

from qkeras.logic import (GroupSum, LUTDense, Thermometer, extract_netlist, fixed_point,
                          materialize_inputs, quantile_thresholds, simulate, save_netlist, load_netlist)
from qkeras.logic.netlist import encode_inputs


def _model(x, T=8, k=3, learn=False, grid=(8, 3)):
    inp = keras.Input((x.shape[1],))
    th = Thermometer(T, trainable_thresholds=learn, grid=grid, name="th")
    th.adapt(x)
    h = th(inp)
    h = LUTDense(40, lut_rank=k, connections="learnable", num_candidates=6, seed=1)(h)
    h = LUTDense(20, lut_rank=k, seed=2)(h)
    return keras.Model(inp, GroupSum(4, tau=2.0)(h))


def test_fixed_point_grid():
    x = np.array([-100, -1.03, 0.03, 0.97, 100], np.float32)
    q = fixed_point(x, 8, 3)
    assert q.min() == -8.0 and q.max() == 127 / 16
    assert np.all(np.abs(q[1:4] - x[1:4]) <= 1 / 32)


def test_quantile_thresholds_monotone():
    x = np.random.default_rng(0).normal(size=(1000, 3))
    t = quantile_thresholds(x, 7)
    assert t.shape == (3, 7) and np.all(np.diff(t, axis=1) > 0)


@pytest.mark.parametrize("learn", [False, True])
def test_thermometer_netlist_and_materialize_exact(learn):
    rng = np.random.default_rng(0)
    x = fixed_point(rng.normal(size=(600, 5)).astype(np.float32))
    y = rng.integers(0, 4, 600)
    m = _model(x, learn=learn)
    m.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    m.fit(x, y, epochs=1, batch_size=64, verbose=0)
    net = extract_netlist(m)
    assert net.input_encoder["type"] == "thermometer" and net.input_encoder["grid"]["bits"] == 8
    # Keras discrete path == netlist == materialised netlist
    keras_cls = np.argmax(m.predict(x, verbose=0), -1)
    _, cls = simulate(net, x)
    hw = materialize_inputs(net)
    _, cls_hw = simulate(hw, x)
    assert np.array_equal(cls, cls_hw)
    assert np.array_equal(keras_cls, cls)
    assert hw.input_encoder["type"] == "uniform" and hw.n_inputs == 5 * 8
    assert all(l.k in (0, 8) for l in hw.layers[0])
    # raw (unquantised) features: encoder quantises first, hardware sees words
    xr = rng.normal(size=(300, 5)).astype(np.float32) * 3
    _, a = simulate(net, xr)
    _, b = simulate(hw, xr)
    assert np.array_equal(a, b)


def test_materialize_roundtrip_file(tmp_path):
    rng = np.random.default_rng(1)
    x = fixed_point(rng.normal(size=(200, 4)).astype(np.float32))
    net = materialize_inputs(extract_netlist(_model(x)))
    p = tmp_path / "n.json"
    save_netlist(net, p)
    net2 = load_netlist(p)
    assert np.array_equal(simulate(net, x)[0], simulate(net2, x)[0])
    assert np.array_equal(encode_inputs(net, x), encode_inputs(net2, x))


def test_snapped_threshold_semantics():
    th = Thermometer(5, grid=(8, 3), thresholds=np.array([[-9.0, -0.01, 0.0, 0.06, 9.0]]))
    th.build((None, 1))
    xs = (np.arange(-128, 128) / 16.0).astype(np.float32)[:, None]
    got = np.asarray(th(xs, training=False))
    want = xs > th.thresholds.numpy()[0]           # continuous threshold on grid values
    assert np.array_equal(got.astype(bool), want)
