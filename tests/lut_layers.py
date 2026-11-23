"""Tests for qkeras.logic (trainable k-LUT layers + HDL export).

Run:  pytest experimental/logic/test_lut_layers.py -q
HDL generation / verification tests live in the logicnet-fpga backend repository.
"""
import os
import shutil
import subprocess

import numpy as np
import pytest
import keras
from keras import ops

from qkeras.logic import (LUT, BinaryThreshold, GroupSum, LUTDense, Netlist,
                          extract_netlist, load_netlist, save_netlist,
                          set_lut_temperature, simulate)
from qkeras.logic.layers import _address_bits, walsh_matrix


KS = [1, 2, 3, 4, 5, 6]
MODES = [("light", "random", None), ("walsh", "random", None),
         ("light", "learnable", None), ("walsh", "learnable", 3)]


def _bits(n, d, seed=0):
    return (np.random.default_rng(seed).random((n, d)) > 0.5).astype("float32")


def _numpy_lut_layer(layer, x):
    idx, tab = layer.get_luts()
    addr = (x[:, idx] > 0.5).astype(int) @ (1 << np.arange(layer.lut_rank))
    return tab[np.arange(layer.units)[None], addr].astype("float32")


def _net(k, d=30, units=(40, 30), classes=3, name=None, init="random"):
    lay = [keras.Input((d,))]
    for i, u in enumerate(units):
        lay.append(LUTDense(u, lut_rank=k, init=init, seed=10 * k + i))
    if classes:
        lay.append(GroupSum(classes, tau=2.0))
    return keras.Sequential(lay, name=name or f"net_k{k}")


# ----------------------------------------------------------------- layers
def test_walsh_matrix_orthogonal():
    for k in KS:
        H = walsh_matrix(k)
        assert np.allclose(H @ H.T, (1 << k) * np.eye(1 << k))
        assert np.allclose(H[:, 0], 1)


@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("par,con,nc", MODES)
def test_discrete_forward_is_table_lookup(k, par, con, nc):
    x = _bits(64, 20)
    lay = LUTDense(16, lut_rank=k, parametrization=par, connections=con,
                   num_candidates=nc, init="random", seed=k)
    y = ops.convert_to_numpy(lay(x, training=False))
    assert y.shape == (64, 16)
    np.testing.assert_array_equal(y, _numpy_lut_layer(lay, x))


@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("par,con,nc", MODES)
def test_relaxed_converges_to_discrete(k, par, con, nc):
    x = _bits(64, 20, seed=1)
    lay = LUTDense(16, lut_rank=k, parametrization=par, connections=con,
                   num_candidates=nc, init="random", seed=k)
    yd = ops.convert_to_numpy(lay(x, training=False))
    lay.temperature.assign(1e-9)
    yr = ops.convert_to_numpy(lay(x, training=True))
    assert np.abs(yr - yd).max() < 1e-3


@pytest.mark.parametrize("par", ["light", "walsh"])
def test_residual_init_is_identity_of_pin0(par):
    x = _bits(256, 40)
    lay = LUTDense(64, lut_rank=6, parametrization=par, seed=0)
    y = ops.convert_to_numpy(lay(x, training=False))
    idx, _ = lay.get_luts()
    assert (y == x[:, idx[:, 0]]).mean() > 0.97


def test_random_connections_distinct_and_cover_inputs():
    lay = LUTDense(200, lut_rank=6, seed=0)
    lay.build((None, 784))
    idx, _ = lay.get_luts()
    assert all(len(set(r)) == 6 for r in idx)
    assert len(np.unique(idx)) == 784


@pytest.mark.parametrize("par,con,nc", MODES)
def test_training_updates_all_trainable_weights(par, con, nc):
    x = _bits(128, 24)
    y = np.random.default_rng(0).integers(0, 2, 128)
    m = keras.Sequential([keras.Input((24,)),
                          LUTDense(20, lut_rank=4, parametrization=par,
                                   connections=con, num_candidates=nc, seed=0),
                          GroupSum(2, tau=2.0)])
    m.compile(keras.optimizers.Adam(0.05),
              keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    before = [ops.convert_to_numpy(w).copy() for w in m.trainable_weights]
    m.fit(x, y, epochs=2, batch_size=32, verbose=0)
    for b, w in zip(before, m.trainable_weights):
        assert not np.allclose(b, ops.convert_to_numpy(w)), w.name


def test_save_load_roundtrip(tmp_path):
    m = keras.Sequential([keras.Input((30,)), BinaryThreshold(0.5),
                          LUTDense(30, lut_rank=3, connections="learnable",
                                   num_candidates=4, seed=1),
                          LUTDense(20, lut_rank=5, parametrization="walsh", seed=2),
                          GroupSum(4, tau=3.0)])
    set_lut_temperature(m, 0.3)
    p = str(tmp_path / "m.keras")
    m.save(p)
    m2 = keras.models.load_model(p)
    x = np.random.default_rng(0).random((50, 30)).astype("float32")
    np.testing.assert_array_equal(m.predict(x, verbose=0), m2.predict(x, verbose=0))
    assert float(ops.convert_to_numpy(m2.layers[1].temperature)) == pytest.approx(0.3)


# ----------------------------------------------------------------- netlist
@pytest.mark.parametrize("k", [2, 3, 4, 6])
def test_netlist_matches_keras(k):
    m = _net(k)
    x = _bits(2000, 30, seed=k)
    keras_scores = np.rint(m.predict(x, verbose=0) * 2.0)
    sc, cls = simulate(extract_netlist(m), x)
    np.testing.assert_array_equal(sc, keras_scores)


def test_netlist_file_roundtrip(tmp_path):
    m = _net(5, units=(24, 12))
    net = extract_netlist(m)
    p = str(tmp_path / "net.json")
    save_netlist(net, p)
    net2 = load_netlist(p)
    x = _bits(500, 30)
    np.testing.assert_array_equal(simulate(net, x)[0], simulate(net2, x)[0])
    assert net2.meta["tau"] == pytest.approx(2.0) and net2.num_classes == 3


def test_netlist_file_readable_by_logicnet(tmp_path):
    ln = pytest.importorskip("logicnet.ir")
    m = _net(4)
    p = str(tmp_path / "net.json")
    save_netlist(extract_netlist(m), p)
    x = _bits(500, 30)
    np.testing.assert_array_equal(simulate(extract_netlist(m), x)[0], ln.simulate(ln.load_netlist(p), x)[0])


def test_extract_rejects_unsupported_layers():
    m = keras.Sequential([keras.Input((10,)), keras.layers.Dense(4)])
    with pytest.raises(ValueError):
        extract_netlist(m)


def test_candidate_routing_scatter_equals_gather():
    """Dense scatter-matmul routing == explicit gather, duplicates summed."""
    x = np.random.default_rng(0).random((32, 10)).astype("float32")
    lay = LUTDense(12, lut_rank=3, connections="learnable", num_candidates=4,
                   routing_sampling="soft", seed=0)
    lay.build((None, 10))
    cand = ops.convert_to_numpy(lay.candidates).copy()
    cand[:, :, 1] = cand[:, :, 0]
    lay.candidates.assign(cand)
    w = ops.convert_to_numpy(ops.softmax(lay.routing_logits / lay.temperature, axis=-1))
    ref = (x[:, cand] * w[None]).sum(-1)
    np.testing.assert_allclose(ops.convert_to_numpy(lay._route_train(x)), ref, atol=1e-5)
