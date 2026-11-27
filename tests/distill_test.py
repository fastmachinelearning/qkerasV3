"""Tests for qkeras.distill (model-agnostic): presets, factories, weights, losses, cache, trainer, hw."""
import numpy as np
import pytest
import keras
from keras import layers

from qkeras.distill import factory as F
from qkeras.distill import losses as L
from qkeras.distill import presets as P
from qkeras.distill import teacher as T
from qkeras.distill import trainer as TR
from qkeras.distill import weights as W
from qkeras.distill.hw import fixedpoint as fxp
from qkeras.distill.hw import gatemate as gm
from qkeras.distill.layers import causal_mask


# ----------------------------------------------------------------------------- a tiny generic model
def tiny_transformer(f: F.LayerFactory, seq=8, d_in=4, d=8, heads=2, out=3, name="tiny"):
    """Pre-norm one-block transformer built only through the factory hooks."""
    hd = d // heads
    inp = keras.Input((seq, d_in))
    x = f.dense(d, True, "embed")(inp)
    n = f.norm(1e-6, "norm0")(x)
    qkv = f.dense(3 * d, False, "wQKV")(n)
    qkv = layers.Reshape((seq, 3, heads, hd))(qkv)
    q, k, v = (layers.Lambda(lambda t, i=i: t[:, :, i], output_shape=(seq, heads, hd))(qkv) for i in range(3))
    a = f.attention(causal_mask(seq), "sdpa")(q, k, v)
    g = f.act("sigmoid", "gate_act")(f.dense(d, False, "gate")(n))
    a = f.dense(d, False, "wO")(layers.Multiply()([a, g]))
    post = f.post("attn_post")
    a = post(a) if post is not None else a
    x = layers.Add()([x, a])
    h = f.act("silu", "mlp_act")(f.dense(2 * d, False, "w1")(f.norm(1e-6, "norm1")(x)))
    x = layers.Add()([x, f.dense(d, False, "w2")(h)])
    y = f.dense(out, True, "head")(layers.GlobalAveragePooling1D()(x))
    return keras.Model(inp, y, name=name)


def build(preset, pruner=None, act_map=None):
    return tiny_transformer(F.make_factory(preset, act_map, pruner=pruner))


# ----------------------------------------------------------------------------- presets
def test_resolve_fills_role_defaults():
    r = P.resolve("W8A14")
    assert set(r) == set(P.ROLES)
    assert r["gain"] == r["w"] and r["exp"] == r["prob"] and r["sig_out"] == r["lut_out"]
    assert P.resolve("B3")["sig_out"] == P.PRESETS["B3"]["sig_out"]


def test_register_preset_and_bits():
    P.register_preset("_test_W4", {"w": "quantized_bits(4,0,alpha=1)", "a": "quantized_bits(8,3)"}, overwrite=True)
    assert P.preset_bits("_test_W4") == {"w": 4, "a": 8}
    with pytest.raises(ValueError):
        P.register_preset("_bad", {"not_a_role": "x"})
    assert not P.is_quantized("float") and P.is_quantized("B3") and P.is_quantized({"w": None})
    assert P.bits_int("quantized_relu(7,3)") == (7, 3)


# ----------------------------------------------------------------------------- factory + weights
@pytest.mark.parametrize("preset,act_map", [("float", None), ("A8", None), ("B3", {"silu": "relu",
                                                                                   "sigmoid": "hard_sigmoid"}),
                                            ("W8A14", None)])
def test_same_layer_names_and_transfer(preset, act_map):
    fm = build("float", act_map=act_map)
    qm = build(preset, act_map=act_map)
    x = np.random.default_rng(0).normal(size=(2, 8, 4)).astype("float32")
    assert {l.name for l in fm.layers if l.weights} <= {l.name for l in qm.layers}
    n = W.transfer_weights(fm, qm)
    assert n == len([l for l in fm.layers if l.weights])
    y = np.asarray(qm(x))
    assert y.shape == (2, 3) and np.isfinite(y).all()
    if preset == "float":
        np.testing.assert_array_equal(y, np.asarray(fm(x)))


def test_quant_factory_options():
    from qkeras.pruning import ConstantSparsity, MagnitudePruner
    f = F.QuantFactory("B3", pruner=MagnitudePruner(ConstantSparsity(0.5)), prune_filter=lambda n: n != "head",
                       min_norm_eps=1e-4, skip_post=("attn",))
    assert f.post("attn_post") is None and f.post("other") is not None
    assert f.norm(1e-6, "n").epsilon == 1e-4
    with pytest.raises(ValueError):
        F.make_factory("float", pruner=object())


def test_dense_sparsity_counts_quantized_zeros():
    qm = build("B3")
    for l in qm.layers:
        if hasattr(l, "kernel"):
            l.kernel.assign(np.zeros(l.kernel.shape, "float32"))
    s = W.dense_sparsity(qm)
    assert s["zero_fraction"] == 1.0 and s["nonzero"] == 0 and s["total"] > 0


# ----------------------------------------------------------------------------- losses
def test_quantile_loss_limits():
    knots = (0.1, 0.5, 0.9)
    y = np.zeros((2, 4, 4), np.float32)
    q = np.zeros((2, 4, 3), np.float32)
    assert float(L.QuantileDistillLoss(knots, 0.5)(y, q)) == 0.0
    y[..., 1:] = 1.0
    assert np.isclose(float(L.QuantileDistillLoss(knots, 0.0)(y, q)), 1.0)
    assert float(L.QuantileDistillLoss(knots, 1.0)(y, q)) == 0.0


def test_logit_loss():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, 7)).astype("float32")
    labels = rng.integers(0, 7, 5)
    packed = L.pack_targets(labels, logits)
    ce = float(keras.ops.mean(keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)))
    assert np.isclose(float(L.LogitDistillLoss(alpha=1.0)(packed, logits)), ce, atol=1e-6)
    assert np.isclose(float(L.LogitDistillLoss(alpha=0.0, temperature=3.0)(packed, logits)), 0.0, atol=1e-6)
    assert float(L.LogitDistillLoss(alpha=0.0)(packed, logits + rng.normal(size=logits.shape))) > 0


def test_feature_loss():
    y = np.stack([np.zeros(6), np.ones(6)], -1).astype("float32")
    assert np.isclose(float(L.FeatureDistillLoss(0.5)(y, np.zeros(6, "float32"))), 0.5)


# ----------------------------------------------------------------------------- teacher cache
def test_teacher_cache(tmp_path):
    calls = []
    model = keras.Sequential([keras.Input((3,)), layers.Dense(2)])
    t = T.KerasTeacher(model, batch_size=4)

    def compute():
        calls.append(1)
        return t.predict(np.ones((10, 3), "float32"))

    key = T.cache_key(n=10, what="toy")
    a = T.cached(tmp_path, key, compute)
    b = T.cached(tmp_path, key, compute)
    assert a.shape == (10, 2) and len(calls) == 1
    np.testing.assert_array_equal(a, b)


# ----------------------------------------------------------------------------- trainer
@pytest.mark.parametrize("prune", [None, {"sparsity": 0.5, "ramp_frac": 0.5}])
def test_distill_two_phase(tmp_path, prune):
    keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(96, 8, 4)).astype("float32")
    truth = x.mean((1, 2))
    teacher = np.stack([truth - 0.5, truth, truth + 0.5], -1)
    y = L.pack_targets(truth, teacher)
    tcfg = {"batch_size": 32, "float_epochs": 1, "qat_epochs": 2, "verbose": 0}
    if prune:
        tcfg["prune"] = prune
    stages = []
    res = TR.distill(build, "B4", (x[:64], y[:64]), (x[64:], y[64:]), L.QuantileDistillLoss((0.1, 0.5, 0.9)),
                     [L.gt_pinball_metric((0.1, 0.5, 0.9))], tcfg, out=tmp_path, monitor="val_gt_pinball",
                     evaluate=lambda s, m: stages.append(s) or {"stage": s})
    assert stages == ["float", "qat_ptq", "qat"]
    assert (tmp_path / "float.weights.h5").exists() and (tmp_path / "qat.weights.h5").exists()
    assert res["qat"] is not None and "qat" in res["history"]
    if prune:
        assert res["prune"]["target"] == 0.5
        assert W.dense_sparsity(res["qat"])["zero_fraction"] >= 0.45


# ----------------------------------------------------------------------------- hw
def test_csd_and_frac_bits():
    assert gm.csd_digits(np.array([0, 1, 3, 7, 15, 5, 6])).tolist() == [0, 1, 2, 2, 2, 2, 2]
    assert gm.exact_frac_bits(np.array([0.0625, -0.1875, 0.375])) == 4


def test_dense_cpe_estimate_is_positive():
    res = gm.estimate_dense_cpe(build("B4"), in_bits=8, w_bits=4, use_da4ml=False)
    assert res["method"] == "csd" and res["dense_cpe"] > 0 and res["nonzero_weights"] > 0


def test_fixedpoint_formats():
    assert fxp.fmt("quantized_bits(7,3,alpha=1)")["f"] == 3
    assert fxp.fmt("quantized_bits(5,0,keep_negative=0)")["f"] == 5
    assert fxp.fmt("quantized_relu(7,3)")["f"] == 4
    assert fxp.qround(np.array([0.5, 1.5, -0.5])).tolist() == [1.0, 2.0, 0.0]


@pytest.mark.parametrize("preset", ["B3", "B4", "A8"])
def test_fixedpoint_rmsnorm_softmax_match_qkeras(preset):
    from qkeras.qtransformer import QRMSNorm, QSoftmax
    p = P.resolve(preset)
    rng = np.random.default_rng(0)
    luts = fxp.build_luts(p, 16, 1e-4)
    fa, fp_ = fxp.fmt(p["a"]), fxp.fmt(p["prob"])
    norm = QRMSNorm(epsilon=1e-4, **{f"{k}_quantizer": fxp.make_quantizer(p[r]) for k, r in
                                     (("ms", "ms"), ("rsqrt", "rsqrt"), ("gain", "gain"), ("output", "a"))})
    norm.build((None, 16))
    sm = QSoftmax(input_quantizer=fxp.make_quantizer(p["score"]), exp_quantizer=fxp.make_quantizer(p["exp"]),
                  inv_quantizer=fxp.make_quantizer(p["inv"]), output_quantizer=fxp.make_quantizer(p["prob"]))
    for _ in range(20):
        X = rng.integers(fa["lo"] // 2, fa["hi"] // 2, (1, 16))
        scale = rng.uniform(0.3, 1.7, 16)
        norm.scale.assign(scale.astype("float32"))
        ref = np.asarray(norm(fxp.to_value(X, fa))).reshape(16)
        got = fxp.to_value(fxp.rmsnorm_ref(X, fxp.to_code(scale, fxp.fmt(p["gain"])), luts, p), fa)
        assert np.abs(ref - got).max() <= 2.0 ** -fa["f"] + 1e-9
        s = rng.normal(0, 2, (1, 1, 1, 20)).astype("float32")
        ref = np.asarray(sm(s)).reshape(20)
        got = fxp.to_value(fxp.softmax_ref(s.reshape(20), luts, p), fp_)
        assert np.abs(ref - got).max() <= 2.0 ** -fp_["f"] + 1e-9
