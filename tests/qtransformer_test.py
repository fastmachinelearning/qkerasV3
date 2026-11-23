"""Tests for qkeras.qtransformer (quantised Transformer building blocks)."""

import math

import keras
import numpy as np
import pytest
from keras import layers, ops

from qkeras import QDense
from qkeras.qtransformer import (
    QDotProductAttention,
    QEinsumDense,
    QRMSNorm,
    QRotary,
    QSoftmax,
    QUnary,
    rotary_matrices,
)
from qkeras.utils import quantized_model_from_json

RNG = np.random.default_rng(1234)


def _np(x):
    return np.asarray(keras.ops.convert_to_numpy(x))


def _on_grid(x, frac_bits, tol=1e-6):
    y = np.asarray(x) * 2.0**frac_bits
    return np.all(np.abs(y - np.round(y)) < tol)


# --------------------------------------------------------------------- float limits
def test_qeinsumdense_float_matches_einsumdense():
    x = RNG.normal(size=(2, 5, 8)).astype("float32")
    ref = layers.EinsumDense("bsd,de->bse", output_shape=(None, 6), bias_axes="e")
    q = QEinsumDense("bsd,de->bse", output_shape=(None, 6), bias_axes="e")
    ref.build(x.shape)
    q.build(x.shape)
    q.set_weights(ref.get_weights())
    np.testing.assert_allclose(_np(q(x)), _np(ref(x)), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("fn,ref", [
    ("sigmoid", lambda x: 1 / (1 + np.exp(-x))),
    ("silu", lambda x: x / (1 + np.exp(-x))),
    ("softplus", lambda x: np.log1p(np.exp(x))),
    ("exp", np.exp),
])
def test_qunary_float(fn, ref):
    x = RNG.normal(size=(3, 7)).astype("float32")
    np.testing.assert_allclose(_np(QUnary(fn)(x)), ref(x), rtol=1e-5, atol=1e-6)


def test_qrmsnorm_float():
    x = RNG.normal(size=(2, 4, 16)).astype("float32")
    lyr = QRMSNorm(epsilon=1e-6)
    lyr.build(x.shape)
    g = RNG.normal(size=(16,)).astype("float32")
    lyr.set_weights([g])
    ref = x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6) * g
    np.testing.assert_allclose(_np(lyr(x)), ref, rtol=1e-5, atol=1e-5)


def test_qsoftmax_float_and_mask():
    x = RNG.normal(size=(2, 3, 4, 4)).astype("float32")
    mask = np.tril(np.ones((4, 4), bool))[None, None]
    y = _np(QSoftmax()(x, mask=ops.convert_to_tensor(mask)))
    e = np.where(mask, np.exp(x - x.max(-1, keepdims=True)), 0)
    ref = e / e.sum(-1, keepdims=True)
    np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-6)
    assert np.all(y[..., ~mask[0, 0]] == 0)


def test_qrotary_matches_rotate_half():
    s, h, d, rd = 6, 2, 8, 4
    x = RNG.normal(size=(1, s, h, d)).astype("float32")
    inv = np.zeros(d // 2)
    inv[: rd // 2] = 1.0 / (10000.0 ** (np.arange(0, rd, 2) / rd))
    emb = np.concatenate([np.arange(s)[:, None] * inv] * 2, -1)
    cos, sin = np.cos(emb)[None, :, None], np.sin(emb)[None, :, None]
    rot = np.concatenate([-x[..., d // 2:], x[..., : d // 2]], -1)
    ref = x * cos + rot * sin
    np.testing.assert_allclose(_np(QRotary(s, d, rd)(x)), ref, rtol=1e-5, atol=1e-5)
    R = rotary_matrices(s, d, rd)
    assert R.shape == (s, d, d)
    np.testing.assert_allclose(np.einsum("sde,sfe->sdf", R, R), np.broadcast_to(np.eye(d), (s, d, d)), atol=1e-5)


def test_qdotproductattention_float():
    b, s, h, d = 2, 5, 3, 4
    q, k, v = (RNG.normal(size=(b, s, h, d)).astype("float32") for _ in range(3))
    mask = np.tril(np.ones((s, s), bool))
    y = _np(QDotProductAttention(attention_mask=mask)(q, k, v))
    sc = np.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(d)
    sc = np.where(mask, sc, -np.inf)
    a = np.exp(sc - sc.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    ref = np.einsum("bhqk,bkhd->bqhd", a, v).reshape(b, s, h * d)
    np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------- quantisation
def test_qeinsumdense_kernel_on_grid():
    lyr = QEinsumDense("bd,de->be", output_shape=4, kernel_quantizer="quantized_bits(4,0,alpha=1)")
    lyr.build((None, 8))
    qk = _np(lyr.kernel_quantizer_internal(lyr.kernel))
    assert _on_grid(qk, 3)


def test_quantised_outputs_on_grid():
    x = RNG.normal(size=(2, 4, 4)).astype("float32")
    y = _np(QUnary("sigmoid", "quantized_bits(8,3,alpha=1)", "quantized_bits(8,0,alpha=1)")(x))
    assert _on_grid(y, 7)
    p = _np(QSoftmax(input_quantizer="quantized_bits(10,4,alpha=1)", exp_quantizer="quantized_bits(10,1,alpha=1)",
                     inv_quantizer="quantized_bits(12,4,alpha=1)", output_quantizer="quantized_bits(9,1,alpha=1)")(x))
    assert _on_grid(p, 8)
    assert np.allclose(p.sum(-1), 1.0, atol=0.05)
    n = QRMSNorm(output_quantizer="quantized_bits(8,3,alpha=1)", rsqrt_quantizer="quantized_bits(12,4,alpha=1)")
    assert _on_grid(_np(n(x)), 4)


def test_quantised_attention_close_to_float():
    b, s, h, d = 2, 8, 2, 8
    q, k, v = (RNG.normal(size=(b, s, h, d)).astype("float32") for _ in range(3))
    mask = np.tril(np.ones((s, s), bool))
    f = _np(QDotProductAttention(attention_mask=mask)(q, k, v))
    qa = QDotProductAttention(attention_mask=mask, score_quantizer="quantized_bits(12,4,alpha=1)",
                              softmax_exp_quantizer="quantized_bits(12,1,alpha=1)",
                              softmax_inv_quantizer="quantized_bits(14,4,alpha=1)",
                              prob_quantizer="quantized_bits(10,1,alpha=1)",
                              output_quantizer="quantized_bits(12,4,alpha=1)")
    y = _np(qa(q, k, v))
    assert np.max(np.abs(y - f)) < 0.05


# --------------------------------------------------------------------- training/serialisation
def test_ste_gradients_flow():
    import tensorflow as tf

    x = tf.constant(RNG.normal(size=(2, 4, 8)).astype("float32"))
    lyr = QEinsumDense("bsd,de->bse", output_shape=(None, 8), kernel_quantizer="quantized_bits(4,0,alpha=1)")
    norm = QRMSNorm(gain_quantizer="quantized_bits(8,2,alpha=1)", output_quantizer="quantized_bits(8,3,alpha=1)")
    act = QUnary("silu", "quantized_bits(8,3,alpha=1)", "quantized_bits(8,3,alpha=1)")
    with tf.GradientTape() as tape:
        y = act(norm(lyr(x)))
        loss = tf.reduce_sum(y**2)
    grads = tape.gradient(loss, lyr.trainable_weights + norm.trainable_weights)
    assert all(g is not None and float(tf.reduce_sum(tf.abs(g))) > 0 for g in grads)


def _tiny_transformer(qb="quantized_bits(8,3,alpha=1)", s=6, d=8, h=2):
    inp = keras.Input((s, d))
    x = QRMSNorm(ms_quantizer=qb, rsqrt_quantizer=qb, gain_quantizer=qb, output_quantizer=qb, name="norm")(inp)
    q = QEinsumDense("bsd,dhe->bshe", (s, h, d // h), kernel_quantizer=qb, name="wq")(x)
    k = QEinsumDense("bsd,dhe->bshe", (s, h, d // h), kernel_quantizer=qb, name="wk")(x)
    v = QEinsumDense("bsd,dhe->bshe", (s, h, d // h), kernel_quantizer=qb, name="wv")(x)
    rope = QRotary(s, d // h, table_quantizer=qb, output_quantizer=qb, name="rope")
    q, k = rope(q), rope(k)
    a = QDotProductAttention(attention_mask=np.tril(np.ones((s, s), bool)), score_quantizer=qb, softmax_exp_quantizer=qb,
                             softmax_inv_quantizer=qb, prob_quantizer=qb, output_quantizer=qb, name="attn")(q, k, v)
    g = QUnary("sigmoid", qb, qb, name="gate")(QDense(d, kernel_quantizer=qb, use_bias=False, name="wg")(x))
    o = layers.Multiply()([a, g])
    o = QDense(d, kernel_quantizer=qb, bias_quantizer=qb, name="wo")(o)
    return keras.Model(inp, layers.Add()([inp, o]))


def test_json_roundtrip():
    m = _tiny_transformer()
    x = RNG.normal(size=(3, 6, 8)).astype("float32")
    y = _np(m(x))
    m2 = quantized_model_from_json(m.to_json())
    m2.set_weights(m.get_weights())
    np.testing.assert_allclose(_np(m2(x)), y, rtol=1e-6, atol=1e-6)


def test_fit_one_step():
    m = _tiny_transformer()
    m.compile(optimizer="adam", loss="mse")
    x = RNG.normal(size=(8, 6, 8)).astype("float32")
    hist = m.fit(x, x, epochs=1, verbose=0)
    assert np.isfinite(hist.history["loss"][0])
