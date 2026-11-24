"""Parity of the Keras 3 Timestral port with the released PyTorch reference.

Skipped unless the reference package ``timestral`` and a checkpoint directory are
available (set ``TIMESTRAL_CKPT`` to a folder with config.json + model.safetensors).
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
timestral = pytest.importorskip("timestral")
safetensors = pytest.importorskip("safetensors.numpy")

CKPT = os.environ.get("TIMESTRAL_CKPT", "")
pytestmark = pytest.mark.skipif(not os.path.isfile(os.path.join(CKPT, "model.safetensors")),
                                reason="set TIMESTRAL_CKPT to a Timestral checkpoint directory")


@pytest.fixture(scope="module")
def setup():
    from timestral.loading import load_model
    from experimental.tsfm import keras_timestral as kt

    ref = load_model(CKPT).eval()
    spec = kt.TimestralSpec.from_json(os.path.join(CKPT, "config.json"))
    weights = safetensors.load_file(os.path.join(CKPT, "model.safetensors"))
    return kt, ref, spec, weights


def _series(b, t, seed=0, level=10.0, amp=(0.5, 5.0), noise=0.3):
    rng = np.random.default_rng(seed)
    x = np.arange(t)
    rows = [level + rng.uniform(*amp) * np.sin(2 * np.pi * x / rng.integers(12, 200)) + rng.normal(0, noise, t)
            for _ in range(b)]
    return np.asarray(rows, np.float32)


def _reference(ref, x2d, group_ids, horizon_patches):
    with torch.no_grad():
        return ref.predict_block(torch.tensor(x2d), torch.ones(x2d.shape, dtype=torch.bool),
                                 torch.tensor(group_ids), horizon_patches).numpy()


@pytest.mark.parametrize("context_patches,horizon_patches", [(16, 2), (8, 1)])
def test_univariate_matches_reference(setup, context_patches, horizon_patches):
    kt, ref, spec, weights = setup
    model = kt.build_timestral(spec, context_patches, horizon_patches)
    kt.load_timestral_weights(model, weights, spec)
    x = _series(3, context_patches * spec.patch_size)
    y_ref = _reference(ref, x, np.arange(3), horizon_patches)
    y = kt.forecast(model, x, spec)
    assert y.shape == y_ref.shape
    np.testing.assert_allclose(y, y_ref, rtol=1e-4, atol=1e-4)


def test_near_constant_high_level_signal(setup):
    """float64 causal scaling: a quiet signal at level ~1000 must not collapse its scale."""
    kt, ref, spec, weights = setup
    model = kt.build_timestral(spec, 16, 2)
    kt.load_timestral_weights(model, weights, spec)
    x = _series(2, 16 * spec.patch_size, seed=3, level=980.0, amp=(0.0, 0.01), noise=0.05)
    np.testing.assert_allclose(kt.forecast(model, x, spec), _reference(ref, x, np.arange(2), 2), rtol=1e-5, atol=2e-3)


def test_multivariate_matches_joint_reference(setup):
    """V variates of one series attend each other (variate attention) like group_ids in the reference."""
    kt, ref, spec, weights = setup
    b, v, lp, hp = 2, 3, 16, 2
    x = _series(b * v, lp * spec.patch_size, seed=7).reshape(b, v, -1)
    model = kt.build_timestral_multivariate(spec, v, lp, hp)
    kt.load_timestral_weights(model, weights, spec)
    y = kt.forecast_mv(model, x, spec)
    y_ref = _reference(ref, x.reshape(b * v, -1), np.repeat(np.arange(b), v), hp).reshape(y.shape)
    np.testing.assert_allclose(y, y_ref, rtol=1e-3, atol=1e-3)
    # the joint forecast must differ from independent per-variate forecasts
    y_ind = _reference(ref, x.reshape(b * v, -1), np.arange(b * v), hp).reshape(y.shape)
    assert np.max(np.abs(y_ind - y_ref)) > 1e-3


def test_quantile_widths_follow_knot_gaps(setup):
    kt, _, spec, _ = setup
    w = kt.quantile_increment_widths(spec.quantile_knots, spec.quantile_width_factor)
    m = spec.quantile_knots.index(0.5)
    assert w[m] == 0 and np.isclose(w[m + 1], spec.quantile_width_factor * 0.05)
    assert np.isclose(w[-1], spec.quantile_width_factor * (0.99 - 0.95))
