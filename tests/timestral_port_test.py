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


def _series(b, t, seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(t)
    rows = [10 + rng.uniform(0.5, 5) * np.sin(2 * np.pi * x / rng.integers(12, 200)) + rng.normal(0, 0.3, t)
            for _ in range(b)]
    return np.asarray(rows, np.float32)


@pytest.mark.parametrize("context_patches,horizon_patches", [(16, 2), (8, 1)])
def test_keras_port_matches_reference(context_patches, horizon_patches):
    from timestral.loading import load_model
    from experimental.tsfm import keras_timestral as kt

    ref = load_model(CKPT).eval()
    spec = kt.TimestralSpec.from_json(os.path.join(CKPT, "config.json"))
    weights = safetensors.load_file(os.path.join(CKPT, "model.safetensors"))
    model = kt.build_timestral(spec, context_patches, horizon_patches)
    kt.load_timestral_weights(model, weights, spec)
    x = _series(3, context_patches * spec.patch_size)
    with torch.no_grad():
        y_ref = ref.predict_block(torch.tensor(x), torch.ones(x.shape, dtype=torch.bool),
                                  torch.arange(3), horizon_patches).numpy()
    y = kt.forecast(model, x, spec)
    assert y.shape == y_ref.shape
    np.testing.assert_allclose(y, y_ref, rtol=1e-4, atol=1e-4)
