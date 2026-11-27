"""Tests for the Timestral example (examples/timestral/timestral_kd): data, student, cost, training, RTL formats."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "examples", "timestral"))

from qkeras.distill.losses import QuantileDistillLoss  # noqa: E402
from timestral_kd import cost as fc  # noqa: E402
from timestral_kd import data as dd  # noqa: E402
from timestral_kd import student as stu  # noqa: E402


def test_synthetic_shapes_and_coupling():
    s = dd.synthetic_process(3, 2000, n_drivers=2, n_targets=3, seed=1)
    assert len(s) == 3 and s[0].shape == (5, 2000) and np.isfinite(s[0]).all()


def test_split_and_windows_are_chronological():
    s = [np.arange(1000, dtype=np.float32)[None].repeat(2, 0)]
    tr, va, te = dd.chronological_split(s, (0.6, 0.2, 0.2))
    w_tr, w_te = dd.make_windows(tr, 64, 16, 16), dd.make_windows(te, 64, 16, 16)
    assert w_tr.y.max() < w_te.x.min()  # every test window starts after all training targets


def test_metrics():
    y = np.ones((4, 2, 8))
    q = np.ones((4, 2, 8, 3))
    x = np.cumsum(np.ones((4, 2, 32)), -1)
    m = dd.summarize(q, y, x, (0.1, 0.5, 0.9), 1)
    assert m["mase"] == 0 and m["wql"] == 0 and m["coverage_80"] == 1


@pytest.mark.parametrize("preset", ["float", "A8", "B4", "B3"])
def test_student_builds_and_counts(preset):
    cfg = stu.StudentConfig(d_model=16, num_heads=2, mlp_hidden=32, n_blocks=2, variate_every=2, patch_size=16,
                            ctx_steps=64, hor_steps=32, knots=(0.1, 0.5, 0.9), preset=preset,
                            act_fn="relu" if preset.startswith("B") else "silu")
    m = stu.build_student(cfg, 3, preset)
    assert m.count_params() == stu.count_ops(cfg, 3)["params"]
    y = np.asarray(m(np.random.default_rng(0).normal(size=(2, 3, 64)).astype("float32")))
    assert y.shape == (2, 3, 32, 3) and np.isfinite(y).all()
    assert np.all(np.diff(y, axis=-1) >= -1e-6)  # monotone quantiles
    cost = fc.estimate(cfg, 3, m if preset != "float" else None)
    assert cost["A"]["bram_bits"] > 0 and cost["B"]["cpe"] > 0


def test_csd_digits():
    assert fc._csd_digits(np.array([0, 1, 3, 7, 15, 5, 6])).tolist() == [0, 1, 2, 2, 2, 2, 2]


def test_loss_limits():
    knots = (0.1, 0.5, 0.9)
    y = np.zeros((2, 1, 4, 4), np.float32)
    q = np.zeros((2, 1, 4, 3), np.float32)
    assert float(QuantileDistillLoss(knots, 0.5)(y, q)) == 0.0
    y[..., 1:] = 1.0  # teacher differs, ground truth equal
    assert np.isclose(float(QuantileDistillLoss(knots, 0.0)(y, q)), 1.0)
    assert float(QuantileDistillLoss(knots, 1.0)(y, q)) == 0.0


def test_short_training_run(tmp_path):
    from timestral_kd import train

    rng = np.random.default_rng(0)
    s = dd.synthetic_process(6, 1500, seed=2)
    parts = dd.chronological_split(s, (0.6, 0.2, 0.2))
    bundle = {}
    for split, p in zip(("train", "val", "test"), parts):
        w = dd.make_windows(p, 128, 32, 32)
        tq = np.repeat(w.y[..., None], 9, -1) + rng.normal(0, 0.1, (*w.y.shape, 9))
        bundle.update({f"{split}_x": w.x, f"{split}_y": w.y, f"{split}_tq": np.sort(tq, -1).astype(np.float16)})
    np.savez(tmp_path / "toy.npz", **bundle)
    (tmp_path / "toy_info.json").write_text(json.dumps({"dataset": "toy", "season": 1, "variates": None}))
    cfg = {"student": {"name": "t", "d_model": 16, "num_heads": 2, "mlp_hidden": 16, "n_blocks": 1,
                       "ctx_steps": 64, "hor_steps": 32, "preset": "B4", "deployment": "B", "act_fn": "relu",
                       "gate_fn": "hard_sigmoid", "knots": [0.1, 0.5, 0.9]},
           "train": {"float_epochs": 1, "qat_epochs": 1, "batch_size": 32, "verbose": 0}}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    train.main(["--config", str(tmp_path / "cfg.json"), "--data", str(tmp_path / "toy.npz"), "--out", str(tmp_path / "run")])
    m = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert np.isfinite(m["qat"]["mase"]) and "B" in m["cost"]


def test_csd_count_uses_exact_kernel_scaling():
    """Regression: auto_po2 weights below 2**-1 must not be rounded away (sweep-v1 bug)."""
    k = np.array([[0.0625, -0.1875], [0.375, 0.0]])
    assert fc._exact_frac_bits(k) == 4
    (_, ki), = [("x", np.round(k * 2 ** fc._exact_frac_bits(k)))]
    assert np.count_nonzero(ki) == 3 and int(fc._csd_digits(ki).sum()) == 5   # 1 | -3 = -4+1 | 6 = 8-2


def test_rtl_quantizer_formats():
    from timestral_kd.hw import rtl
    assert rtl._parse_q("quantized_bits(7,3,alpha=1)") == (1, 3, 3)
    assert rtl._parse_q("quantized_bits(5,0,keep_negative=0,alpha=1)") == (0, 0, 5)
    assert rtl._parse_q("quantized_relu(7,3)") == (0, 3, 4)
    fm = rtl.layer_input_formats(None, stu.PRESETS["B3"])
    assert fm["A"] == (1, 3, 3) and fm["RELU"] == (0, 3, 4) and fm["WO"] == (1, 3, 8)


def test_fixedpoint_b3_is_the_training_preset():
    from timestral_kd.hw import fixedpoint as fp
    assert fp.B3 is stu.PRESETS["B3"]


def test_fixedpoint_references_match_qkeras_layers():
    from timestral_kd.hw import fixedpoint as fp
    worst = fp._validate()
    assert set(worst) == {"rmsnorm", "softmax", "quantiles", "rope", "quantiles_hw"}
