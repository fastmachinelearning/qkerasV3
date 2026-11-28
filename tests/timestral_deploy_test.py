"""Whole-student GateMate deployment (examples/timestral/timestral_kd: hw/fullmodel, deploy, board)."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "examples", "timestral"))
sys.path.insert(0, HERE)

from timestral_distill_test import _toy_bundle  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from timestral_kd import __main__ as cli
    from timestral_kd import paths
    root = tmp_path_factory.mktemp("runs")
    old = paths.RUNS
    paths.RUNS = root
    _toy_bundle(root / "data")
    cfg = {"student": {"name": "tiny", "d_model": 8, "num_heads": 2, "mlp_hidden": 8, "n_blocks": 1,
                       "variate_every": 0, "embed_hidden": 8, "head_hidden": 8, "ctx_steps": 64, "hor_steps": 32,
                       "preset": "B3", "deployment": "B", "act_fn": "relu", "gate_fn": "hard_sigmoid",
                       "knots": [0.1, 0.5, 0.9]},
           "train": {"float_epochs": 1, "qat_epochs": 1, "batch_size": 32, "verbose": 0}}
    (root / "tiny.json").write_text(json.dumps(cfg))
    cli.main(["train", str(root / "tiny.json"), "--dataset", "toy"])
    yield root
    paths.RUNS = old


def test_deploy_project(tiny):
    from timestral_kd import deploy
    out = tiny / "deploy" / "toy" / "tiny"
    s = deploy.build("toy/tiny", out, n_selftest=2)
    nn = json.loads((out / "nn.json").read_text())
    assert nn["interface"] == "memory" and nn["top"] == "timestral_engine"
    m = nn["mem"]
    assert m["runs"] == s["variates"] and m["in_len"] == s["in_len"] == 64 and m["out_len"] == s["out_len"]
    assert nn["in_bits"] == m["runs"] * m["in_len"] * m["in_word_bits"]
    for f in ["rtl/engine.v"] + nn["memfiles"]:
        assert (out / f).stat().st_size > 0
    h = (out / "firmware" / "selftest.h").read_text()
    assert "#define SELFTEST_N 2" in h and "selftest_out" in h
    assert (out / "codec.npz").exists() and (out / "README.md").exists()


def test_board_codec_roundtrip_matches_integer_program(tiny):
    from timestral_kd import board, train
    from timestral_kd import student as stu
    from timestral_kd.hw.fullmodel import program as pg
    codec = board.Codec(tiny / "deploy" / "toy" / "tiny" / "codec.npz")
    cfg = stu.StudentConfig.from_dict(codec.prog.cfg)
    b, _ = train.load_bundle(tiny / "data" / "toy.npz")
    te = train.prepare_split(b, "test", cfg)
    for i in range(3):
        hx, loc, sc = codec.encode(te["x"][i])
        assert len(hx) == te["x"].shape[1] * 64 * 3
        q = codec.decode(codec.offline(hx), loc, sc)
        codes = pg.quantize_input(te["xs"][i:i + 1], codec.prog.hw)
        raw = pg.forward(codec.prog, codes)
        z = pg.quantiles_from_raw(raw.reshape(-1, *q.shape[1:]), codec.prog)
        from timestral_kd import teacher as tch
        ref = tch.from_student_space(z[None], loc[None], sc[None])[0]
        np.testing.assert_allclose(q, ref, rtol=0, atol=1e-6)
        assert np.all(np.diff(q, axis=-1) >= 0)                            # monotone quantiles


def test_run_name_and_studio_url(tmp_path, monkeypatch):
    from timestral_kd import deploy, paths
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    inside = tmp_path / "runs" / "students" / "kdd" / "s1"
    outside = tmp_path / "old" / "runs" / "students" / "kdd" / "s2"
    inside.mkdir(parents=True)
    outside.mkdir(parents=True)
    assert deploy.run_name("kdd/s0") == "kdd/s0"
    assert deploy.run_name(str(inside)) == "kdd/s1"
    assert deploy.run_name(str(outside)) == "kdd/s2"
    assert deploy.studio_url("studio.example.org/") == "https://studio.example.org"
    assert deploy.studio_url("http://localhost:8000") == "http://localhost:8000"
