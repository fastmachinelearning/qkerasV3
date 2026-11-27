"""Where the example reads inputs and writes outputs.

Everything generated lives under one work directory (default ``examples/timestral/runs``,
override with ``$TIMESTRAL_KD_RUNS``)::

    runs/data/<dataset>.npz            windows + teacher labels   (python -m timestral_kd data)
    runs/data/raw/                     downloaded source data
    runs/data/cache/                   teacher forecasts (re-used across bundles)
    runs/students/<dataset>/<name>/    trained student            (... train / sweep)
    runs/rtl/<dataset>/<name>/         Verilog + P&R per layer    (... export / pnr / report)
    runs/hw/                           Verilator builds of the full-model blocks

The Timestral-5M checkpoint defaults to ``<repo>/ts/Timestral-5M`` (``$TIMESTRAL_CKPT``).
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1]            # examples/timestral
REPO = EXAMPLE.parents[1]                                 # qkeras checkout
RUNS = Path(os.environ.get("TIMESTRAL_KD_RUNS", EXAMPLE / "runs"))
CKPT = Path(os.environ.get("TIMESTRAL_CKPT", REPO / "ts" / "Timestral-5M"))
CONFIGS = EXAMPLE / "configs"

KDD_URL = ("https://huggingface.co/datasets/autogluon/chronos_datasets/resolve/main/"
           "monash_kdd_cup_2018/train-00000-of-00001.parquet")


def data_dir() -> Path:
    return RUNS / "data"


def bundle(dataset: str) -> Path:
    return data_dir() / f"{dataset}.npz"


def student_dir(run: str) -> Path:
    """``"kdd/B_d12"`` -> ``runs/students/kdd/B_d12`` (an existing directory path is used as is)."""
    p = Path(run)
    return p if p.is_dir() else RUNS / "students" / run


def rtl_dir(run: str, digit_bits: int | None = None) -> Path:
    p = Path(run)
    rel = p.relative_to(RUNS / "students") if p.is_absolute() and p.is_relative_to(RUNS / "students") else p
    return RUNS / "rtl" / (f"{rel}_d{digit_bits}" if digit_bits else str(rel))


def kdd_parquet(download: bool = True) -> Path:
    """Local copy of the Monash KDD Cup 2018 parquet (downloaded once from Hugging Face)."""
    path = data_dir() / "raw" / "monash_kdd_cup_2018.parquet"
    if not path.exists() and download:
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {KDD_URL}\n  -> {path}", flush=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(KDD_URL, tmp)
        tmp.replace(path)
    return path
