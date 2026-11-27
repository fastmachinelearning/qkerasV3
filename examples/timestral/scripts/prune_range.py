import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # examples/timestral

from pathlib import Path

import keras
from keras import ops
import numpy as np
from qkeras.utils import print_model_sparsity
from qkeras.pruning import (
    MagnitudePruner,
    PolynomialDecay,
    UpdatePruningStep,
    apply_pruning_masks,
)
from qkeras.range_analysis import RangeAnalysisCallback, optimize_bitwidths
from safetensors.numpy import load_file

from timestral_kd.legacy import legacy_factory
from timestral_kd.model import (
    TimestralSpec,
    build_timestral,
    forecast,
    load_timestral_weights,
    scale_context,
)

# ----------------------------------------------------------------------------- configuration
MODEL_DIR = Path(os.environ.get("TIMESTRAL_CKPT", Path(__file__).resolve().parents[3] / "ts" / "Timestral-5M"))
CONTEXT_PATCHES = 16
HORIZON_PATCHES = 3


def load_weights_into_prunable_model(
    model: keras.Model, weights: dict[str, np.ndarray], spec: TimestralSpec
) -> None:
    """Load safetensors weights directly into kernel/bias/scale variables, ignoring pruning masks."""

    def setw(name, kernel_or_scale, bias=None):
        lyr = model.get_layer(name)
        if hasattr(lyr, "kernel"):
            assert lyr.kernel.shape == kernel_or_scale.shape, (
                name,
                lyr.kernel.shape,
                kernel_or_scale.shape,
            )
            lyr.kernel.assign(np.asarray(kernel_or_scale, np.float32))
            if bias is not None:
                assert lyr.bias is not None and lyr.bias.shape == bias.shape
                lyr.bias.assign(np.asarray(bias, np.float32))
        else:
            target = lyr.scale if hasattr(lyr, "scale") else lyr.token
            assert target.shape == kernel_or_scale.shape, (
                name,
                target.shape,
                kernel_or_scale.shape,
            )
            target.assign(np.asarray(kernel_or_scale, np.float32))

    def lin(name, key, bias=True):
        wt = weights[f"{key}.weight"].T
        b = weights[f"{key}.bias"] if (bias and f"{key}.bias" in weights) else None
        setw(name, wt, b)

    rp = "representation.patch_proj"
    for part in ("hidden", "output", "residual"):
        lin(f"embed_{part}", f"{rp}.{part}_layer")
    setw("forecast_query", weights["representation.forecast_query"])

    for i, kind in enumerate(spec.block_schedule):
        tp = f"transformer.layers.{i}"
        nax = 1 if kind == "time" else 2
        for j in range(nax):
            setw(f"b{i}_attn_norm{j}", weights[f"{tp}.attn_norms.{j}.scale"])
            lin(f"b{i}_attn{j}_wQKV", f"{tp}.attns.{j}.wQKV", bias=False)
            lin(f"b{i}_attn{j}_gate", f"{tp}.attns.{j}._gate", bias=False)
            lin(f"b{i}_attn{j}_wO", f"{tp}.attns.{j}.wO", bias=False)
        setw(f"b{i}_mlp_norm", weights[f"{tp}.mlp_norm.scale"])
        lin(f"b{i}_w12", f"{tp}.mlp.w12", bias=False)
        lin(f"b{i}_w3", f"{tp}.mlp.w3", bias=False)

    setw("out_norm", weights["transformer.out_norm.scale"])
    for part in ("hidden", "output", "residual"):
        lin(f"head_{part}", f"head.projection.{part}_layer", bias=False)


def make_pinball_loss(quantile_knots: tuple[float, ...]):
    """Quantile (pinball) loss in normalized model output space (B, H, Q)."""
    q = ops.convert_to_tensor(np.asarray(quantile_knots, dtype=np.float32)[None, None, :])

    def pinball_loss(y_true, y_pred):
        err = y_true - y_pred
        return ops.mean(ops.maximum(q * err, (q - 1.0) * err))

    return pinball_loss


def generate_normalized_dataset(
    spec: TimestralSpec, ctx_len: int, hor_len: int, n_series: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (scaled_context, scaled_horizon) pairs in the anchor patch's frame."""
    rng = np.random.default_rng(seed)
    total_len = ctx_len + hor_len
    t = np.arange(total_len, dtype=np.float32)[None, :]

    slopes = rng.uniform(-0.06, 0.06, size=(n_series, 1)).astype(np.float32)
    offsets = rng.uniform(-100.0, 100.0, size=(n_series, 1)).astype(np.float32)
    periods = rng.uniform(16.0, 96.0, size=(n_series, 1)).astype(np.float32)
    amps = rng.uniform(1.0, 15.0, size=(n_series, 1)).astype(np.float32)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(n_series, 1)).astype(np.float32)
    noise_stds = rng.uniform(0.05, 1.0, size=(n_series, 1)).astype(np.float32)

    raw = offsets + slopes * t + amps * np.sin(2.0 * np.pi * t / periods + phases)
    raw += rng.normal(0.0, noise_stds, size=(n_series, total_len)).astype(np.float32)

    raw_ctx = raw[:, :ctx_len]
    raw_hor = raw[:, ctx_len:]

    s_ctx, anchor_loc, anchor_scale = scale_context(
        raw_ctx, spec.patch_size, spec.minimum_scale, spec.use_asinh
    )
    s_hor = (raw_hor - anchor_loc) / np.maximum(anchor_scale, np.finfo(np.float32).eps)
    if spec.use_asinh:
        s_hor = np.arcsinh(s_hor)
    return s_ctx.astype(np.float32), s_hor[..., None].astype(np.float32)


def generate_test_batch(ctx_len: int, hor_len: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate the 4-row raw synthetic benchmark batch."""
    rng = np.random.default_rng(seed)
    t = np.arange(ctx_len + hor_len, dtype=np.float32)[None, :]
    full = (
        np.array([[10.0], [50.0], [-5.0], [100.0]], np.float32)
        + np.array([[0.02], [-0.015], [0.04], [0.005]], np.float32) * t
        + np.array([[3.0], [8.0], [1.5], [12.0]], np.float32)
        * np.sin(
            2.0 * np.pi * t / np.array([[32.0], [64.0], [24.0], [48.0]], np.float32)
            + np.array([[0.0], [1.1], [2.3], [0.5]], np.float32)
        )
        + rng.normal(
            0.0, np.array([[0.2], [0.5], [0.1], [0.8]], np.float32), size=(4, ctx_len + hor_len)
        ).astype(np.float32)
    )
    return full[:, :ctx_len], full[:, ctx_len:]


def compute_effective_size_mb(model: keras.Model) -> tuple[int, int, float, float]:
    """Return (total_params, nonzero_params, float32_mb, nonzero_quantized_mb)."""
    total_params = 0
    nonzero_params = 0
    nonzero_bits = 0

    for layer in model.layers:
        q_map = {}
        if hasattr(layer, "kernel"):
            q_map[id(layer.kernel)] = getattr(layer, "kernel_quantizer_internal", None)
        if getattr(layer, "bias", None) is not None:
            q_map[id(layer.bias)] = getattr(layer, "bias_quantizer_internal", None)
        if hasattr(layer, "scale"):
            q_map[id(layer.scale)] = getattr(layer, "gain_quantizer_internal", None)

        # Only count trainable model parameters (exclude non-trainable pruning mask sub-layers)
        for w in layer.trainable_weights:
            arr = w.numpy()
            n = int(arr.size)
            nz = int(np.count_nonzero(arr))
            q = q_map.get(id(w))
            bits = int(getattr(q, "bits", 32)) if q is not None else 32
            total_params += n
            nonzero_params += nz
            nonzero_bits += nz * bits

    float_mb = (total_params * 4) / 1024**2
    effective_mb = (nonzero_bits / 8.0) / 1024**2
    return total_params, nonzero_params, float_mb, effective_mb


def main() -> None:
    spec = TimestralSpec.from_json(MODEL_DIR / "config.json")
    weights = load_file(MODEL_DIR / "model.safetensors")
    ctx_len = CONTEXT_PATCHES * spec.patch_size
    hor_len = HORIZON_PATCHES * spec.patch_size
    p50_idx = spec.quantile_knots.index(0.5)

    # 1. Float Baseline evaluation
    test_ctx, test_gt = generate_test_batch(ctx_len, hor_len)
    float_model = build_timestral(spec, CONTEXT_PATCHES, HORIZON_PATCHES, name="timestral_float")
    load_timestral_weights(float_model, weights, spec)
    float_pred = forecast(float_model, test_ctx, spec)[:, :, p50_idx]
    float_mae = float(np.mean(np.abs(float_pred - test_gt)))

    # 2. Prepare synthetic fine-tuning & calibration datasets
    x_train, y_train = generate_normalized_dataset(spec, ctx_len, hor_len, n_series=512, seed=1)
    x_val, y_val = generate_normalized_dataset(spec, ctx_len, hor_len, n_series=64, seed=2)

    batch_size = 32
    epochs = 10
    total_steps = (len(x_train) // batch_size) * epochs

    # 3. Configure PolynomialDecay pruner (0% -> 50% sparsity)
    pruner = MagnitudePruner(
        PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.50,
            begin_step=0,
            end_step=int(0.8 * total_steps),
            frequency=20,
        )
    )

    # 4. Build prunable + quantized model & load pretrained weights directly
    factory = legacy_factory(pruner=pruner, prune_embed_and_head=False)
    model = build_timestral(
        spec, CONTEXT_PATCHES, HORIZON_PATCHES, factory=factory, name="timestral_pruned_q"
    )
    load_weights_into_prunable_model(model, weights, spec)

    # Initial range calibration before fine-tuning
    optimize_bitwidths(model, data=x_val, policy="keep_fractional", headroom=1, verbose=0)

    # 5. Joint Pruning + QAT fine-tuning with RangeAnalysisCallback
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4),
        loss=make_pinball_loss(spec.quantile_knots),
    )

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            UpdatePruningStep(verbose=1),
            RangeAnalysisCallback(
                calibration_data=x_val,
                frequency=2,
                policy="keep_fractional",
                headroom=1,
                patience=2,
            ),
        ],
    )

    # 6. Bake pruning masks into kernels in-place (avoids clone_model on Lambda layers)
    apply_pruning_masks(model)
    print_model_sparsity(model)

    # 7. Final post-training range tightening on the exact sparse model
    optimize_bitwidths(model, data=x_val, policy="keep_fractional", headroom=1, min_bits=4, verbose=1)

    # 8. Evaluate final compressed & pruned model
    compressed_pred = forecast(model, test_ctx, spec)[:, :, p50_idx]
    compressed_mae = float(np.mean(np.abs(compressed_pred - test_gt)))
    drift_mae = float(np.mean(np.abs(float_pred - compressed_pred)))
    total_params, nonzero_params, float_mb, effective_mb = compute_effective_size_mb(model)

    print("" + "=" * 84)
    print(f"Parameters:        {total_params:,} total | {nonzero_params:,} non-zero "
          f"({100.0 * (1.0 - nonzero_params / total_params):.1f}% overall sparsity)")
    print(f"Memory Footprint:  Float32={float_mb:.2f} MB -> Pruned+Quantized={effective_mb:.2f} MB "
          f"({float_mb / effective_mb:.2f}x compression)")
    print(f"Test MAE:          Float={float_mae:.4f} | Pruned+Quantized={compressed_mae:.4f}")
    print(f"Float vs Quant:    Drift MAE={drift_mae:.6f}")
    print("=" * 84)


if __name__ == "__main__":
    main()
