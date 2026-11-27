import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # examples/timestral

from dataclasses import asdict
from pathlib import Path
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
from safetensors.numpy import load_file

from qkeras.distill.factory import QuantFactory
from timestral_kd.legacy import legacy_factory
from timestral_kd.model import (
    TimestralSpec,
    build_timestral,
    forecast,
    load_timestral_weights,
)

# ----------------------------------------------------------------------------- configuration
MODEL_DIR = Path(os.environ.get("TIMESTRAL_CKPT", Path(__file__).resolve().parents[3] / "ts" / "Timestral-5M"))
CONTEXT_PATCHES = 16  # L = 16 -> 512 context steps
HORIZON_PATCHES = 3   # h = 3  -> 96 horizon steps


# ----------------------------------------------------------------------------- quantization presets
PRESETS: dict[str, QuantFactory] = {
    "0_qkeras_float_sanity": legacy_factory(None, skip_head_post=False),
    "1_weights_8b_only": legacy_factory(
        None,
        skip_head_post=False,
        weight_q="quantized_bits(8, 1, alpha='auto_po2')",
        bias_q="quantized_bits(12, 4, alpha='auto_po2')",
    ),
    # Isolate each activation block with 8-bit weights to see exact sensitivity:
    "2_w8_plus_rms_only": legacy_factory(
        None,
        skip_head_post=False,
        weight_q="quantized_bits(8, 1, alpha='auto_po2')",
        bias_q="quantized_bits(12, 4, alpha='auto_po2')",
        rms_ms_q="quantized_bits(16, 6, keep_negative=False)",
        rms_rsqrt_q="quantized_bits(16, 10, keep_negative=False)",
    ),
    "3_w8_plus_lut_only": legacy_factory(
        None,
        skip_head_post=False,
        weight_q="quantized_bits(8, 1, alpha='auto_po2')",
        bias_q="quantized_bits(12, 4, alpha='auto_po2')",
        lut_in_q="quantized_bits(12, 4, symmetric=1)",
        lut_out_q="quantized_bits(12, 2, symmetric=1)",
    ),
    "4_w8_plus_attn_fixed_inv": legacy_factory(
        None,
        skip_head_post=False,
        weight_q="quantized_bits(8, 1, alpha='auto_po2')",
        bias_q="quantized_bits(12, 4, alpha='auto_po2')",
        attn_score_q="quantized_bits(12, 5, symmetric=1)",
        softmax_inv_q="quantized_bits(16, 0, keep_negative=False)",  # 0 int bits! 1/sum(exp) <= 1.0
        softmax_prob_q="quantized_bits(12, 0, keep_negative=False)",
    ),
    "5_full_ptq_fixed": legacy_factory(
        None,
        skip_head_post=False,
        weight_q="quantized_bits(8, 1, alpha='auto_po2')",
        bias_q="quantized_bits(12, 4, alpha='auto_po2')",
        act_q="quantized_bits(16, 5, symmetric=1)",
        lut_in_q="quantized_bits(12, 4, symmetric=1)",
        lut_out_q="quantized_bits(12, 2, symmetric=1)",
        rms_ms_q="quantized_bits(16, 6, keep_negative=False)",
        rms_rsqrt_q="quantized_bits(16, 10, keep_negative=False)",
        attn_score_q="quantized_bits(12, 5, symmetric=1)",
        softmax_inv_q="quantized_bits(16, 0, keep_negative=False)",  # Fixed: was 10 int bits!
        softmax_prob_q="quantized_bits(12, 0, keep_negative=False)",
    ),
    "6_full_ptq_12b_fixed": legacy_factory(
        None,
        skip_head_post=False,
        weight_q="quantized_bits(8, 1, alpha='auto_po2')",
        bias_q="quantized_bits(12, 4, alpha='auto_po2')",
        act_q="quantized_bits(12, 4, symmetric=1)",
        lut_in_q="quantized_bits(10, 4, symmetric=1)",
        lut_out_q="quantized_bits(10, 2, symmetric=1)",
        rms_ms_q="quantized_bits(14, 6, keep_negative=False)",
        rms_rsqrt_q="quantized_bits(14, 10, keep_negative=False)",
        attn_score_q="quantized_bits(10, 5, symmetric=1)",
        softmax_inv_q="quantized_bits(12, 0, keep_negative=False)",
        softmax_prob_q="quantized_bits(10, 0, keep_negative=False)",
    ),
}


def generate_synthetic_data(
    context_len: int, horizon_len: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate 4 synthetic univariate series (trend + sinusoid + Gaussian noise)."""
    rng = np.random.default_rng(seed)
    total_len = context_len + horizon_len
    t = np.arange(total_len, dtype=np.float32)[None, :]

    slopes = np.array([0.02, -0.015, 0.04, 0.005], dtype=np.float32)[:, None]
    offsets = np.array([10.0, 50.0, -5.0, 100.0], dtype=np.float32)[:, None]
    periods = np.array([32.0, 64.0, 24.0, 48.0], dtype=np.float32)[:, None]
    amps = np.array([3.0, 8.0, 1.5, 12.0], dtype=np.float32)[:, None]
    phases = np.array([0.0, 1.1, 2.3, 0.5], dtype=np.float32)[:, None]
    noise_stds = np.array([0.2, 0.5, 0.1, 0.8], dtype=np.float32)[:, None]

    trend = offsets + slopes * t
    seasonality = amps * np.sin(2.0 * np.pi * t / periods + phases)
    noise = rng.normal(0.0, noise_stds, size=(4, total_len)).astype(np.float32)

    full_series = trend + seasonality + noise
    return full_series[:, :context_len], full_series[:, context_len:]


def plot_sweep_results(
    context: np.ndarray,
    ground_truth: np.ndarray,
    float_q: np.ndarray,
    results: dict[str, dict],
    spec: TimestralSpec,
    save_path: Path = Path("quantization_sweep_comparison.png"),
) -> None:
    """Plot horizon forecasts across all quantization presets and bar chart of drift."""
    p50_idx = spec.quantile_knots.index(0.5)
    ctx_len = context.shape[1]
    hor_len = ground_truth.shape[1]
    t_hor = np.arange(ctx_len, ctx_len + hor_len)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    colors = plt.cm.tab10.colors

    # First 4 panels: Horizon zoom per synthetic series
    for i in range(4):
        ax = axes[i // 2, i % 2]
        ax.plot(t_hor, ground_truth[i], label="Ground Truth", color="black", linestyle="--", linewidth=1.3)
        ax.plot(t_hor, float_q[i, :, p50_idx], label="Float Baseline", color="crimson", linewidth=1.6)

        for idx, (name, res) in enumerate(results.items()):
            # Skip exploded runs from line plot so y-axis stays readable
            if res["overall_drift"] > 100.0:
                continue
            ax.plot(
                t_hor,
                res["quant_q"][i, :, p50_idx],
                label=f"{name} (drift={res['drift_per_row'][i]:.3f})",
                color=colors[idx % len(colors)],
                linestyle="-.",
                linewidth=1.2,
            )

        ax.set_title(f"Series {i + 1} Horizon Comparison", fontsize=10)
        ax.set_xlabel("Time Step")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7.5)

    # Panel 5 (top-right): Overall Drift MAE bar chart (log scale)
    ax_bar = axes[0, 2]
    names = list(results.keys())
    drifts = [max(results[k]["overall_drift"], 1e-8) for k in names]
    bars = ax_bar.barh(names, drifts, color=[colors[i % len(colors)] for i in range(len(names))])
    ax_bar.set_xscale("log")
    ax_bar.set_xlabel("Mean Float-vs-Quant Drift MAE (log scale)")
    ax_bar.set_title("Quantization Drift by Preset", fontsize=10)
    ax_bar.invert_yaxis()
    ax_bar.grid(True, which="both", axis="x", alpha=0.3)
    for bar, val in zip(bars, [results[k]["overall_drift"] for k in names]):
        ax_bar.text(bar.get_width() * 1.15, bar.get_y() + bar.get_height() / 2, f"{val:.4g}", va="center", fontsize=8)

    # Panel 6 (bottom-right): Quantized MAE vs Ground Truth
    ax_mae = axes[1, 2]
    float_overall_mae = float(np.mean(np.abs(float_q[:, :, p50_idx] - ground_truth)))
    q_maes = [min(results[k]["overall_mae"], float_overall_mae * 5.0) for k in names]
    ax_mae.barh(names, q_maes, color=[colors[i % len(colors)] for i in range(len(names))])
    ax_mae.axvline(float_overall_mae, color="crimson", linestyle="--", linewidth=1.4, label=f"Float MAE ({float_overall_mae:.3f})")
    ax_mae.set_xlabel("Ground-Truth MAE (capped at 5x Float)")
    ax_mae.set_title("Forecast Accuracy vs Ground Truth", fontsize=10)
    ax_mae.invert_yaxis()
    ax_mae.grid(True, axis="x", alpha=0.3)
    ax_mae.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved comparison plot to: {save_path.resolve()}")
    plt.show()


def print_model_size(model: keras.Model, label: str = "Model") -> None:
    """Print parameter counts and effective weight memory footprint (Float32 vs. Quantized)."""
    total_params = model.count_params()
    float_bytes = total_params * 4  # 32-bit float = 4 bytes/param

    total_bits = 0
    quantized_params = 0

    for layer in model.layers:
        # Map each weight variable in the layer to its QKeras quantizer (if any)
        q_map = {}
        if hasattr(layer, "kernel"):
            q_map[id(layer.kernel)] = getattr(layer, "kernel_quantizer_internal", None)
        if getattr(layer, "bias", None) is not None:
            q_map[id(layer.bias)] = getattr(layer, "bias_quantizer_internal", None)
        if hasattr(layer, "scale"):  # RMSNorm / QRMSNorm gain
            q_map[id(layer.scale)] = getattr(layer, "gain_quantizer_internal", None)

        for w in layer.weights:
            n = int(np.prod(w.shape))
            q = q_map.get(id(w))
            bits = getattr(q, "bits", None) if q is not None else None
            if bits is not None:
                total_bits += n * int(bits)
                quantized_params += n
            else:
                total_bits += n * 32

    quant_bytes = total_bits / 8.0
    ratio = float_bytes / quant_bytes if quant_bytes > 0 else 1.0

    print(
        f"[{label}] Params: {total_params:,} ({quantized_params:,} quantized) | "
        f"Float32: {float_bytes / 1024**2:.2f} MB | "
        f"Quantized: {quant_bytes / 1024**2:.2f} MB ({ratio:.2f}x smaller)"
    )


def main() -> None:
    spec = TimestralSpec.from_json(MODEL_DIR / "config.json")
    weights = load_file(MODEL_DIR / "model.safetensors")

    ctx_len = CONTEXT_PATCHES * spec.patch_size
    hor_len = HORIZON_PATCHES * spec.patch_size
    context, ground_truth = generate_synthetic_data(ctx_len, hor_len)
    print(context[0])
    exit()

    # Run Float Baseline
    float_model = build_timestral(spec, CONTEXT_PATCHES, HORIZON_PATCHES, name="timestral_float")
    load_timestral_weights(float_model, weights, spec)
    float_q = forecast(float_model, context, spec)

    p50_idx = spec.quantile_knots.index(0.5)
    float_pred = float_q[:, :, p50_idx]
    float_mae = np.mean(np.abs(float_pred - ground_truth), axis=-1)

    print("=" * 88)
    print(f"Float Baseline | Overall MAE: {np.mean(float_mae):.4f} | Per-row MAE: {np.round(float_mae, 4)}")
    print("=" * 88)
    print(f"{'Preset':<24} | {'Overall MAE':>12} | {'Overall Drift':>13} | {'Max Drift':>11} | {'Time (s)':>8}")
    print("-" * 88)

    results: dict[str, dict] = {}
    for preset_name, factory in PRESETS.items():
        t0 = time.perf_counter()
        keras.backend.clear_session()

        q_model = build_timestral(
            spec, CONTEXT_PATCHES, HORIZON_PATCHES, factory=factory, name=f"timestral_{preset_name}"
        )
        print_model_size(q_model, label=preset_name)
        load_timestral_weights(q_model, weights, spec)
        quant_q = forecast(q_model, context, spec)
        elapsed = time.perf_counter() - t0

        quant_pred = quant_q[:, :, p50_idx]
        mae_per_row = np.mean(np.abs(quant_pred - ground_truth), axis=-1)
        drift_per_row = np.mean(np.abs(float_pred - quant_pred), axis=-1)

        overall_mae = float(np.mean(mae_per_row))
        overall_drift = float(np.mean(drift_per_row))
        max_drift = float(np.max(np.abs(float_pred - quant_pred)))

        results[preset_name] = {
            "config": asdict(factory),
            "quant_q": quant_q,
            "mae_per_row": mae_per_row,
            "drift_per_row": drift_per_row,
            "overall_mae": overall_mae,
            "overall_drift": overall_drift,
            "max_drift": max_drift,
        }

        print(
            f"{preset_name:<24} | {overall_mae:12.4f} | {overall_drift:13.6f} | {max_drift:11.6f} | {elapsed:8.2f}"
        )

    print("=" * 88)
    plot_sweep_results(context, ground_truth, float_q, results, spec)


if __name__ == "__main__":
    main()
