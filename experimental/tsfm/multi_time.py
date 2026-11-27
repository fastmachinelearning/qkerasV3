from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import keras
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qkeras import QActivation, QDense
from qkeras.qtransformer import QDotProductAttention, QRMSNorm, QUnary
from qkeras.range_analysis import optimize_bitwidths
from safetensors.numpy import load_file

from keras_timestral import (
    LayerFactory,
    TimestralSpec,
    build_timestral_multivariate,
    forecast_mv,
    load_timestral_weights,
    scale_context_mv,
)

# ----------------------------------------------------------------------------- configuration
MODEL_DIR = Path("/Users/mariuskoppel/cms/qkeras/ts/Timestral-5M")
DEFAULT_CSV = Path("process_data.csv")
CONTEXT_PATCHES = 16   # 16 * 32 = 512 steps (~17 min at 2s sampling)
HORIZON_PATCHES = 3    #  3 * 32 =  96 steps (~3.2 min forecast horizon)

# Target variates to forecast (plus covariates to condition on)
TARGET_COLUMNS = [
    "NOx",
    "Sauerstoff",
    "Brennkammertemperatur",
    "Lambda 1",
    "Lambda 2",
]

# Additional process covariates passed jointly into multivariate attention
COVARIATE_COLUMNS = [
    "Normvolumenstrom Abgas",
    "Eintragsschnecke 1 Geschwindigkeit",
    "Eintragsschnecke 2 Geschwindigkeit",
    "Reaktor 1 Prozessgastemperatur",
    "Reaktor 2 Prozessgastemperatur",
    "Betriebsvolumenstrom Abgas Gesamt",
    "Stellung AGR Klappe Brenner",
    "Temperatur Abgas Eintritt Brennkammer",
    "Drehzahl Verbrennungsluftgebläse",
    "Massendurchsatz [Intervallmessung]",
]


@dataclass
class QTimestralFactory(LayerFactory):
    """QKeras factory compatible with range_analysis bit-width optimization."""

    weight_q: str | None = "quantized_bits(8, 2, alpha=None)"
    bias_q: str | None = "quantized_bits(12, 4, alpha=None)"
    act_q: str | None = "quantized_bits(14, 5, symmetric=1)"
    lut_in_q: str | None = "quantized_bits(12, 4, symmetric=1)"
    lut_out_q: str | None = "quantized_bits(12, 3, symmetric=1)"
    rms_ms_q: str | None = "quantized_bits(16, 6, keep_negative=False)"
    rms_rsqrt_q: str | None = "quantized_bits(16, 10, keep_negative=False)"
    attn_score_q: str | None = "quantized_bits(12, 5, symmetric=1)"
    softmax_inv_q: str | None = "quantized_bits(14, 0, keep_negative=False)"
    softmax_prob_q: str | None = "quantized_bits(12, 0, keep_negative=False)"

    def dense(self, units, use_bias, name, activation=None):
        return QDense(
            units,
            use_bias=use_bias,
            activation=activation,
            kernel_quantizer=self.weight_q,
            bias_quantizer=self.bias_q if use_bias else None,
            name=name,
        )

    def norm(self, eps, name):
        return QRMSNorm(
            epsilon=eps,
            ms_quantizer=self.rms_ms_q,
            rsqrt_quantizer=self.rms_rsqrt_q,
            gain_quantizer=self.weight_q,
            output_quantizer=self.act_q,
            name=name,
        )

    def act(self, kind, name):
        if kind in ("silu", "sigmoid"):
            return QUnary(
                function=kind,
                input_quantizer=self.lut_in_q,
                output_quantizer=self.lut_out_q,
                name=name,
            )
        return keras.layers.Activation(kind, name=name)

    def attention(self, mask, name):
        return QDotProductAttention(
            attention_mask=mask,
            score_quantizer=self.attn_score_q,
            softmax_exp_quantizer=self.softmax_prob_q,
            softmax_inv_quantizer=self.softmax_inv_q,
            prob_quantizer=self.softmax_prob_q,
            output_quantizer=self.act_q,
            name=name,
        )

    def post(self, name):
        if "head" in name or not self.act_q:
            return None
        return QActivation(self.act_q, name=name)


def load_process_csv(
    csv_path: Path,
    target_cols: list[str] = TARGET_COLUMNS,
    covariate_cols: list[str] = COVARIATE_COLUMNS,
) -> tuple[pd.DataFrame, list[str]]:
    """Load European semicolon/comma-decimal CSV and clean missing/non-numeric entries."""
    df = pd.read_csv(csv_path, sep=";", decimal=",", encoding="utf-8-sig")
    df["Datetime_UTC"] = pd.to_datetime(df["Datetime_UTC"], errors="coerce")
    df = df.dropna(subset=["Datetime_UTC"]).sort_values("Datetime_UTC").reset_index(drop=True)

    # Support either "Lambda 1" / "Lambda 2" or a single "Lambda" column if present
    resolved_targets = []
    for col in target_cols:
        if col in df.columns:
            resolved_targets.append(col)
        elif col == "Lambda 1" and "Lambda" in df.columns:
            resolved_targets.append("Lambda")

    available_covs = [c for c in covariate_cols if c in df.columns and c not in resolved_targets]
    all_cols = resolved_targets + available_covs

    for col in all_cols:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )

    # Forward/backward fill brief sensor dropouts
    df[all_cols] = df[all_cols].ffill().bfill()
    return df, all_cols


def build_sliding_windows(
    values: np.ndarray, ctx_len: int, hor_len: int, stride: int = 32
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Slice ``(T, V)`` array into ``(B, V, ctx_len)`` contexts, ``(B, V, hor_len)`` horizons, and start indices."""
    total_steps, _ = values.shape
    win_len = ctx_len + hor_len
    if total_steps < win_len:
        raise ValueError(
            f"Need at least {win_len} rows ({ctx_len} context + {hor_len} horizon), got {total_steps}."
        )

    contexts, horizons, starts = [], [], []
    for start in range(0, total_steps - win_len + 1, stride):
        chunk = values[start : start + win_len].T  # (V, win_len)
        contexts.append(chunk[:, :ctx_len])
        horizons.append(chunk[:, ctx_len:])
        starts.append(start)
    return np.asarray(contexts, dtype=np.float32), np.asarray(horizons, dtype=np.float32), starts


def plot_multivariate_forecast(
    ctx_ts: pd.Series,
    hor_ts: list[pd.Timestamp] | pd.Series,
    ctx_window: np.ndarray,
    pred_window: np.ndarray,
    gt_window: np.ndarray | None,
    active_cols: list[str],
    quantile_knots: tuple[float, ...],
    out_path: Path = Path("forecast_multivariate.png"),
    show: bool = False,
) -> None:
    """Plot context history, P50 forecast, P10-P90 / P25-P75 bands, and ground truth for target variates."""
    target_indices = [
        (idx, name)
        for idx, name in enumerate(active_cols)
        if name in TARGET_COLUMNS or name == "Lambda"
    ]
    n_plots = len(target_indices)
    if n_plots == 0:
        return

    p10 = quantile_knots.index(0.1)
    p25 = quantile_knots.index(0.25)
    p50 = quantile_knots.index(0.5)
    p75 = quantile_knots.index(0.75)
    p90 = quantile_knots.index(0.9)

    fig, axes = plt.subplots(
        n_plots, 1, figsize=(13, 2.7 * n_plots), sharex=True, constrained_layout=True
    )
    if n_plots == 1:
        axes = [axes]

    ctx_x = pd.to_datetime(ctx_ts)
    hor_x = pd.to_datetime(hor_ts)
    split_x = ctx_x.iloc[-1]

    for ax, (v_idx, col_name) in zip(axes, target_indices):
        ctx_vals = ctx_window[v_idx]
        q_vals = pred_window[v_idx]  # (hor_len, Q)

        ax.plot(ctx_x, ctx_vals, color="#1f77b4", linewidth=1.4, label="Context history")
        ax.fill_between(
            hor_x,
            q_vals[:, p10],
            q_vals[:, p90],
            color="#ff7f0e",
            alpha=0.18,
            label="P10–P90 interval",
        )
        ax.fill_between(
            hor_x,
            q_vals[:, p25],
            q_vals[:, p75],
            color="#ff7f0e",
            alpha=0.32,
            label="P25–P75 interval",
        )
        ax.plot(
            hor_x,
            q_vals[:, p50],
            color="#d95f02",
            linewidth=2.0,
            label="Forecast (P50)",
        )

        if gt_window is not None:
            gt_vals = gt_window[v_idx]
            mae = float(np.mean(np.abs(q_vals[:, p50] - gt_vals)))
            ax.plot(
                hor_x,
                gt_vals,
                color="#2ca02c",
                linestyle="--",
                linewidth=1.5,
                label=f"Ground truth (MAE={mae:.3f})",
            )

        ax.axvline(split_x, color="0.35", linestyle=":", linewidth=1.1)
        ax.set_ylabel(col_name, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper left", ncol=5, fontsize=8.5, frameon=True)

    axes[0].set_title(
        "Multivariate Timestral Forecast — NOx, Sauerstoff, Brennkammertemperatur & Lambda",
        fontsize=12,
        fontweight="bold",
    )
    axes[-1].set_xlabel("Datetime (UTC)", fontsize=10)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()

    fig.savefig(out_path, dpi=160)
    print(f"Saved forecast plot to {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multivariate Timestral Forecasting on Process CSV")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to semicolon-separated CSV")
    parser.add_argument("--quantize", action="store_true", help="Run QKeras + range_analysis compression")
    parser.add_argument("--include-covariates", action="store_true", help="Condition on process covariates")
    parser.add_argument("--stride", type=int, default=32, help="Sliding window stride in rows")
    parser.add_argument("--window-idx", type=int, default=-1, help="Window index to plot (-1 = latest)")
    parser.add_argument("--plot-out", type=Path, default=Path("forecast_multivariate.png"), help="Output plot path")
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    args = parser.parse_args()

    spec = TimestralSpec.from_json(MODEL_DIR / "config.json")
    weights = load_file(MODEL_DIR / "model.safetensors")
    ctx_len = CONTEXT_PATCHES * spec.patch_size
    hor_len = HORIZON_PATCHES * spec.patch_size

    covs = COVARIATE_COLUMNS if args.include_covariates else []
    df, active_cols = load_process_csv(args.csv, TARGET_COLUMNS, covs)
    num_variates = len(active_cols)
    print(f"Loaded {len(df):,} rows from {args.csv} | Variates ({num_variates}): {active_cols}")

    data_matrix = df[active_cols].to_numpy(dtype=np.float32)  # (T, V)

    if len(data_matrix) >= ctx_len + hor_len:
        ctx_windows, gt_windows, starts = build_sliding_windows(
            data_matrix, ctx_len, hor_len, stride=args.stride
        )
        has_ground_truth = True
    elif len(data_matrix) >= ctx_len:
        ctx_windows = data_matrix[-ctx_len:].T[None, :, :]  # (1, V, ctx_len)
        gt_windows = None
        starts = [len(data_matrix) - ctx_len]
        has_ground_truth = False
    else:
        raise ValueError(f"CSV has {len(data_matrix)} rows, fewer than required context length {ctx_len}.")

    factory = QTimestralFactory() if args.quantize else LayerFactory()
    model = build_timestral_multivariate(
        spec,
        num_variates=num_variates,
        context_patches=CONTEXT_PATCHES,
        horizon_patches=HORIZON_PATCHES,
        factory=factory,
        name="timestral_process_mv",
    )
    load_timestral_weights(model, weights, spec)

    if args.quantize:
        s_calib, _, _ = scale_context_mv(
            ctx_windows[:32], spec.patch_size, spec.minimum_scale, spec.use_asinh
        )
        optimize_bitwidths(model, data=s_calib, policy="keep_fractional", headroom=1, verbose=1)

    # Run multivariate forecast: (B, V, hor_len, Q)
    preds = forecast_mv(model, ctx_windows, spec)
    p10_idx = spec.quantile_knots.index(0.1)
    p50_idx = spec.quantile_knots.index(0.5)
    p90_idx = spec.quantile_knots.index(0.9)

    print("\n" + "=" * 76)
    print(f"Forecast summary ({len(ctx_windows)} window(s), horizon={hor_len} steps):")
    for v_idx, col_name in enumerate(active_cols):
        if col_name not in TARGET_COLUMNS and col_name != "Lambda":
            continue
        p50 = preds[:, v_idx, :, p50_idx]
        if has_ground_truth:
            gt = gt_windows[:, v_idx, :]
            mae = float(np.mean(np.abs(p50 - gt)))
            rmse = float(np.sqrt(np.mean((p50 - gt) ** 2)))
            print(f"  {col_name:<25} | MAE={mae:8.4f} | RMSE={rmse:8.4f} | last P50={p50[-1, -1]:8.3f}")
        else:
            print(
                f"  {col_name:<25} | next-step P50={p50[0, 0]:8.3f} | "
                f"end-horizon P50={p50[0, -1]:8.3f} "
                f"(P10..P90: {preds[0, v_idx, -1, p10_idx]:.3f} .. {preds[0, v_idx, -1, p90_idx]:.3f})"
            )
    print("=" * 76)

    # Resolve selected window timestamps for CSV export and plotting
    win_idx = args.window_idx % len(ctx_windows)
    start_row = starts[win_idx]
    ctx_ts = df["Datetime_UTC"].iloc[start_row : start_row + ctx_len]
    if has_ground_truth:
        hor_ts = df["Datetime_UTC"].iloc[start_row + ctx_len : start_row + ctx_len + hor_len]
        gt_plot = gt_windows[win_idx]
    else:
        dt_step = df["Datetime_UTC"].diff().dropna().median()
        last_ts = ctx_ts.iloc[-1]
        hor_ts = [last_ts + (k + 1) * dt_step for k in range(hor_len)]
        gt_plot = None

    out_df = pd.DataFrame({"Datetime_UTC": hor_ts})
    for v_idx, col_name in enumerate(active_cols):
        if col_name not in TARGET_COLUMNS and col_name != "Lambda":
            continue
        out_df[f"{col_name}_P10"] = preds[win_idx, v_idx, :, p10_idx]
        out_df[f"{col_name}_P50"] = preds[win_idx, v_idx, :, p50_idx]
        out_df[f"{col_name}_P90"] = preds[win_idx, v_idx, :, p90_idx]

    out_path = Path("forecast_multivariate.csv")
    out_df.to_csv(out_path, sep=";", decimal=",", index=False)
    print(f"Saved horizon forecast to {out_path}")

    plot_multivariate_forecast(
        ctx_ts=ctx_ts,
        hor_ts=hor_ts,
        ctx_window=ctx_windows[win_idx],
        pred_window=preds[win_idx],
        gt_window=gt_plot,
        active_cols=active_cols,
        quantile_knots=spec.quantile_knots,
        out_path=args.plot_out,
        show=args.show,
    )


if __name__ == "__main__":
    main()