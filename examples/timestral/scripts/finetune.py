from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # examples/timestral

import argparse
from pathlib import Path

import keras
from keras import ops
import numpy as np
from qkeras.range_analysis import optimize_bitwidths
from safetensors.numpy import load_file

from timestral_kd.legacy import legacy_factory
from timestral_kd.model import (
    LayerFactory,
    TimestralSpec,
    build_timestral_multivariate,
    forecast_mv,
    load_timestral_weights,
    scale_context_mv,
)
from forecast_csv import (
    CONTEXT_PATCHES,
    COVARIATE_COLUMNS,
    DEFAULT_CSV,
    HORIZON_PATCHES,
    MODEL_DIR,
    TARGET_COLUMNS,
    build_sliding_windows,
    load_process_csv,
    plot_multivariate_forecast,
)


def normalize_targets_mv(
    y: np.ndarray, loc: np.ndarray, scale: np.ndarray, use_asinh: bool = True
) -> np.ndarray:
    """Transform raw horizon targets ``(B, V, H)`` into normalized model space using context anchors."""
    y = np.asarray(y, dtype=np.float32)
    y_norm = (y - loc) / scale
    if use_asinh:
        y_norm = np.arcsinh(y_norm)
    return y_norm.astype(np.float32)


class MultiQuantilePinballLoss(keras.losses.Loss):
    """Pinball loss averaged across all quantile knots for ``y_true (B, V, H)`` and ``y_pred (B, V, H, Q)``."""

    def __init__(self, quantile_knots: tuple[float, ...], name: str = "pinball_loss"):
        super().__init__(name=name)
        self.quantile_knots = tuple(float(k) for k in quantile_knots)

    def call(self, y_true, y_pred):
        taus = ops.convert_to_tensor(self.quantile_knots, dtype=y_pred.dtype)[None, None, None, :]
        err = y_true[..., None] - y_pred
        loss = ops.maximum(taus * err, (taus - 1.0) * err)
        return ops.mean(loss)

    def get_config(self):
        cfg = super().get_config()
        cfg["quantile_knots"] = list(self.quantile_knots)
        return cfg


def set_backbone_trainable(model: keras.Model, trainable: bool) -> None:
    """Freeze/unfreeze pretrained univariate layers while keeping head & variate-attention trainable."""
    always_trainable_prefixes = ("head_", "forecast_query", "out_norm")
    for layer in model.layers:
        if any(layer.name.startswith(p) for p in always_trainable_prefixes) or "_attn1" in layer.name:
            layer.trainable = True
        else:
            layer.trainable = trainable


def evaluate_mae(
    model: keras.Model,
    ctx_windows: np.ndarray,
    gt_windows: np.ndarray,
    active_cols: list[str],
    spec: TimestralSpec,
    batch_size: int = 32,
) -> dict[str, float]:
    """Compute raw-space P50 MAE per target variate in batches."""
    p50_idx = spec.quantile_knots.index(0.5)
    preds_list = [
        forecast_mv(model, ctx_windows[i : i + batch_size], spec)
        for i in range(0, len(ctx_windows), batch_size)
    ]
    preds = np.concatenate(preds_list, axis=0)
    metrics = {}
    for v_idx, col_name in enumerate(active_cols):
        if col_name not in TARGET_COLUMNS and col_name != "Lambda":
            continue
        p50 = preds[:, v_idx, :, p50_idx]
        gt = gt_windows[:, v_idx, :]
        metrics[col_name] = float(np.mean(np.abs(p50 - gt)))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Multivariate Timestral on Process CSV")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to semicolon-separated CSV")
    parser.add_argument("--include-covariates", action="store_true", help="Condition on process covariates")
    parser.add_argument("--quantize", action="store_true", help="Fine-tune with QKeras QAT")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Chronological validation split fraction")
    parser.add_argument("--train-stride", type=int, default=8, help="Sliding window stride for training")
    parser.add_argument("--val-stride", type=int, default=32, help="Sliding window stride for validation")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Epochs training head + variate attention only")
    parser.add_argument("--epochs", type=int, default=10, help="Full-model fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Peak learning rate for full fine-tuning")
    parser.add_argument("--weights-out", type=Path, default=Path("timestral_mv_finetuned.weights.h5"))
    parser.add_argument("--plot-out", type=Path, default=Path("forecast_finetuned.png"))
    args = parser.parse_args()

    spec = TimestralSpec.from_json(MODEL_DIR / "config.json")
    weights = load_file(MODEL_DIR / "model.safetensors")
    ctx_len = CONTEXT_PATCHES * spec.patch_size
    hor_len = HORIZON_PATCHES * spec.patch_size

    covs = COVARIATE_COLUMNS if args.include_covariates else []
    df, active_cols = load_process_csv(args.csv, TARGET_COLUMNS, covs)
    num_variates = len(active_cols)
    data_matrix = df[active_cols].to_numpy(dtype=np.float32)

    # Strict chronological split with a full window gap to prevent data leakage
    split_row = int(len(data_matrix) * (1.0 - args.val_fraction))
    gap = ctx_len + hor_len
    train_mat = data_matrix[: max(split_row - gap, gap + 1)]
    val_mat = data_matrix[split_row:]

    train_ctx, train_gt, _ = build_sliding_windows(train_mat, ctx_len, hor_len, stride=args.train_stride)
    val_ctx, val_gt, val_starts = build_sliding_windows(val_mat, ctx_len, hor_len, stride=args.val_stride)
    print(f"Train windows: {len(train_ctx):,} | Val windows: {len(val_ctx):,} | Variates: {num_variates}")

    # Scale contexts & normalize targets with the same causal anchor
    x_train, loc_tr, sc_tr = scale_context_mv(train_ctx, spec.patch_size, spec.minimum_scale, spec.use_asinh)
    y_train = normalize_targets_mv(train_gt, loc_tr, sc_tr, spec.use_asinh)

    x_val, loc_va, sc_va = scale_context_mv(val_ctx, spec.patch_size, spec.minimum_scale, spec.use_asinh)
    y_val = normalize_targets_mv(val_gt, loc_va, sc_va, spec.use_asinh)

    factory = legacy_factory() if args.quantize else LayerFactory()
    model = build_timestral_multivariate(
        spec,
        num_variates=num_variates,
        context_patches=CONTEXT_PATCHES,
        horizon_patches=HORIZON_PATCHES,
        factory=factory,
        name="timestral_mv_ft",
    )
    load_timestral_weights(model, weights, spec)

    if args.quantize:
        optimize_bitwidths(model, data=x_train[:32], policy="keep_fractional", headroom=1, verbose=1)

    baseline_mae = evaluate_mae(model, val_ctx, val_gt, active_cols, spec, batch_size=args.batch_size)
    print("Zero-shot validation MAE:")
    for k, v in baseline_mae.items():
        print(f"  {k:<25}: {v:.4f}")

    loss_fn = MultiQuantilePinballLoss(spec.quantile_knots)

    # Stage 1: Warmup head + variate attention
    if args.warmup_epochs > 0:
        print(f"--- Stage 1: Warmup head & variate attention ({args.warmup_epochs} epochs) ---")
        set_backbone_trainable(model, trainable=False)
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4, clipnorm=1.0),
            loss=loss_fn,
        )
        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            batch_size=args.batch_size,
            epochs=args.warmup_epochs,
            verbose=1,
        )

    # Stage 2: Full model fine-tuning with cosine decay
    if args.epochs > 0:
        print(f"--- Stage 2: Full fine-tuning ({args.epochs} epochs) ---")
        set_backbone_trainable(model, trainable=True)
        total_steps = max(1, (len(x_train) // args.batch_size) * args.epochs)
        lr_sched = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.lr,
            decay_steps=total_steps,
            alpha=0.1,
        )
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=lr_sched, weight_decay=1e-4, clipnorm=1.0),
            loss=loss_fn,
        )
        ckpt_cb = keras.callbacks.ModelCheckpoint(
            filepath=str(args.weights_out),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )
        early_cb = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=[ckpt_cb, early_cb],
            verbose=1,
        )

    ft_mae = evaluate_mae(model, val_ctx, val_gt, active_cols, spec, batch_size=args.batch_size)
    print("Validation MAE comparison (Zero-shot -> Fine-tuned):")
    for k in baseline_mae:
        b, f_val = baseline_mae[k], ft_mae[k]
        rel = 100.0 * (f_val - b) / max(b, 1e-8)
        print(f"  {k:<25}: {b:.4f} -> {f_val:.4f} ({rel:+.1f}%)")

    # Plot the latest validation window
    win_idx = -1 % len(val_ctx)
    start_row = split_row + val_starts[win_idx]
    ctx_ts = df["Datetime_UTC"].iloc[start_row : start_row + ctx_len]
    hor_ts = df["Datetime_UTC"].iloc[start_row + ctx_len : start_row + ctx_len + hor_len]
    sample_pred = forecast_mv(model, val_ctx[win_idx : win_idx + 1], spec)[0]

    plot_multivariate_forecast(
        ctx_ts=ctx_ts,
        hor_ts=hor_ts,
        ctx_window=val_ctx[win_idx],
        pred_window=sample_pred,
        gt_window=val_gt[win_idx],
        active_cols=active_cols,
        quantile_knots=spec.quantile_knots,
        out_path=args.plot_out,
        show=False,
    )


if __name__ == "__main__":
    main()
