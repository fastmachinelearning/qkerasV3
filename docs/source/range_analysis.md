# Range analysis and bit-width adaptation

`qkeras.range_analysis` measures the values that reach every fixed-point
quantizer of a model and sizes the integer part of each quantizer to the
observed range. It runs after training (one shot) or during training (a
callback), on the TensorFlow, JAX and PyTorch backends.

## What is measured

For every QKeras layer the module records the **input** of each quantizer:

| quantizer               | value recorded                                   |
|-------------------------|--------------------------------------------------|
| `kernel_quantizer`, ... | float weights                                    |
| `bias_quantizer`        | float bias                                       |
| `activation` of QDense/QConv | pre-activation accumulator (incl. bias)     |
| `QActivation`           | the layer input                                  |

The input is used, not the output, because a quantizer clips: its output can
never show an overflow. In addition, every layer's input and output range is
stored in `report.layer_ranges`.

## How widths are chosen

For a quantizer with `S` sign bits, `I` integer bits (QKeras convention, sign
excluded) and `F = bits - S - I` fractional bits, the smallest `I` that holds
the observed range after rounding is chosen, plus `headroom` bits.

* `policy="keep_fractional"` (default): `F` is kept, so `bits` follows the
  range. This removes unused MSBs without changing the resolution; outputs are
  unchanged unless values were clipped before.
* `policy="keep_total"`: `bits` is kept and the binary point moves.

Calibration is repeated until no quantizer changes. A single pass would size a
downstream quantizer on values an upstream quantizer has already clipped.

Supported: `quantized_bits` with `alpha=None` or `alpha=1`, and
`quantized_relu` without sigmoid. Other quantizers are reported but not
changed. Note that `QDense`/`QConv*` switch a kernel `quantized_bits` with
`alpha=None` to `alpha="auto_po2"`; for those kernels the per-channel scale
already follows the range, so they are left alone. Use `alpha=1` for a plain
fixed-point kernel.

## Post-training

```python
from qkeras import optimize_bitwidths

# with real data
report = optimize_bitwidths(model, x_calib, policy="keep_fractional")

# with random inputs (uniform in [low, high] per input)
report = optimize_bitwidths(model, None, random_low=0.0, random_high=1.0)

print(report.summary())          # table: range, saturation, old -> new
report.to_dataframe()            # pandas, optional
report.hls4ml_precisions()       # {'layer': {'activation': 'fixed<14,5>', ...}}
```

`analyze_ranges(model, data)` only measures; `optimize_bitwidths(...,
dry_run=True)` measures and proposes without changing the model. Fine-tune
afterwards with `model.fit` as usual; compiled train/test/predict functions
are rebuilt automatically.

## During training

```python
from qkeras import RangeAnalysisCallback

cb = RangeAnalysisCallback(
    x_calib,            # or "random" with random_low / random_high
    frequency=1,        # check every epoch
    start_epoch=2,      # optional window
    stop_epoch=None,
    patience=2,         # shrink only after 2 agreeing checks; grow at once
    headroom=0,
)
model.fit(x, y, epochs=30, callbacks=[cb])

cb.enabled = False      # switch off (e.g. from another callback); True to resume
cb.history              # one RangeReport per check
```

The callback runs at epoch end, after the JAX trainer has synced its state.
Widths are Python attributes of the quantizers, so after a change the train
function is retraced (`make_train_function(force=True)`); the new widths are
active from the next step. Each check adds `range_analysis_bit_delta` to the
epoch logs.

## Options shared by both entry points

| argument | meaning |
|----------|---------|
| `quantile` | size from e.g. the 0.9999 quantile per batch instead of min/max |
| `headroom` | extra integer bits on top of the observed range |
| `min_bits`, `max_bits`, `min_integer`, `max_integer` | clamps |
| `adjust_weights`, `adjust_activations` | restrict to one role |
| `include`, `exclude` | layer-name filters |
| `layer.range_analysis = False` | per-layer opt-out |
| `max_batches`, `batch_size` | calibration size |

## Caveats

* The width is only as safe as the calibration data. Random inputs over the
  full input box usually give wider ranges than real data; use `headroom` for
  safety margin in hardware.
* Weight `Clip` constraints created by QKeras are updated together with the
  quantizer, so weights can use the new range.
* `QBatchNormalization` currently fails on the JAX and PyTorch backends
  independently of this module.
