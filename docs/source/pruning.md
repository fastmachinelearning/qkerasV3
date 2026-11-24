# Pruning

QKerasV3 ships its own Keras 3 native magnitude pruning (`qkeras.pruning`). It
replaces `tensorflow-model-optimization` (tfmot), which does not support
Keras 3, and works with the TensorFlow, JAX and PyTorch backends.

## Supported layers

`QDense`, `QConv1D`, `QConv2D`, `QConv1DTranspose`, `QConv2DTranspose`, `QDepthwiseConv2D`,
`QSeparableConv1D`, `QSeparableConv2D` (depthwise and pointwise kernel),
`QDenseBatchnorm`, `QConv2DBatchnorm` and `QDepthwiseConv2DBatchnorm`.
Biases are not pruned. Recurrent layers are not supported yet.

The mask is applied **before** the kernel quantizer, i.e. the layer computes
`quantizer(kernel * mask)`. The callback updates masks between training steps
(not inside the compiled step), following a schedule.

## Workflow 1: set the pruner in the layer

```python
import keras
from qkeras import QDense, QActivation
from qkeras.pruning import (MagnitudePruner, PolynomialDecay,
                            UpdatePruningStep, strip_pruning)

pruner = MagnitudePruner(
    PolynomialDecay(initial_sparsity=0.0, final_sparsity=0.75,
                    begin_step=0, end_step=2000, frequency=100))

inputs = keras.Input((16,))
x = QDense(64, kernel_quantizer="quantized_bits(6,0,1)",
           bias_quantizer="quantized_bits(6,0,1)", kernel_pruner=pruner)(inputs)
x = QActivation("quantized_relu(6)")(x)
outputs = QDense(5, kernel_quantizer="quantized_bits(6,0,1)",
                 kernel_pruner=pruner)(x)
model = keras.Model(inputs, outputs)

model.compile("adam", "categorical_crossentropy")
model.fit(x_train, y_train, epochs=20, callbacks=[UpdatePruningStep(verbose=1)])

model = strip_pruning(model)   # plain QKeras model, pruned weights are exactly 0
```

## Workflow 2: prune an existing (pre-trained) model, tfmot style

Magnitude pruning works best on a pre-trained model:

```python
from qkeras.pruning import prune_low_magnitude

pruned = prune_low_magnitude(model, PolynomialDecay(0.0, 0.5, 0, 2000))
# optionally: prune_low_magnitude(model, 0.5, layer_names=["fc1", "fc2"])
pruned.compile(...)
pruned.fit(..., callbacks=[UpdatePruningStep()])
final = strip_pruning(pruned)
```

`prune_low_magnitude` and `strip_pruning` clone Sequential and functional
models. For subclassed models set `kernel_pruner` in the layer constructors and
call `apply_pruning_masks(model)` after training.

## Migrating from tfmot

| tfmot (Keras 2)                                   | QKerasV3                                   |
|---------------------------------------------------|--------------------------------------------|
| `tfmot.sparsity.keras.prune_low_magnitude(m, pruning_schedule=s)` | `prune_low_magnitude(m, s)` |
| `tfmot.sparsity.keras.PolynomialDecay(...)`       | `PolynomialDecay(...)` (same arguments)    |
| `tfmot.sparsity.keras.ConstantSparsity(...)`      | `ConstantSparsity(...)` (same arguments)   |
| `tfmot.sparsity.keras.UpdatePruningStep()`        | `UpdatePruningStep()`                      |
| `tfmot.sparsity.keras.strip_pruning(m)`           | `strip_pruning(m)`                         |
| `PruneLowMagnitude` wrapper layer                 | `kernel_pruner=` argument on the layer     |

Not (yet) supported: `block_size`/`block_pooling_type`, `PruningSummaries`,
`PrunableLayer` for user-defined layers (use `qkeras.pruning.PrunableLayerMixin`).

## Details

* Schedules count optimizer steps. Updates happen at
  `begin_step + k * frequency` and at `end_step`; `end_step=-1` (constant
  sparsity) never stops. `steps_per_execution > 1` is handled.
* `MagnitudePruner` prunes exactly `round(sparsity * n)` weights per tensor.
  Pruned weights stay pruned (ranking uses `kernel * mask`).
* Masks are stored in a non-trainable `pruning_masks` sub-layer. Models with
  pruning save/load with `model.save` / `keras.models.load_model` /
  `qkeras.utils.load_qmodel`; layers without a pruner are unchanged.
* At the end of `fit` the callback bakes the masks into the kernels
  (`bake_on_train_end=True`), so stored kernels are exactly sparse.
* `get_pruning_summary(model)` returns per-layer sparsities;
  `qkeras.utils.print_model_sparsity(model)` prints them.
* For hls4ml export always call `strip_pruning` first.
