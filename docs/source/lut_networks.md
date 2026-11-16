# LUT networks (`qkeras.logic`)

`qkeras.logic` trains networks whose neurons are FPGA look-up tables and exports them as a
framework-neutral netlist (`logicnet-netlist` JSON).  HDL/C generation, verification,
synthesis and place & route are done by the separate `logicnet-fpga` backend.

## Training LUT networks directly

```python
import keras
from qkeras.logic import LUTDense, GroupSum, extract_netlist, save_netlist

model = keras.Sequential([
    keras.Input((784,)),                                   # {0,1} inputs
    LUTDense(2000, lut_rank=3, connections="learnable", num_candidates=8),
    LUTDense(2000, lut_rank=3),
    LUTDense(1000, lut_rank=3),
    GroupSum(10, tau=100 / 16),
])
model.compile(keras.optimizers.Adam(0.03),
              keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
model.fit(x_train, y_train, epochs=20)
save_netlist(extract_netlist(model), "netlist.json")       # -> logicnet hdl netlist.json -o hdl/
```

## Converting a quantised QKeras network (range analysis, teacher -> student)

`qkeras.range_analysis.convert_to_lut` (same as `qkeras.logic.convert_to_lut`) turns a trained
quantised network into a LUT netlist:

* `analyze_value_sets` -- the finite set of values every quantised activation takes on the
  calibration data; it defines how many wires (code bits) the activation needs
  (`grid="range"`: the quantiser grid clipped to the observed range).
* `analyze_patterns` -- the input codes each neuron actually sees (its *care set*).
* Neurons with at most `max_exact_bits` (default 16) input wires are converted **exactly** by
  enumerating all input codes through the teacher's own ops (quantised kernel and bias,
  BatchNormalization with moving statistics, activation quantiser).  Unseen patterns are
  don't-cares (`dont_care="observed"`), or the table is fully specified (`dont_care="none"`).
* A final linear `QDense` becomes an exact integer weighted sum (`head="linear"`), with no fan-in
  limit.
* Wider neurons are **distilled** into LUTDense blocks (`distill=DistillConfig(...)`):
  layer-wise hints on the teacher's code bits, then end-to-end knowledge distillation through
  the frozen remainder of the teacher.

```python
from qkeras import QDense, QActivation
from qkeras.logic import FanInConstraint, FanInScheduler
from qkeras.range_analysis import convert_to_lut

x = keras.Input((784,))
h = QDense(128, kernel_quantizer="quantized_bits(3,0,1,alpha=1)", use_bias=False,
           kernel_constraint=FanInConstraint(784), name="d1")(x)
h = keras.layers.BatchNormalization()(h)
h = QActivation("quantized_relu(2,1)")(h)
o = QDense(10, kernel_quantizer="quantized_bits(3,0,1,alpha=1)",
           bias_quantizer="quantized_bits(6,2,1,alpha=1)")(h)
teacher = keras.Model(x, o)
teacher.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True))
teacher.fit(x_train, y_train, epochs=20,
            callbacks=[FanInScheduler({"d1": (784, 12)}, begin_epoch=1, end_epoch=8)])

conv = convert_to_lut(teacher, x_train, dont_care="observed")
print(conv.summary())
print(conv.evaluate(x_test, y_test, teacher=teacher))   # agreement 1.0 for exact conversions
save_netlist(conv.netlist, "teacher_lut.json")           # backend lowers wide LUTs + don't-cares
```

Requirements for exact conversion: fixed-point kernel/bias quantisers (`alpha=1`), quantised
activations on every hidden layer, a fixed-point bias on the output layer, valid-padding
`QConv2D` only.
