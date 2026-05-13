# Quick start

This page shows a minimal end-to-end example using quantized layers.

```{tip}
Always set the backend before importing `keras` / `qkeras`:
`export KERAS_BACKEND=tensorflow`
```

## Minimal quantized model

```python
import tensorflow as tf
from keras import layers, models
from qkeras import QDense, quantized_bits

model = models.Sequential(
    [
        layers.Input(shape=(128,)),
        QDense(
            64,
            activation="relu",
            kernel_quantizer=quantized_bits(8, 0, 1),
            bias_quantizer=quantized_bits(8, 0, 1),
        ),
        QDense(
            10,
            activation="softmax",
            kernel_quantizer=quantized_bits(8, 0, 1),
            bias_quantizer=quantized_bits(8, 0, 1),
        ),
    ]
)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
```

## Train briefly

```python
import numpy as np

x = np.random.randn(512, 128).astype("float32")
y = np.random.randint(0, 10, size=(512,), dtype="int32")

model.fit(x, y, epochs=1, batch_size=32)
```

## Next steps

- Browse the {doc}`examples/index` for runnable scripts.
- Read the {doc}`notebooks` section for tutorial-style walkthroughs.
- See the {doc}`api/index` for the full API reference.