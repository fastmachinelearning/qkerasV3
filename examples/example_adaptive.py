import numpy as np
import keras
from qkeras import QAdaptiveActivation


class QKerasAdaptiveStepLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get("accuracy", 0.0)
        loss = logs.get("loss", 0.0)

        for layer in self.model.layers:
            if isinstance(layer, QAdaptiveActivation):
                integer_bits = layer.quantizer.integer.value.numpy()[0]
                total_bits = layer.total_bits
                unsigned_bits = layer.current_total_bits - int(layer.keep_negative)
                fractional_bits = unsigned_bits - integer_bits
                v_max = 2.0 ** (integer_bits - 1) - 2.0 ** (-fractional_bits)
                v_min = -2.0 ** (integer_bits - 1) # the most significant bit) is 1 and all other bits are 0

                print(
                    f"Epoch {epoch + 1:02d} -> Total Bits: {total_bits}b | "
                    f"Integer Bits: {integer_bits} | Vmax: {v_max} | Vmin: {v_min} | ema_min: {layer.ema_min.value.numpy()[0]} | ema_max: {layer.ema_max.value.numpy()[0]} | Loss: {loss:.4f} | Acc: {acc:.4f}"
                )

(X_train, y_train), _ = keras.datasets.mnist.load_data()

X_train = X_train.astype("float32") / 255.0
X_train = X_train.reshape(-1, 28 * 28)

y_train = keras.utils.to_categorical(y_train, 10)

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(32),
    QAdaptiveActivation(
        total_bits=8,
        min_bits=0,
        activation="quantized_bits",
        trainable=True,
        name="adaptive_qat_act"
    ),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

adaptive_logger = QKerasAdaptiveStepLogger()

model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0, callbacks=[adaptive_logger])
