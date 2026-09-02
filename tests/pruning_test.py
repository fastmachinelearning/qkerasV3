import numpy as np
import keras

from keras import layers

from qkeras import QDense

from qkeras.pruning import (
    MagnitudePruner,
    PruningCallback,
)


def test_pruner_initial_mask():
    pruner = MagnitudePruner(
        final_sparsity=0.8,
        begin_epoch=1,
        end_epoch=5,
    )

    pruner.build((4, 4))

    assert pruner.mask is not None
    assert np.all(pruner.mask.numpy() == 1)


def test_pruner_mask_update():
    pruner = MagnitudePruner(
        final_sparsity=0.5,
        begin_epoch=0,
        end_epoch=1,
        update_probability=1,
    )

    weights = keras.Variable(
        [
            [1.0, 2.0],
            [0.1, 0.2],
        ],
        dtype="float32",
    )

    pruner.build(weights.shape)

    pruner.set_steps_per_epoch(1)

    pruner.update(
        weights,
        step=1,
    )

    mask = pruner.mask.numpy()

    sparsity = np.mean(mask == 0)

    assert sparsity >= 0.25
    assert sparsity <= 0.75


def test_pruner_removes_small_weights():
    pruner = MagnitudePruner(
        final_sparsity=0.5,
        begin_epoch=0,
        end_epoch=1,
        update_probability=1,
    )

    weights = keras.Variable(
        [
            [10.0, 0.01],
            [0.02, 20.0],
        ],
        dtype="float32",
    )

    pruner.build(weights.shape)

    pruner.set_steps_per_epoch(1)

    pruner.update(
        weights,
        step=1,
    )

    result = keras.ops.convert_to_numpy(pruner(weights))

    assert result[0, 1] == 0
    assert result[1, 0] == 0
    assert result[0, 0] != 0
    assert result[1, 1] != 0


def test_qdense_with_pruning():

    inputs = keras.Input((16,))

    outputs = QDense(
        8,
        kernel_pruner=MagnitudePruner(
            final_sparsity=0.5,
            begin_epoch=0,
            end_epoch=1,
        ),
    )(inputs)

    model = keras.Model(
        inputs,
        outputs,
    )

    x = np.random.random((4, 16))

    y = model(x)

    assert y.shape == (4, 8)

    dense = model.layers[-1]

    # Important: pruning must not add Keras weights
    assert len(dense.weights) == 2


def test_pruning_training():

    np.random.seed(1)

    x = np.random.random(
        (256, 32)
    ).astype("float32")

    y = keras.utils.to_categorical(
        np.random.randint(
            0,
            10,
            size=(256,),
        ),
        10,
    )

    inputs = keras.Input(
        (32,)
    )

    x1 = QDense(
        32,
        activation="relu",
        kernel_pruner=MagnitudePruner(
            final_sparsity=0.8,
            begin_epoch=1,
            end_epoch=3,
            update_probability=0.3,
        ),
        name="dense0",
    )(inputs)

    outputs = QDense(
        10,
        activation="softmax",
        kernel_pruner=MagnitudePruner(
            final_sparsity=0.8,
            begin_epoch=1,
            end_epoch=3,
            update_probability=0.5
        ),
        name="dense1",
    )(x1)

    model = keras.Model(
        inputs,
        outputs,
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
    )

    model.fit(
        x,
        y,
        batch_size=32,
        epochs=4,
        verbose=0,
        callbacks=[
            PruningCallback(
                verbose=True
            )
        ],
    )

    sparsities = []

    for layer in model.layers:
        if hasattr(layer, "kernel_pruner") and layer.kernel_pruner:
            sparsities.append(
                layer.kernel_pruner.sparsity()
            )

    assert len(sparsities) == 2

    for sparsity in sparsities:
        assert sparsity > 0.5
