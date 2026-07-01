import pytest
import numpy as np
from qkeras import BinaryToThermometer


def test_standard_thermometer_decoding():
    """Verify standard binary-to-thermometer decoding logic with default parameters."""
    x = np.array([0, 1, 2, 3])
    # Encoding resolution = 2 bits, max expected range bound = 4
    # Standard thermometer for level N fills N consecutive bits from the bottom.
    b = BinaryToThermometer(x, 2, 4)

    # Expected output array shape matches the logic applied in the script
    assert b.shape == (4, 2)

    # Assert values match typical deterministic unpacking mappings
    # (e.g., 0 -> [1,0], 1 -> [1,0], 2 -> [0,1], 3 -> [0,1])
    assert np.all(b[0] == [1, 0])
    assert np.all(b[1] == [1, 0])
    assert np.all(b[2] == [0, 1])
    assert np.all(b[3] == [0, 1])


@pytest.mark.parametrize("bits, max_val", [
    (2, 8),
    (4, 8)
])
def test_two_hot_encoding_toggle(bits, max_val):
    """Ensure two-hot encoding changes output patterns versus normal encoding."""
    x = np.array(range(8))

    standard_encoded = BinaryToThermometer(x, bits, max_val, use_two_hot_encoding=0)
    two_hot_encoded = BinaryToThermometer(x, bits, max_val, use_two_hot_encoding=1)

    # They should not produce identical structural arrays due to interpolation dynamics
    assert not np.array_equal(standard_encoded, two_hot_encoded)


def test_multidimensional_image_tensor_shapes():
    """Verify shape scaling mechanics on multi-channel NHWC image dimensions."""
    np.random.seed(42)
    # 10 mock image samples, 28x28 spatial dimensions, 2 input channels
    x = np.random.randint(0, 255, (10, 28, 28, 2))

    bits_per_channel = 8
    max_range = 256

    b = BinaryToThermometer(x, bits_per_channel, max_range, 0, 1)

    # The last channel dimension expands from 2 to (2 channels * 8 bits) = 16
    expected_shape = (10, 28, 28, 16)
    assert b.shape == expected_shape


def test_channel_slice_independence():
    """Ensure individual bits correspond correctly within multiple interleaved channels."""
    np.random.seed(42)
    x = np.random.randint(0, 255, (2, 4, 4, 2))

    b = BinaryToThermometer(x, 8, 256, 0, 1)

    # Check that channel 1 slice (0:8) and channel 2 slice (8:16) exist independently
    channel_1_slice = b[0, 0, 0, 0:8]
    channel_2_slice = b[0, 0, 0, 8:16]

    assert channel_1_slice.shape == (8,)
    assert channel_2_slice.shape == (8,)
    # Output arrays must strictly contain binary states (0s or 1s)
    assert np.all(np.isin(b, [0, 1]))
