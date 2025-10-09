# Copyright 2025 Marius Snella Köppel
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import keras
import keras.ops.numpy as knp
import numpy as np
import pytest
from keras import ops

from qkerasV3 import quantizer_registry

# Keep it deterministic
keras.utils.set_random_seed(812)

QUANTIZERS = [
    "quantized_linear",
    "quantized_bits",
    "bernoulli",
    "ternary",
    "stochastic_ternary",
    "binary",
    "stochastic_binary",
    "quantized_relu",
    "quantized_ulaw",
    "quantized_tanh",
    "quantized_sigmoid",
    "quantized_po2",
    "quantized_relu_po2",
    "quantized_hswish",
]

def _instantiate(quantizer):
    """lookup_quantizer may return a class or a callable. Instantiate if needed."""
    return quantizer() if isinstance(quantizer, type) else quantizer

def dtype_name(x):
    """Return dtype name consistently across backends."""
    dt = getattr(x, "dtype", None)
    if hasattr(dt, "name"):   # NumPy / TF dtypes
        return dt.name
    return str(dt).replace("<dtype: '", "").replace("'>", "")

@pytest.mark.parametrize("quantizer_name", QUANTIZERS)
@pytest.mark.parametrize("shape", [(8,), (4, 3), (2, 3, 2)])
@pytest.mark.parametrize("dtype", ["float32"])  # extend if your API supports others
def test_quantizer_output_type_dtype_shape(quantizer_name, shape, dtype):
    q = quantizer_registry.lookup_quantizer(quantizer_name)
    q = _instantiate(q)

    x_np = np.random.normal(size=shape).astype(np.float32)
    x = knp.array(x_np, dtype=dtype)

    y = q(x)

    assert ops.is_tensor(y), f"{quantizer_name} must return a tensor-like object"
    assert y.shape == x.shape, f"{quantizer_name} changed shape: {x.shape} -> {y.shape}"

    # Dtype check (many quantizers keep float32)
    assert str(y.dtype) == str(x.dtype), (
        f"{quantizer_name} changed dtype: {x.dtype} -> {y.dtype}"
    )

@pytest.mark.parametrize("quantizer_name", QUANTIZERS)
def test_quantizer_accepts_numpy_input_and_returns_tensor(quantizer_name):
    q = _instantiate(quantizer_registry.lookup_quantizer(quantizer_name))
    x_np = np.random.normal(size=(5,)).astype(np.float32)
    y = q(x_np)
    # Should convert NumPy to backend tensor internally
    assert ops.is_tensor(y), f"{quantizer_name} should return a tensor when given NumPy input"
    assert y.shape == (5,)
    assert dtype_name(y) == "float32"
    assert dtype_name(y) == dtype_name(x_np)


# Different kinds of input values we want to test
def make_input(kind, shape):
    x_np = np.random.normal(size=shape).astype(np.float32)
    if kind == "numpy_array":
        return x_np
    elif kind == "numpy_scalar":
        return np.float32(x_np.flat[0])
    elif kind == "python_list":
        return x_np.tolist()
    elif kind == "python_scalar":
        return float(x_np.flat[0])
    elif kind == "tensor":
        return ops.array(x_np, dtype="float32")
    else:
        raise ValueError(kind)

# @pytest.mark.parametrize("quantizer_name", QUANTIZERS)
# @pytest.mark.parametrize("shape", [(8,), (2, 3)])
# @pytest.mark.parametrize("input_kind", ["numpy_array", "numpy_scalar", "python_list", "python_scalar", "tensor"])
# def test_quantizer_accepts_various_input_types(quantizer_name, shape, input_kind):
#     q = _instantiate(quantizer_registry.lookup_quantizer(quantizer_name))
#     x = make_input(input_kind, shape)

#     y = q(x)

#     # Check result is tensor-like
#     assert ops.is_tensor(y), f"{quantizer_name}({input_kind}) did not return a tensor"

#     # Shape check (scalars may broadcast to shape)
#     if np.ndim(x) == 0 or (isinstance(x, (list, float)) and not hasattr(x, "shape")):
#         # Scalar input should produce scalar-shaped tensor ()
#         assert y.shape == (), f"{quantizer_name}({input_kind}) expected scalar output, got {y.shape}"
#     else:
#         assert y.shape == tuple(shape), f"{quantizer_name}({input_kind}) wrong shape: {y.shape}, expected {shape}"

#     # Dtype check
#     assert dtype_name(y) == "float32", f"{quantizer_name}({input_kind}) returned wrong dtype: {dtype_name(y)}"


if __name__ == "__main__":
    pytest.main([__file__])
