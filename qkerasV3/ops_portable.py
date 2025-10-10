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

from collections.abc import Mapping, Sequence

import keras
import numpy as np
from keras import ops as Kops

# recognize common tensor leaves across backends
try:
    import tensorflow as tf
    _tf_tensor = (tf.Tensor, tf.RaggedTensor)
except Exception:
    _tf_tensor = ()
try:
    import torch
    _torch_tensor = (torch.Tensor,)
except Exception:
    _torch_tensor = ()
try:
    import jax
    _jax_tensor = (jax.Array,)  # jax>=0.4
except Exception:
    _jax_tensor = ()

_TENSOR_LEAVES = _tf_tensor + _torch_tensor + _jax_tensor + (np.ndarray, np.number)


def bias_add_portable(x, bias, data_format="channels_last"):
    """
    Portable replacement for bias_add_portable.

    Args:
        x: input tensor, shape (N, H, W, C) if channels_last else (N, C, H, W)
        bias: 1D tensor of shape (C,)
        data_format: "channels_last" or "channels_first"
    """
    if data_format == "channels_last":
        # shape (C,) will broadcast to (N,H,W,C)
        return x + bias
    elif data_format == "channels_first":
        # reshape bias to (1,C,1,1) to broadcast to (N,C,H,W)
        bias_reshaped = keras.ops.reshape(bias, (1, -1, 1, 1))
        return x + bias_reshaped
    else:
        raise ValueError(f"Unsupported data_format: {data_format}")


def to_python_bool_if_possible(x):
    # Already a Python bool/int?
    if isinstance(x, (bool, int, np.bool_)):
        return bool(x)
    # Try to fold constant scalars in eager mode
    try:
        v = Kops.convert_to_numpy(x)   # works only in eager; raises/returns array in graph
        if np.ndim(v) == 0:
            return bool(v)
    except Exception:
        pass
    # Could not fold -> keep as tensor (safe for BN's `training=` arg)
    return x


def constant_bool_value(x):
    """Return a Python bool if x is statically known, else None."""
    # Plain Python / NumPy bools
    if isinstance(x, (bool, int, np.bool_)):
        return bool(x)
    # Try to fold constant scalar tensors in eager
    try:
        # works in eager; may raise in graph
        v = keras.ops.convert_to_numpy(x)
        if np.ndim(v) == 0:
            return bool(v)
    except Exception:
        pass

def is_nested(x):
    if isinstance(x, (str, bytes)):
        return False
    if isinstance(x, _TENSOR_LEAVES):
        return False
    return isinstance(x, (Mapping, Sequence))
