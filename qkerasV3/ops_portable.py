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
