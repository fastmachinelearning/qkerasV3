# Copyright 2019 Google LLC
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
"""Export quantizer package."""


from .accumulator_factory import AccumulatorFactory
from .accumulator_impl import (
    FixedPointAccumulator,
    FloatingPointAccumulator,
    IAccumulator,
)
from .divider_factory import IDivider
from .fused_bn_factory import FusedBNFactory
from .merge_factory import MergeFactory
from .multiplier_factory import MultiplierFactory
from .multiplier_impl import (
    Adder,
    AndGate,
    FixedPointMultiplier,
    FloatingPointMultiplier,
    IMultiplier,
    Mux,
    Shifter,
    XorGate,
)
from .qbn_factory import QBNFactory
from .quantizer_factory import QuantizerFactory
from .quantizer_impl import (
    Binary,
    FloatingPoint,
    IQuantizer,
    PowerOfTwo,
    QuantizedBits,
    QuantizedRelu,
    ReluPowerOfTwo,
    Ternary,
)
from .subtractor_factory import ISubtractor
