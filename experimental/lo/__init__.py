# Copyright 2020 Google LLC
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
"""Exports logic optimization module."""

from .compress import Compressor
from .conv2d import optimize_conv2d_logic
from .dense import optimize_dense_logic
from .generate_rf_code import *
from .optimizer import mp_rf_optimizer_func, run_abc_optimizer, run_rf_optimizer
from .receptive import model_to_receptive_field
from .table import load
from .utils import *  # pylint: disable=wildcard-import
# __version__ = "0.5.0"
