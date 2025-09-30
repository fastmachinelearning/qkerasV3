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
"""Example the usage of activation functions in qkerasV3."""

import numpy as np
import tensorflow as tf
from qkerasV3 import (
    bernoulli,
    binary,
    hard_sigmoid,
    hard_tanh,
    quantized_bits,
    quantized_po2,
    quantized_relu,
    quantized_relu_po2,
    quantized_tanh,
    set_internal_sigmoid,
    smooth_sigmoid,
    smooth_tanh,
    stochastic_binary,
    stochastic_ternary,
    ternary,
)


def main():
    # check the mean value of samples from stochastic_rounding for po2
    np.random.seed(42)
    count = 100000
    val = 42
    a = tf.constant([val] * count)
    b = quantized_po2(use_stochastic_rounding=True)(a)
    res = np.sum(b.numpy()) / count
    print(res, "should be close to ", val)
    b = quantized_relu_po2(use_stochastic_rounding=True)(a)
    res = np.sum(b.numpy()) / count
    print(res, "should be close to ", val)
    a = tf.constant([-1] * count)
    b = quantized_relu_po2(use_stochastic_rounding=True)(a)
    res = np.sum(b.numpy()) / count
    print(res, "should be all ", 0)

    # non-stochastic rounding quantizer.
    a = tf.constant([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])
    a = tf.constant([0.194336])
    print(" a =", a.numpy().astype(np.float16))
    print("qa =", quantized_relu(6, 2)(a).numpy().astype(np.float16))
    print("ss =", smooth_sigmoid(a).numpy().astype(np.float16))
    print("hs =", hard_sigmoid(a).numpy().astype(np.float16))
    print("ht =", hard_tanh(a).numpy().astype(np.float16))
    print("st =", smooth_tanh(a).numpy().astype(np.float16))
    c = tf.constant(np.arange(-1.5, 1.51, 0.3), dtype=tf.float32)
    print(" c =", c.numpy().astype(np.float16))
    print("qb_111 =", quantized_bits(1, 1, 1)(c).numpy().astype(np.float16))
    print("qb_210 =", quantized_bits(2, 1, 0)(c).numpy().astype(np.float16))
    print("qb_211 =", quantized_bits(2, 1, 1)(c).numpy().astype(np.float16))
    print("qb_300 =", quantized_bits(3, 0, 0)(c).numpy().astype(np.float16))
    print("qb_301 =", quantized_bits(3, 0, 1)(c).numpy().astype(np.float16))

    c_1000 = tf.constant(np.array([list(c.numpy())] * 1000), dtype=tf.float32)
    b = np.sum(bernoulli()(c_1000).numpy().astype(np.int32), axis=0) / 1000.0
    print("       hs =", hard_sigmoid(c).numpy().astype(np.float16))
    print("    b_all =", b.astype(np.float16))

    T = 0.0
    t = stochastic_ternary(alpha="auto")(c_1000).numpy()
    for i in range(10):
        print(f"stochastic_ternary({i}) =", t[i])
    print(
        "   st_all =",
        np.round(
            np.sum(t.astype(np.float32), axis=0).astype(np.float16) / 1000.0, 2
        ).astype(np.float16),
    )

    print("  ternary =", ternary(threshold=0.5)(c).numpy().astype(np.int32))
    print(" c =", c.numpy().astype(np.float16))
    print(" b_10 =", binary(1)(c).numpy().astype(np.float16))
    print("qr_10 =", quantized_relu(1, 0)(c).numpy().astype(np.float16))
    print("qr_11 =", quantized_relu(1, 1)(c).numpy().astype(np.float16))
    print("qr_20 =", quantized_relu(2, 0)(c).numpy().astype(np.float16))
    print("qr_21 =", quantized_relu(2, 1)(c).numpy().astype(np.float16))
    print("qr_101 =", quantized_relu(1, 0, 1)(c).numpy().astype(np.float16))
    print("qr_111 =", quantized_relu(1, 1, 1)(c).numpy().astype(np.float16))
    print("qr_201 =", quantized_relu(2, 0, 1)(c).numpy().astype(np.float16))
    print("qr_211 =", quantized_relu(2, 1, 1)(c).numpy().astype(np.float16))
    print("qt_200 =", quantized_tanh(2, 0)(c).numpy().astype(np.float16))
    print("qt_210 =", quantized_tanh(2, 1)(c).numpy().astype(np.float16))
    print("qt_201 =", quantized_tanh(2, 0, 1)(c).numpy().astype(np.float16))
    print("qt_211 =", quantized_tanh(2, 1, 1)(c).numpy().astype(np.float16))

    set_internal_sigmoid("smooth")
    print("with smooth sigmoid")
    print("qr_101 =", quantized_relu(1, 0, 1)(c).numpy().astype(np.float16))
    print("qr_111 =", quantized_relu(1, 1, 1)(c).numpy().astype(np.float16))
    print("qr_201 =", quantized_relu(2, 0, 1)(c).numpy().astype(np.float16))
    print("qr_211 =", quantized_relu(2, 1, 1)(c).numpy().astype(np.float16))
    print("qt_200 =", quantized_tanh(2, 0)(c).numpy().astype(np.float16))
    print("qt_210 =", quantized_tanh(2, 1)(c).numpy().astype(np.float16))
    print("qt_201 =", quantized_tanh(2, 0, 1)(c).numpy().astype(np.float16))
    print("qt_211 =", quantized_tanh(2, 1, 1)(c).numpy().astype(np.float16))

    set_internal_sigmoid("real")
    print("with real sigmoid")
    print("qr_101 =", quantized_relu(1, 0, 1)(c).numpy().astype(np.float16))
    print("qr_111 =", quantized_relu(1, 1, 1)(c).numpy().astype(np.float16))
    print("qr_201 =", quantized_relu(2, 0, 1)(c).numpy().astype(np.float16))
    print("qr_211 =", quantized_relu(2, 1, 1)(c).numpy().astype(np.float16))
    print("qt_200 =", quantized_tanh(2, 0)(c).numpy().astype(np.float16))
    print("qt_210 =", quantized_tanh(2, 1)(c).numpy().astype(np.float16))
    print("qt_201 =", quantized_tanh(2, 0, 1)(c).numpy().astype(np.float16))
    print("qt_211 =", quantized_tanh(2, 1, 1)(c).numpy().astype(np.float16))

    set_internal_sigmoid("hard")
    print(" c =", c.numpy().astype(np.float16))
    print("q2_31 =", quantized_po2(3, 1)(c).numpy().astype(np.float16))
    print("q2_32 =", quantized_po2(3, 2)(c).numpy().astype(np.float16))
    print("qr2_21 =", quantized_relu_po2(2, 1)(c).numpy().astype(np.float16))
    print("qr2_22 =", quantized_relu_po2(2, 2)(c).numpy().astype(np.float16))
    print("qr2_44 =", quantized_relu_po2(4, 1)(c).numpy().astype(np.float16))

    print("q2_32_2 =", quantized_relu_po2(32, 2)(c).numpy().astype(np.float16))

    b = stochastic_binary()(c_1000).numpy().astype(np.int32)
    for i in range(5):
        print(f"sbinary({i}) =", b[i])
    print("sbinary =", np.round(np.sum(b, axis=0) / 1000.0, 2).astype(np.float16))
    print(" binary =", binary()(c).numpy().astype(np.int32))
    print(" c      =", c.numpy().astype(np.float16))

    for i in range(10):
        print(
            f" s_bin({i}) =",
            binary(use_stochastic_rounding=1)(c).numpy().astype(np.int32),
        )
    for i in range(10):
        print(
            f" s_po2({i}) =",
            quantized_po2(use_stochastic_rounding=1)(c).numpy().astype(np.int32),
        )
    for i in range(10):
        print(
            f" s_relu_po2({i}) =",
            quantized_relu_po2(use_stochastic_rounding=1)(c).numpy().astype(np.int32),
        )


if __name__ == "__main__":
    main()
