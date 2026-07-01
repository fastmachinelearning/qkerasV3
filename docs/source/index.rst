QKerasV3 Documentation
=====================

QKerasV3 is a Keras 3–compatible continuation of QKeras, focused on quantization-aware training and model compression.

.. important::

   QKerasV3 supports **TensorFlow**, **JAX** and **PyTorch** as backend (default is **TensorFlow**).
   Set the backend before importing:

   .. code-block:: bash

      export KERAS_BACKEND=tensorflow/jax/torch

   In the current version v1.2.x AutoQKeras and Pruning are not working.
   There will be an update in the the future to support this feature again.

QKerasV3 Layer Backend Support Matrix
=====================================

The following matrix tracks multi-backend framework support for quantization-aware training (QAT) layers in ``qkerasV3``.

.. list-table:: Layer Support Matrix
   :widths: 30 12 12 12 44
   :header-rows: 1

   * - Layer Name
     - TensorFlow
     - JAX
     - PyTorch
     - Implementation Notes & Constraints
   * - ``QDense``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``QConv1D``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``QConv2D``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``QDepthwiseConv2D``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``QSeparableConv1D``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``QSeparableConv2D``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``QMobileNetSeparableConv2D``
     - ✅ 
     - ✅ 
     - ✅ 
     - MobileNet-specific; explicitly quantizes activation values immediately after the depthwise step. TODO: needs a test.
   * - ``QConv2DTranspose``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``QActivation``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``QAdaptiveActivation``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``QAveragePooling2D``
     - ✅ 
     - ✅ 
     - ⚠️
     - Combines ``AveragePooling2D`` with a ``QActivation`` layer. PyTorch lacks native asymmetric padding (``padding="same"``) for all shapes.
   * - ``QBatchNormalization`` / ``QConv2DBatchnorm``
     - ⚠️
     - ⚠️
     - ⚠️
     - **Experimental Stage:** Stochastic activation functions often offset its regularization needs. JAX/Torch rely on Keras 3 epoch variable updates.
   * - ``QOctaveConv2D``
     - ✅ 
     - ⚠️
     - ⚠️
     - Multi-frequency feature extraction relies on complex tensor splitting and slicing across backends. TODO: needs a test.
   * - ``QSimpleRNN`` / ``QSimpleRNNCell``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``QLSTM`` / ``QLSTMCell``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``QGRU`` / ``QGRUCell``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``QBidirectional``
     - ✅ 
     - ✅ 
     - ✅ 
     - 

**Legend:**

* ✅ **Supported**: Tested and functions smoothly natively across the backend via Keras 3.
* ⚠️ **Partial / Experimental / Conditional**: Functions, but exhibits structural constraints, layout edge cases, or relies on features currently in testing.

QKerasV3 Activation Function Backend Support Matrix
===================================================

The following matrix tracks multi-backend framework support for quantization activation functions in ``qkerasV3``.

.. list-table:: Activation Support Matrix
   :widths: 45 11 11 11 22
   :header-rows: 1

   * - Activation Function
     - TensorFlow
     - JAX
     - PyTorch
     - Implementation Notes & Constraints
   * - ``smooth_sigmoid(x)``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``hard_sigmoid(x)``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``binary_sigmoid(x)``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``binary_tanh(x)``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``smooth_tanh(x)``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``hard_tanh(x)``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``quantized_bits(bits=8, integer=0, symmetric=0, keep_negative=1)(x)``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``bernoulli(alpha=1.0)(x)``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``stochastic_ternary(alpha=1.0, threshold=0.33)(x)``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``ternary(alpha=1.0, threshold=0.33)(x)``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``stochastic_binary(alpha=1.0)(x)``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``binary(alpha=1.0)(x)``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``quantized_relu(bits=8, integer=0, use_sigmoid=0, negative_slope=0.0)(x)``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``quantized_ulaw(bits=8, integer=0, symmetric=0, u=255.0)(x)``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``quantized_tanh(bits=8, integer=0, symmetric=0)(x)``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``quantized_po2(bits=8, max_value=-1)(x)``
     - ✅ 
     - ✅ 
     - ✅ 
     - 
   * - ``quantized_relu_po2(bits=8, max_value=-1)(x)``
     - ✅ 
     - ✅ 
     - ✅ 
     - 

**Legend:**

* ✅ **Supported**: Tested and functions smoothly natively across the backend via Keras 3.
* ⚠️ **Partial / Experimental / Conditional**: Functions, but exhibits structural constraints, layout edge cases, or relies on features currently in testing.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Guides

   examples/index
   notebooks

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

Unsupported Keras 3 Layers & Activations
----------------------------------------

* **``MultiHeadAttention`` / ``GroupQueryAttention`` (Layer)**
* **``ConvLSTM1D`` / ``ConvLSTM2D`` / ``ConvLSTM3D`` (Layer)**
* **``LayerNormalization`` / ``GroupNormalization`` / ``RMSNormalization`` (Layer)**
* **``PReLU`` / ``ELU`` / ``LeakyReLU`` (Layer)**
* **``AlphaDropout`` / ``GaussianNoise`` / ``GaussianDropout`` (Layer)**
* **``mish(x)`` (Activation)**
* **``swish(x)`` / ``gelu(x)`` (Activation)**
* **``exponential(x)`` (Activation)**
* **``silu(x)`` (Activation)**
