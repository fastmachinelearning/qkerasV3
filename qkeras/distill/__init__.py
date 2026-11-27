"""Knowledge distillation of large models into small quantised students for FPGAs.

Workflow
--------
1. Write the model builder once against a ``LayerFactory`` (``factory``) so teacher, float
   student and QAT student share one graph (layer names identical).
2. Cache teacher outputs on the training data (``teacher.cached``).
3. ``trainer.distill``: float student with a distillation loss (``losses``), then QAT (and
   optionally magnitude pruning) with a preset of quantizers (``presets``).
4. Estimate / implement the hardware (``hw``: GateMate cost model, bit-exact fixed-point refs).

Modules
-------
presets   quantizer presets per tensor role (A8/A6 folded, B4/B3 streamed, W8A14) + registry
layers    float transformer blocks (RMSNorm, Rotary, StaticMaskAttention, AddToken, MonotoneQuantiles)
factory   LayerFactory / FloatFactory / QuantFactory / make_factory
weights   transfer_weights, quantized_kernels, dense_sparsity
losses    QuantileDistillLoss, LogitDistillLoss, FeatureDistillLoss, pack_targets
teacher   Teacher protocol, KerasTeacher, on-disk soft-label cache
trainer   fit_phase, prune_qat, distill (float -> QAT [-> pruning])
hw        gatemate (device + dense CPE estimate), fixedpoint (integer golden models)

See ``examples/timestral`` for a full example (Timestral-5M -> GateMate students).
"""

from .factory import FloatFactory, LayerFactory, QuantFactory, make_factory  # noqa: F401
from .losses import (FeatureDistillLoss, LogitDistillLoss, QuantileDistillLoss, gt_pinball_metric,  # noqa: F401
                     pack_targets)
from .presets import PRESETS, is_quantized, preset_bits, register_preset, resolve  # noqa: F401
from .teacher import KerasTeacher, cache_key, cached  # noqa: F401
from .trainer import TRAIN_DEFAULTS, distill, fit_phase, prune_qat  # noqa: F401
from .weights import dense_sparsity, quantized_kernels, transfer_weights  # noqa: F401
