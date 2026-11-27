# Distilling Timestral-5M into GateMate-sized multivariate students (`qkeras.distill` example)

Timestral-5M (5.2 M parameters, about 21 Mbit at 4 bit) does not fit a small FPGA such as the
Cologne Chip GateMate CCGM1A1 (20,480 CPEs, about 1.3 Mbit BRAM, no DSPs). This package
trains small Timestral-style multivariate transformers with quantisation-aware training
(qkeras + `qkeras.qtransformer`). Timestral-5M serves as the teacher: its quantile forecasts
are additional targets.

```
data (V variates x T)  ->  windows (teacher ctx 512, student ctx 256, horizon 64)
        |                                    |
Timestral-5M (Keras port, multivariate)  -> soft labels (cached, 9 deciles)
                                             |
student (float)  --distillation-->  student (QAT preset)  ->  metrics + GateMate cost estimate
```

## Layout

The generic machinery is the library `qkeras.distill`; this directory is the worked example on top of it.

| `qkeras/distill/` (library) | purpose |
|---|---|
| `presets.py` | quantizer presets per tensor role (`A8 A6` folded, `B4 B3` streamed, `W8A14`), `register_preset` |
| `layers.py` | float transformer blocks: `RMSNorm`, `Rotary`, `StaticMaskAttention`, `AddToken`, `MonotoneQuantiles` |
| `factory.py` | `LayerFactory` (teacher) / `FloatFactory` / `QuantFactory` (QAT, pruning, activation substitution) |
| `weights.py` | `transfer_weights` (float <-> QAT <-> pruned), `dense_sparsity` |
| `losses.py` | `QuantileDistillLoss`, `LogitDistillLoss` (classifiers / LMs), `FeatureDistillLoss`, `pack_targets` |
| `teacher.py` | `Teacher` protocol, `KerasTeacher`, on-disk soft-label cache |
| `trainer.py` | `distill`: float distillation -> QAT [-> magnitude-pruning ramp] |
| `hw/gatemate.py` | GateMate constants, calibrated constant-weight dense CPE estimate (da4ml / CSD) |
| `hw/fixedpoint.py` | quantizer-format derivation, bit-exact integer rmsnorm / softmax / rope references |

| `examples/timestral/` | purpose |
|---|---|
| `timestral_kd/model.py` | Keras 3 port of Timestral-5M (builders take a factory), weight loading, scaling |
| `timestral_kd/data.py` | coupled synthetic process, KDD Cup 2018, process CSV; chronological splits, windows, MASE/WQL |
| `timestral_kd/teacher.py` | teacher forecasts at the student's knots (cached), student-frame transforms |
| `timestral_kd/student.py` | `StudentConfig`, `build_student`, analytic parameter / MAC / lookup counts |
| `timestral_kd/cost.py` | student-level GateMate estimate (class A folded, class B streamed + reuse) |
| `timestral_kd/{prepare,train,sweep,plots}.py` | CLIs |
| `timestral_kd/hw/` | per-layer RTL + P&R (`rtl`), P0-P3a full-model flow (`fixedpoint`, `blocks`, `sequencer`, `model_top`) |
| `timestral_kd/legacy.py` | keyword adapter for the first PTQ / range-analysis scripts |
| `scripts/` | first experiments: `forecast_csv.py` (was `multi_time.py`), `finetune.py`, `quant_sweep.py` (was `test.py`), `prune_range.py` (was `range_ana.py`), `export_da4ml.py`, `export_hls.py` (was `export_fpga_hls.py`) |
| `configs/` | student and sweep configs |
| `docs/FULLMODEL.md` | full-model GateMate design and roadmap |

Outputs go to `runs/` (git-ignored); Verilator builds of `hw.blocks` / `hw.sequencer` to `runs/hw/` (override with
`TIMESTRAL_KD_BUILD`).

## Distilling another model with `qkeras.distill`

1. Write the builder once, creating layers only via `f.dense / f.norm / f.act / f.attention / f.post`
   (see `timestral_kd/model.py` or `tests/distill_test.py::tiny_transformer`). The teacher uses `LayerFactory()`.
2. Port / load the teacher weights, run it once over your data and cache the outputs
   (`qkeras.distill.cached`).
3. Pack targets (`pack_targets(truth, teacher)`) and pick a loss: `QuantileDistillLoss` (probabilistic
   regression), `LogitDistillLoss` (classification, next-token logits), `FeatureDistillLoss` (point outputs).
4. `distill(lambda preset, pruner=None: builder(..., factory=make_factory(preset, act_map, pruner=pruner)),
   "B3", train, val, loss, metrics, {"float_epochs": .., "qat_epochs": .., "prune": {"sparsity": 0.7}})`.
5. Rank configurations with `qkeras.distill.hw.gatemate.estimate_dense_cpe` and verify the arithmetic of the
   non-dense ops with `qkeras.distill.hw.fixedpoint` before writing RTL.

## Quick start

```bash
cd examples/timestral
export PYTHONPATH=$PWD:$PWD/../.. KERAS_BACKEND=tensorflow
# 1. data bundles + teacher labels (about 1 min per dataset on a laptop CPU)
python -m timestral_kd.prepare --dataset kdd --kdd-parquet data/monash_kdd_cup_2018.parquet --ckpt ../../ts/Timestral-5M --out runs/data
python -m timestral_kd.prepare --dataset synthetic --ckpt ../../ts/Timestral-5M --out runs/data
# your data (semicolon CSV like multi_time.py):
python -m timestral_kd.prepare --dataset csv --csv process_data.csv --columns "" --ckpt ../../ts/Timestral-5M --out runs/data
# 2. one student
python -m timestral_kd.train --config configs/smoke.json --data runs/data/kdd.npz --out runs/students/smoke
# 3. a sweep
python -m timestral_kd.sweep --sweep configs/sweep_v1.json --data runs/data/kdd.npz --out runs/students/kdd --jobs 4
```

KDD parquet: `https://huggingface.co/datasets/autogluon/chronos_datasets/resolve/main/monash_kdd_cup_2018/train-00000-of-00001.parquet`.
For bigger sweeps, run `timestral_kd.sweep` on taige-server with `--jobs 32 --threads 4`; each job is a separate process.

## Student knobs (`StudentConfig`)

`patch_size`, `d_model`, `num_heads`, `mlp_hidden`, `n_blocks`, `variate_every` (0 = channel-independent,
1 = variate attention in every block), `embed_hidden`, `head_hidden`, `ctx_steps`, `hor_steps`, `knots`
(e.g. `[0.1, 0.5, 0.9]` for tiny models: the quantile head is `patch_size x len(knots)` outputs per token),
`preset` (QAT bit widths), `deployment` (`A` or `B`), `act_fn` (`silu` | `relu`), `gate_fn` (`sigmoid` | `hard_sigmoid`).

Presets (`qkeras.distill.presets.PRESETS`):

| preset | weights | activations | element-wise tables | intended for |
|---|---|---|---|---|
| A8 | 8 bit | 16 bit | 12 bit in/out | folded, weights in BRAM |
| A6 | 6 bit (po2 per-channel scale) | 12 bit | 10 bit | folded |
| B4 | 4 bit (po2 scale) | 8 bit | 7 bit in / 8 bit out | streamed constant-weight |
| B3 | 3 bit (po2 scale) | 7 bit | 6 / 7 bit | streamed constant-weight |

## GateMate cost model (`timestral_kd/cost.py`, `qkeras/distill/hw/gatemate.py`)

These are first-order estimates for ranking configurations, not synthesis results. Every constant is
editable in `qkeras.distill.hw.gatemate.GATEMATE`:

* 4-LUT about 1.83 CPEs (measured in the logicnet-fpga paper), 1 CPE per adder bit, 50 MHz,
  32 x 40 kbit BRAM, 80 % utilisation limit.
* Class A (folded): BRAM = weight bits + function tables + double-buffered activations and K/V;
  CPEs = `n_mac` (16) multiply-accumulate units + 1.5k control; latency = MACs / n_mac / f.
* Class B (streamed): constant-weight shift-add blocks, one per dense layer. With da4ml installed:
  1.59 CPE per da4ml adder-cost unit; without: 0.56 x CSD digits x (input bits + weight bits +
  log2 fan-in). Both are calibrated on the place-and-route run below (max error 16 % / 23 %). Plus one table per feature lane + activation x activation multipliers for
  attention and RMSNorm. `min_reuse_to_fit` is the smallest reuse factor (time-multiplexing of all
  per-token units, 25 % mux overhead) that fits 80 % of the CPEs.

The class-A numbers and the non-dense class-B terms (lane tables, attention/norm multipliers, reuse
overhead) are still unvalidated first-order assumptions.

## Place & route, stage 1: dense layers of `B_d16_L1_b3` (2026-11-24)

```bash
cd examples/timestral
python -m timestral_kd.hw.rtl export  --run runs/students/kdd/B_d16_L1_b3 --out runs/rtl/kdd_B_d16_L1_b3
python -m timestral_kd.hw.rtl pnr     --out runs/rtl/kdd_B_d16_L1_b3 --jobs 7 --kolibri-src ~/cms/logicnet-fpga/src
python -m timestral_kd.hw.rtl collect --out runs/rtl/kdd_B_d16_L1_b3
```

* Each `QDense` gets the exact QAT-quantised kernel and bias. The input format is the quantiser that feeds the layer:
  RMSNorm/attention output `a` = (1,3,3); ReLU (0,3,4); attention output x gate (1,3,8) for `wO`; the linear half of
  `w12` x ReLU for `w3`; scaled patch values (1,3,3) plus missing flags (0,1,0) for the embedding. Outputs keep
  4 fractional bits.
* da4ml 0.6 builds the shift-add block (CMVM solver, `latency_cutoff=5`, 2-3 pipeline stages). Each block is
  bit-exact against numpy on 500 random on-grid inputs.
* The flow is yosys 0.67 `synth_gatemate -luttree -nomx8`, then `yowasp-nextpnr-himbaechel-gatemate` 0.11
  (CCGM1A1, router2, 100 MHz target), inside the Kolibri shift-register harness. CPE = yosys cell estimate
  (L2T5/ADDF = 2 half-CPEs). The harness input/output registers are included.
* `b0_w3` (585 input bits) did not route with seed 1: congestion stayed at 2 overused wires after 9.3k router
  iterations. Seeds 2 and 3 routed in about 5 min; the best seed is listed.

| layer | shape | non-zero weights | da4ml adder cost | CPE | Fmax (MHz) |
|---|---|---|---|---|---|
| `embed_hidden` (embedding hidden) | 32 x 16 | 417 | 1861 | 2,621 | 38.7 |
| `embed_output` (embedding output) | 16 x 16 | 201 | 1254 | 1,856 | 41.6 |
| `embed_residual` (embedding skip) | 32 x 16 | 423 | 1805 | 2,701 | 42.1 |
| `b0_attn0_wQKV` (time attention QKV) | 16 x 48 | 617 | 3148 | 5,225 | 30.6 |
| `b0_attn0_gate` (time attention gate) | 16 x 16 | 217 | 1393 | 2,358 | 35.3 |
| `b0_attn0_wO` (time attention out) | 16 x 16 | 218 | 2042 | 3,486 | 23.0 |
| `b0_attn1_wQKV` (variate attention QKV) | 16 x 48 | 603 | 3222 | 5,292 | 28.9 |
| `b0_attn1_gate` (variate attention gate) | 16 x 16 | 216 | 1315 | 2,206 | 33.4 |
| `b0_attn1_wO` (variate attention out) | 16 x 16 | 212 | 1971 | 3,408 | 32.4 |
| `b0_w12` (MLP in) | 16 x 64 | 825 | 4098 | 6,468 | 24.6 |
| `b0_w3` (MLP out) | 32 x 16 | 384 | 5376 | 8,666 | 20.6 |
| `head_hidden` (head hidden) | 16 x 16 | 216 | 1275 | 2,142 | 30.3 |
| `head_output` (head output) | 16 x 48 | 578 | 3063 | 4,214 | 35.5 |
| `head_residual` (head skip) | 16 x 48 | 569 | 2932 | 4,672 | 30.9 |
| **total** | | 5,696 | 34,756 | **55,316** | min 20.6 |

* **The dense layers alone need about 55k CPEs, 2.7x the CCGM1A1's 20,480.** Sweep v1 estimated 12.6k for the
  same layers. Its CSD count rounded the kernels to one fractional bit, and the per-channel `auto_po2` scales put
  most 3-bit weights below 0.25, so it counted 899 instead of 5,696 non-zero weights. This is fixed in
  `qkeras/distill/hw/gatemate.py` (exact scaling) and recalibrated on the table above.
* With the corrected model, the class-B students need a reuse factor of R = 8 (`B_d16_L1_b3`, `B_d16_L1_b4`) to
  R = 16 (`B_d24_L1_b4`), not R = 2-8 as in the sweep-v1 table above. The reuse/mux model itself is not yet
  validated.
* Real levers: pruning (da4ml cost scales with the CSD digits), fewer output columns (QKV and head are the
  largest), and time-multiplexed columns rather than full replication. Alternatively, class A (weights in BRAM)
  avoids the constant-weight logic entirely.
* Not in stage 1: attention scores/softmax, RMSNorm, gates, the quantile head and the token scheduler.

## Notes

* Metrics are computed on chronologically later test windows. MASE uses the student's 256-step context
  as its scale, so teacher numbers here differ slightly from `prepare.py` (512-step scale).
* WQL is averaged over the student's own knots: compare WQL only between students with the same knots.
* Weights: Timestral-5M is released by Mistral AI under a Modified MIT licence. Companies with more than
  $20 M monthly revenue need a commercial licence, which also covers distilled derivatives. Check before
  deploying students commercially.

## First sweep (sweep_v1, single seed, 2026-11-24)

Test MASE (median forecast, chronologically later windows; MASE scale = the student's 256-step context). Class-A students: 9 deciles; class-B students: P10/P50/P90. 'fits' uses the sweep-v1 cost model (class A: BRAM; class B: smallest reuse factor R that fits 80 % of the CPEs). **The class-B R values below are too optimistic, see the place-and-route section.**

**kdd**: teacher (zero-shot Timestral-5M) MASE 0.847, seasonal naive 1.149

| student | params | preset | variate attn | distillation | float | after QAT | P10-P90 coverage | folded BRAM | streamed: R, latency | fits |
|---|---|---|---|---|---|---|---|---|---|---|
| A_d64_L2 | 182,016 | A6 | yes | yes | 0.869 | 0.852 | 0.79 | 108 % | R=512, 662 us | no |
| A_d48_L2 | 108,864 | A6 | yes | yes | 0.871 | 0.858 | 0.78 | 70 % | R=512, 662 us | yes |
| A_d32_L2 | 54,144 | A8 | yes | yes | 0.875 | 0.864 | 0.78 | 66 % | R=512, 662 us | yes |
| A_d48_L2_ci | 85,728 | A6 | no | yes | 0.868 | 0.859 | 0.78 | 59 % | R=256, 355 us | yes |
| A_d48_L2_nokd | 108,864 | A6 | yes | no | 0.935 | 0.921 | 0.73 | 70 % | R=256, 355 us | yes |
| B_d24_L1_b4 | 14,400 | B4 | yes | yes | 0.883 | 0.893 | 0.79 | 10 % | R=16, 43 us | yes |
| B_d16_L1_b4 | 7,296 | B4 | yes | yes | 0.887 | 0.887 | 0.80 | 6 % | R=8, 34 us | yes |
| B_d16_L1_b3 | 7,296 | B3 | yes | yes | 0.887 | 0.890 | 0.79 | 5 % | R=2, 26 us | yes |
| B_d16_L1_b4_ci | 6,000 | B4 | no | yes | 0.889 | 0.888 | 0.78 | 6 % | R=8, 34 us | yes |
| B_d16_L1_b4_nokd | 7,296 | B4 | yes | no | 0.924 | 0.926 | 0.80 | 6 % | R=8, 34 us | yes |

**synthetic**: teacher (zero-shot Timestral-5M) MASE 2.223, seasonal naive 4.413

| student | params | preset | variate attn | distillation | float | after QAT | P10-P90 coverage | folded BRAM | streamed: R, latency | fits |
|---|---|---|---|---|---|---|---|---|---|---|
| A_d64_L2 | 182,016 | A6 | yes | yes | 1.726 | 1.739 | 0.76 | 122 % | R=512, 1104 us | no |
| A_d48_L2 | 108,864 | A6 | yes | yes | 1.729 | 1.726 | 0.76 | 80 % | R=512, 1104 us | no |
| A_d32_L2 | 54,144 | A8 | yes | yes | 1.748 | 1.725 | 0.77 | 75 % | R=512, 1104 us | yes |
| A_d48_L2_ci | 85,728 | A6 | no | yes | 1.678 | 1.687 | 0.78 | 70 % | R=256, 592 us | yes |
| A_d48_L2_nokd | 108,864 | A6 | yes | no | 1.767 | 1.743 | 0.78 | 80 % | R=512, 1104 us | no |
| B_d24_L1_b4 | 14,400 | B4 | yes | yes | 2.048 | 2.393 | 0.70 | 13 % | R=16, 72 us | yes |
| B_d16_L1_b4 | 7,296 | B4 | yes | yes | 2.240 | 2.427 | 0.71 | 8 % | R=8, 56 us | yes |
| B_d16_L1_b3 | 7,296 | B3 | yes | yes | 2.240 | 2.712 | 0.68 | 7 % | R=4, 48 us | yes |
| B_d16_L1_b4_ci | 6,000 | B4 | no | yes | 2.258 | 3.358 | 0.75 | 8 % | R=8, 56 us | yes |
| B_d16_L1_b4_nokd | 7,296 | B4 | yes | no | 2.235 | 2.384 | 0.80 | 8 % | R=8, 56 us | yes |

Observations (single seed; differences below about 0.01-0.02 MASE are within run-to-run noise):

* KDD: distillation improves both classes (A 0.921 -> 0.858, B 0.926 -> 0.887); the best fitting class-A student is within 1.3 % of the teacher; QAT recovers most of the PTQ loss.
* Synthetic: students trained in-domain beat the zero-shot teacher (about 1.7 vs 2.2), so the teacher term mainly regularises; for class B it does not help (2.43 vs 2.38 without).
* Variate attention gives no consistent gain yet on either dataset (per-target MASE within noise). Class-B channel-independent students degrade most under 4-bit QAT on the synthetic set.
* Next levers: range analysis (`optimize_bitwidths(policy='keep_total')`) after float training, pruning (`qkeras.pruning`) to cut class-B CSD digits, multiple seeds, a synthetic task whose targets are only predictable from other variates, and the process CSV.
