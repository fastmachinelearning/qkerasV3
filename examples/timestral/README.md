# Timestral-5M -> GateMate: a `qkeras.distill` example

Timestral-5M (5.2 M parameters, about 21 Mbit at 4 bit) does not fit a small FPGA such as the
Cologne Chip GateMate CCGM1A1 (20,480 CPEs, about 1.3 Mbit BRAM, no DSPs). This example
distils it into small Timestral-style students, trains them quantisation-aware (and pruned)
with `qkeras.distill`, and takes the dense layers through GateMate place & route.

```
dataset (V variates x T) -> windows (teacher ctx 512, student ctx 256, horizon 64)
                                 |
Timestral-5M teacher  ->  soft labels (9 deciles, cached)
                                 |
student: float distillation -> QAT (B3/B4 preset) -> magnitude pruning
                                 |
da4ml Verilog per dense layer -> yosys synth_gatemate + nextpnr (CCGM1A1) -> report
```

## Setup

```bash
pip install -e .                            # qkeras with qkeras.distill (repo root, once)
pip install -e "examples/timestral[hw]"     # this example (+ da4ml, nextpnr for the hardware steps)
export YOSYS=/path/to/yosys                 # >= 0.40 with synth_gatemate (hardware steps)
export KOLIBRI_FPGA_SRC=~/cms/logicnet-fpga/src   # Kolibri harness (a sibling checkout is found automatically)
```

`python -m timestral_kd` then works from any directory; outputs always go to `examples/timestral/runs/`
(`$TIMESTRAL_KD_RUNS` to move them).

The teacher checkpoint is read from `<repo>/ts/Timestral-5M` (`config.json`, `model.safetensors`);
set `TIMESTRAL_CKPT` to use another location.

## Pipeline

```bash
python -m timestral_kd data   kdd                                # 1. windows + teacher labels  (~1 min, downloads KDD once)
python -m timestral_kd train  B_d12_L1_b3_p70_ci --dataset kdd   # 2. float -> QAT -> pruning   (~minutes on a laptop)
python -m timestral_kd export kdd/B_d12_L1_b3_p70_ci             # 3. da4ml Verilog, bit-exact vs numpy
python -m timestral_kd pnr    kdd/B_d12_L1_b3_p70_ci --jobs 7    # 4. GateMate P&R per layer (seeds 1-3)
python -m timestral_kd report kdd/B_d12_L1_b3_p70_ci             # 5. per-layer CPE / Fmax + fit summary
```

`python -m timestral_kd all B_d12_L1_b3_p70_ci --dataset kdd` runs steps 1-5. Further commands:

| command | does |
|---|---|
| `data synthetic` | coupled synthetic process instead of KDD |
| `data process --csv FILE --columns A B ...` | your own semicolon CSV as bundle `process` (then `--dataset process`) |
| `train smoke --dataset kdd` | 1-minute sanity run |
| `sweep prune_v1 --dataset kdd --jobs 4` | every student of `configs/sweeps/prune_v1.json`, plus `student_sweep.csv` |
| `plots kdd` | accuracy vs on-chip memory and ablation figures of the sweep |
| `export RUN --digit-bits 2` / `pnr RUN --digit-bits 2` | digit-serial (time-multiplexed) dense blocks instead of fully parallel |
| `verify [--rtl]` | full-model fixed-point references vs qkeras (+ Verilator checks of the RTL blocks) |
| `deploy RUN [--verify 6] [--studio-url URL]` | **whole student** as one sequential engine + Kolibri Studio LiteX project (`runs/deploy/RUN`) |
| `board RUN --port /dev/ttyACM0` / `board RUN --offline` | forecasts on the GateMate (or through the engine simulator) and their MASE / WQL |

Unknown options are passed to the underlying module, e.g. `train ... --sparsity 0.85 --seed 1`
(`python -m timestral_kd.train -h`).

### Outputs (`runs/`, git-ignored)

```
runs/data/<dataset>.npz, <dataset>_info.json   bundle + teacher / seasonal-naive scores
runs/students/<dataset>/<name>/                config.json, float/qat.weights.h5, metrics.json, history.json
runs/rtl/<dataset>/<name>/                     export.json, <layer>/src/*.v, <layer>/pr_s<seed>/, pnr_layers.csv, deploy.json
runs/deploy/<dataset>/<name>/                  nn.json, rtl/engine.v + *.mem, firmware/ (selftest.h), soc.py, codec.npz
```

`metrics.json` has test MASE / WQL / P10-P90 coverage of the float, post-training-quantised and QAT
student next to the teacher and seasonal naive, the sparsity and the GateMate cost estimate.
`deploy.json` says whether all dense layers fit one bitstream (`sum_cpe`) or only one layer at a time
(`peak_cpe`).

## Students

A student config (`configs/students/*.json`) has a `student` and a `train` block.

`student`: `patch_size`, `d_model`, `num_heads`, `mlp_hidden`, `n_blocks`, `variate_every` (0 =
channel-independent, 1 = variate attention in every block), `embed_hidden`, `head_hidden`, `ctx_steps`,
`hor_steps`, `knots` (e.g. `[0.1, 0.5, 0.9]`), `preset`, `deployment` (`A` folded / `B` streamed),
`act_fn` (`silu` | `relu`), `gate_fn` (`sigmoid` | `hard_sigmoid`).

`train`: `float_epochs`, `qat_epochs`, `lr`, `qat_lr`, `alpha` (1 = no distillation), `batch_size`,
`patience`, `prune: {"sparsity": 0.7, "ramp_frac": 0.5}`.

For P&R use a class-B student: preset `B3` / `B4`, `relu` + `hard_sigmoid` (no lookup tables).

| preset | weights | activations | element-wise tables | intended for |
|---|---|---|---|---|
| A8 | 8 bit | 16 bit | 12 bit in/out | folded, weights in BRAM |
| A6 | 6 bit (po2 per-channel scale) | 12 bit | 10 bit | folded |
| B4 | 4 bit (po2 scale) | 8 bit | 7 bit in / 8 bit out | streamed constant-weight |
| B3 | 3 bit (po2 scale) | 7 bit | 6 / 7 bit | streamed constant-weight |

Presets live in `qkeras.distill.presets` (`register_preset` adds your own).

### GateMate cost model (`timestral_kd/cost.py`, `qkeras/distill/hw/gatemate.py`)

These are first-order estimates for ranking configurations, not synthesis results. Every constant is
editable in `qkeras.distill.hw.gatemate.GATEMATE`:

* 4-LUT about 1.83 CPEs (measured in the logicnet-fpga paper), 1 CPE per adder bit, 50 MHz,
  32 x 40 kbit BRAM, 80 % utilisation limit.
* Class A (folded): BRAM = weight bits + function tables + double-buffered activations and K/V;
  CPEs = `n_mac` (16) multiply-accumulate units + 1.5k control; latency = MACs / n_mac / f.
* Class B (streamed): constant-weight shift-add blocks, one per dense layer. With da4ml installed:
  1.59 CPE per da4ml adder-cost unit; without: 0.56 x CSD digits x (input bits + weight bits +
  log2 fan-in). Both are calibrated on the place-and-route run in `docs/RESULTS.md` (max error 16 % / 23 %). Plus one table per feature lane + activation x activation multipliers for
  attention and RMSNorm. `min_reuse_to_fit` is the smallest reuse factor (time-multiplexing of all
  per-token units, 25 % mux overhead) that fits 80 % of the CPEs.

The class-A numbers and the non-dense class-B terms (lane tables, attention/norm multipliers, reuse
overhead) are still unvalidated first-order assumptions.

## Layout

```
configs/students/    one student per file           configs/sweeps/   lists of students
docs/RESULTS.md      first sweep + stage-1 P&R      docs/FULLMODEL.md single-bitstream design / roadmap
timestral_kd/
  __main__.py        the pipeline CLI                paths.py          runs/, checkpoint, KDD download
  model.py           Keras port of Timestral         data.py           datasets, windows, metrics
  teacher.py         cached teacher labels           student.py        StudentConfig, build_student, op counts
  cost.py            GateMate estimate               prepare/train/sweep/plots.py   the pipeline steps
  hw/dense.py        per-layer Verilog + P&R         deploy.py / board.py   Kolibri Studio project / host client
  hw/fullmodel/      program.py (exact integer program), isa.py (engine compiler + simulator),
                     engine_rtl.py (Verilog), engine_verify.py (Verilator); blocks/sequencer/model_top = older P1-P3a
```

Everything model-agnostic (factories, presets, losses, trainer, GateMate primitives, fixed-point
references) is in `qkeras/distill`; this folder only adds what is specific to Timestral.

## Status

The whole student runs on the GateMate as one sequential tensor engine (`docs/FULLMODEL.md`):
`deploy` lowers the trained student to an exact integer program, compiles it to 69 engine
instructions and writes the Verilog plus a Kolibri Studio LiteX project (VexRiscv firmware streams one
variate per engine run). Integer program == instruction simulator == engine RTL (Verilator, bit-exact);
on the 866 KDD test windows the integer program scores MASE 0.8973 / WQL 0.3723 vs 0.8975 / 0.3724 for
the Keras QAT model. The per-layer flow above (`export` / `pnr`) stays for fully parallel dense layers.

## Notes

* Metrics are computed on chronologically later test windows. MASE uses the student's 256-step context
  as its scale, so teacher numbers here differ slightly from `prepare.py` (512-step scale).
* WQL is averaged over the student's own knots: compare WQL only between students with the same knots.
* Weights: Timestral-5M is released by Mistral AI under a Modified MIT licence. Companies with more than
  $20 M monthly revenue need a commercial licence, which also covers distilled derivatives. Check before
  deploying students commercially.
