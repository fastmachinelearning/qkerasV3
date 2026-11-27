# Results so far

Numbers from the first runs (2026-11-24, KDD Cup 2018 and the synthetic process). The run
directories of that time were removed; re-create them with the pipeline in the README
(`python -m timestral_kd sweep sweep_v1 --dataset kdd`, `... export/pnr/report kdd/B_d16_L1_b3`).

## Place & route, stage 1 (dense layers only): dense layers of `B_d16_L1_b3` (2026-11-24)

```bash
python -m timestral_kd export kdd/B_d16_L1_b3
python -m timestral_kd pnr    kdd/B_d16_L1_b3 --jobs 7
python -m timestral_kd report kdd/B_d16_L1_b3
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
