# Full-model GateMate deployment: design & roadmap

Goal: **one bitstream that runs the whole compressed Timestral student on a single
GateMate CCGM1A1** — embed → (RoPE attention + softmax + gated value mix) → RMSNorm →
SwiGLU-MLP → RMSNorm → decoder-token select → monotone quantile head — with *no host-side
splitting*. Streaming in a context window, streaming out raw-space quantiles.

This replaces stage 1 (`rtl.py`), which only emitted per-layer dense harnesses for timing.

## Target graph (student `B_d16_L1_b3_ci_p70`, dataset kdd)

| symbol | value | note |
|---|---|---|
| V | 3 | variates |
| P | 16 | patch size |
| D | 16 | d_model |
| H | 2 | heads, head_dim hd = 8 |
| M | 32 | MLP hidden |
| L | 16 | context patches |
| h | 4 | horizon patches |
| S | 20 | tokens after forecast-query tokens |
| Q | 3 | quantiles (0.1, 0.5, 0.9) |
| blocks | 1 | `variate_every=0` ⇒ **time attention only** |
| preset | B3 | relu act, hard_sigmoid gate |
| act | relu | `quantized_relu(7,3)` → no LUT |
| gate | hard_sigmoid | `hard_sigmoid` + `quantized_bits(5,0)` → no LUT |

Because `variate_every=0` there is **no variate attention**, and because `act=relu` /
`gate=hard_sigmoid` there are **no SiLU/sigmoid tables**. The only true non-dense
non-linearities left are:

* **softmax** (max-subtract, exp LUT, reciprocal, normalise) — per head, over keys;
* **RMSNorm rsqrt** — one LUT, per token;
* **quantile head softplus** — one LUT, per horizon value × (Q−1).

Everything else is constant-weight dense (da4ml) plus the monotone-quantile cumulative sums.

## B3 fixed-point formats (from `student.PRESETS["B3"]`)

`k`=signed, `i`=integer bits, `f`=fraction bits, `B=k+i+f`.

| tensor | spec | k,i,f | B |
|---|---|---|---|
| activations `a` | `quantized_bits(7,3)` | 1,3,3 | 7 |
| relu out | `quantized_relu(7,3)` | 0,3,4 | 7 |
| weights | `quantized_bits(3,1,auto_po2)` | 1,1,1 | 3 |
| bias | `quantized_bits(8,3)` | 1,3,4 | 8 |
| hard_sigmoid out `sig_out` | `(5,0,keep_negative=0)` | 0,0,5 | 5 |
| mean-square `ms` | `(9,5,keep_negative=0)` | 0,5,4 | 9 |
| rsqrt | `(9,5,keep_negative=0)` | 0,5,4 | 9 |
| gain | `(5,2)` | 1,2,2 | 5 |
| score | `(7,4)` | 1,4,2 | 7 |
| exp | `(7,1,keep_negative=0)` | 0,1,6 | 7 |
| inv | `(7,1,keep_negative=0)` | 0,1,6 | 7 |
| prob | `(6,1,keep_negative=0)` | 0,1,5 | 6 |

These are *derived*, not assumed — `fixedpoint.py` reads them back from the qkeras
quantizer objects and is the single source of truth the RTL must match.

## Non-dense RTL blocks to build

1. **`rmsnorm`** — square-accumulate over D (fixed-point), `mean`, quantize `ms`, LUT
   `rsqrt(ms+eps)`, multiply by `x` and quantized `gain`, quantize to `a`. One token/cycle
   with a D-deep reduction (D=16). `fixedpoint.rmsnorm_ref` is the golden model.
2. **`softmax`** — per (head, query): max over keys, subtract, `exp` LUT, sum, `inv` LUT,
   multiply, quantize `prob`. Static causal+forecast mask baked as constants (L, h known).
   `fixedpoint.softmax_ref` is the golden model.
3. **`rope`** — partial rotate-half with constant cos/sin tables (S=20, hd=8, fraction 0.5).
   Pure constants → no runtime LUT; folded into wiring.
4. **`quantile_head`** — `median ∓ cumsum(width·softplus(raw))`, softplus LUT, fixed widths
   from `quantile_increment_widths`. `fixedpoint.quantiles_ref` is the golden model.
5. **`token_select` / residual adds / hard_sigmoid gate / relu** — trivial datapath glue.

## Top-level `model_top` architecture

Single streaming design, one variate row per inference (channel-independent embed/MLP,
shared weights). Pipeline:

```
 scaled[V][L*P] ─► patch/embed dense ─► +forecast tokens ─►
   ┌────────────────────── block 0 (time attention) ──────────────────────┐
   │ RMSNorm → WQKV → RoPE → SDPA(softmax) → ⊙gate(hard_sigmoid) → WO → +res │
   │ RMSNorm → W12 → (gate, relu(up)) ⊙ → W3 → +res                          │
   └────────────────────────────────────────────────────────────────────────┘
 out_norm → decoder tokens[h] → head(relu→out→res add) → quantiles → sinh·scale+loc ─► out[V][h*P][Q]
```

Weights are **constant** (da4ml bakes them into wiring). Area budget: stage-1 measured
2 722 CPE for the largest single dense block and 0.85× chip for *all* dense layers summed.
Adding softmax/rmsnorm/head datapath pushes past 1× — so the top-level **must** time-multiplex
the D=16 dot-products through a small shared multiplier array (a handful of DSP/CPE MACs)
rather than instantiating every dense layer fully parallel. This is the central design
decision (see "Dataflow" below).

## Dataflow decision: shared-MAC sequencer vs. fully-parallel

* **Fully parallel** (stage-1 style): correct and simple, but ~0.85× chip *before* the
  non-dense ops → does not fit with attention/softmax/head resident.
* **Shared constant-weight MAC array**: a bank of ~D/2 multipliers time-multiplexed across
  every dense layer, weights read from BRAM. Fits comfortably (target < 0.5× chip) at the
  cost of a micro-sequencer. Chosen path.

The sequencer runs a small instruction list per layer (input addr, weight addr, accumulate,
store) generated from the export metadata, so the Verilog stays data-driven.

## Verification strategy (bit-exact, not "close enough")

Every block is checked in **Verilator against a numpy golden model built from the exact
qkeras quantizers** (`fixedpoint.py`), the same discipline `rtl.py` uses for dense layers.
Then a **full-model co-simulation**: drive the RTL top with a KDD context window, capture
quantiles, compare to `keras_timestral.forecast_mv` on the same window. Gate: bit-exact
integer match at the RTL interface; < 1e-2 normalised MAE after de-normalisation as a
secondary sanity bound.

## Roadmap (phased, each phase independently verified)

- **P0 — foundation (this commit).** `fixedpoint.py`: exact quantizer emulation + format
  derivation + golden references for rmsnorm / softmax / rope / quantiles. Unit-tested
  against the loaded model's own layers.
- **P1 — non-dense RTL.** Verilog generators for `rmsnorm`, `softmax`, `rope`,
  `quantile_head`; Verilator bit-exact vs. P0 references.
- **P2 — glue + sequencer.** relu, hard_sigmoid, residual add, token select, shared-MAC
  sequencer, BRAM weight loading from `export` metadata.
- **P3 — `model_top` assembly + full-model co-sim** against `forecast_mv`.
- **P4 — GateMate P&R of `model_top`, fix timing (target ≥ 50 MHz), emit `.bit`** via
  `nextpnr --vopt out=…` + `gmpack` (confirmed working; use native nextpnr to avoid the
  WASM OOM seen on the largest layers).
- **P5 — board bring-up:** `.ccf` pins, host link (UART/SPI) to stream windows, on-board
  quantile readout.

## Honest status

P0 is implemented now. P1–P5 are substantial (multi-session). This document is the
contract; each phase lands with its own Verilator proof before moving on.
