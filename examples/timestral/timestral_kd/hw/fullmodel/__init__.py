"""Full-model GateMate design (work in progress, see docs/FULLMODEL.md).

fixedpoint  P0  bit-exact integer references in the B3 formats (+ quantile head), validated vs qkeras
blocks      P1  Verilog for rmsnorm / softmax / rope / quantile head, Verilator-checked vs P0
sequencer   P2  glue (relu, hard_sigmoid, residual, select) + shared-MAC sequencer
model_top   P3a integer golden model of the whole student vs the QAT model
"""
