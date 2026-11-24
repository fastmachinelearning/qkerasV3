"""Distillation of Timestral into small quantised multivariate students for FPGAs.

Modules
-------
data       multivariate datasets (coupled synthetic, KDD Cup 2018, process CSV), windows, metrics
teacher    Timestral-5M teacher forecasts cached as soft labels
student    student specification, float/QAT layer factories, size and MAC counts
losses     pinball + distillation losses
fpga_cost  first-order GateMate cost model (folded/BRAM and token-streamed constant-weight)
train      CLI: train one student config (float distillation -> QAT) and evaluate
sweep      CLI: run a list of configs and collect results
"""
