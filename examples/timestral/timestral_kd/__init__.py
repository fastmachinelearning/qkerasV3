"""Timestral-5M -> GateMate students: the worked example of ``qkeras.distill``.

Modules
-------
model     Keras 3 port of Timestral (spec, uni-/multivariate builders, weight loading, scaling)
data      coupled synthetic process, KDD Cup 2018 air quality, process CSV; windows; MASE/WQL/coverage
teacher   Timestral-5M forecasts at the student's knots, cached soft labels, student-frame transforms
student   StudentConfig, build_student (via qkeras.distill.make_factory), analytic op counts
cost      GateMate cost model for a student (class A folded, class B streamed)
legacy    keyword adapter for the first PTQ / range-analysis scripts in ``scripts/``
prepare   CLI: dataset bundle (windows + teacher labels + teacher scores)
train     CLI: float distillation -> QAT [-> pruning] -> metrics + cost (qkeras.distill.trainer)
sweep     CLI: many configs in parallel subprocesses -> student_sweep.csv
plots     CLI: Pareto and ablation figures of a sweep
hw        RTL export, place & route, bit-exact references, Verilog blocks, sequencer, model_top
"""
