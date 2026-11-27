"""Timestral-5M -> GateMate students: the worked example of ``qkeras.distill``.

Entry point: ``python -m timestral_kd`` (see ``__main__``).

Model and data
    model     Keras 3 port of Timestral (spec, builders taking a qkeras.distill factory, weights, scaling)
    data      synthetic coupled process, KDD Cup 2018, process CSV; windows; MASE / WQL / coverage
    teacher   Timestral-5M forecasts at the student's quantile knots, cached; student-frame transforms
    student   StudentConfig, build_student, analytic parameter / MAC / lookup counts
    cost      first-order GateMate estimate of a student (class A folded, class B streamed)
    paths     where inputs and outputs live (runs/, checkpoint, KDD download)

Pipeline steps (each also runnable as ``python -m timestral_kd.<module>``)
    prepare   data bundle: windows + teacher labels
    train     float distillation -> QAT [-> pruning] -> metrics + cost
    sweep     many students in parallel subprocesses
    plots     Pareto / ablation figures

Hardware
    hw.dense      da4ml Verilog per dense layer, GateMate P&R, summary
    hw.fullmodel  single-bitstream design in progress (P0-P3a)
"""
