"""Trainable k-input LUT networks (k = 1..6) for Keras 3.

    from qkeras.logic import LUTDense, GroupSum, BinaryThreshold
    from qkeras.logic import extract_netlist, simulate, save_netlist
    from qkeras.logic import convert_to_lut, FanInConstraint     # QKeras teacher -> LUT net

``save_netlist`` writes the framework-neutral ``logicnet-netlist`` JSON; VHDL/Verilog/C,
simplification, synthesis and place & route are done by the ``logicnet-fpga`` backend
(``logicnet hdl net.json -o hdl/``).
"""
from .layers import (BinaryThreshold, GroupSum, LUTDense,  # noqa: F401
                     LUTTemperatureScheduler, lut_layers, set_lut_temperature)
from .netlist import (FORMAT, VERSION, LUT, Netlist, extract_netlist,  # noqa: F401
                      load_netlist, save_netlist, simulate)
from .netlist import encode_inputs, materialize_inputs  # noqa: F401,E402
from .encoders import Thermometer, fixed_point, quantile_thresholds  # noqa: F401,E402
from .convert import (Conversion, DistillConfig, FanInConstraint,  # noqa: F401,E402
                      FanInScheduler, convert_to_lut, fan_in_report)
