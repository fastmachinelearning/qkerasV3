"""Trainable k-input LUT networks (k = 1..6) for Keras 3.

    from qkeras.logic import LUTDense, GroupSum, BinaryThreshold
    from qkeras.logic import extract_netlist, simulate, save_netlist

``save_netlist`` writes the framework-neutral ``logicnet-netlist`` JSON; VHDL/Verilog/C,
simplification, synthesis and place & route are done by the ``logicnet-fpga`` backend
(``logicnet hdl net.json -o hdl/``).
"""
from .layers import (BinaryThreshold, GroupSum, LUTDense,  # noqa: F401
                     LUTTemperatureScheduler, lut_layers, set_lut_temperature)
from .netlist import (FORMAT, VERSION, LUT, Netlist, extract_netlist,  # noqa: F401
                      load_netlist, save_netlist, simulate)
