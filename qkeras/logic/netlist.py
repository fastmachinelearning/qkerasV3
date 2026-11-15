"""Boolean netlist of a trained LUT network: extraction, neutral file format,
bit-exact simulation.

HDL / C generation, netlist simplification, synthesis and place & route live in the
separate ``logicnet-fpga`` backend, which reads the file written by :func:`save_netlist`.

A :class:`Netlist` is strictly layered: LUTs of layer ``L`` read outputs of
layer ``L-1`` (layer ``-1`` = primary inputs).  Each LUT is ``(pins, table)``:
``pins[j]`` is the source index driving address bit ``j`` and ``table[a]`` the
output for address ``a``.  ``k = len(pins)`` may differ per LUT after
:func:`simplify` (k = 0 is a constant).
"""

import json
from dataclasses import dataclass, field

import numpy as np
import keras

from .layers import BinaryThreshold, GroupSum, LUTDense

__all__ = ["LUT", "Netlist", "extract_netlist", "simulate", "save_netlist", "load_netlist",
           "FORMAT", "VERSION"]

FORMAT, VERSION = "logicnet-netlist", 1


@dataclass
class LUT:
    pins: tuple
    table: np.ndarray            # bool, length 2**len(pins)

    @property
    def k(self):
        return len(self.pins)

    def init_int(self):
        """Truth table as integer, bit a = output for address a."""
        return int(sum(1 << int(a) for a in np.flatnonzero(self.table)))

    def is_const(self):
        return self.k == 0


@dataclass
class Netlist:
    n_inputs: int
    layers: list                       # list[list[LUT]]
    num_classes: int = None            # GroupSum, None = raw output bits
    input_threshold: float = None      # BinaryThreshold value if present
    name: str = "logic_network"
    meta: dict = field(default_factory=dict)

    @property
    def n_outputs(self):
        return len(self.layers[-1])

    @property
    def group_size(self):
        return self.n_outputs // self.num_classes if self.num_classes else None

    def stats(self):
        per_layer = []
        for L in self.layers:
            ks = np.array([l.k for l in L])
            per_layer.append({k: int((ks == k).sum()) for k in range(7) if (ks == k).any()})
        tot = sum(len(L) for L in self.layers)
        real = sum(1 for L in self.layers for l in L if l.k >= 2)
        return {"luts": tot, "luts_k>=2": real, "per_layer": per_layer,
                "depth": len(self.layers)}

    # ------------------------------------------------------------ file format
    def to_dict(self):
        layers = []
        for L in self.layers:
            layers.append([{"p": [int(p) for p in l.pins],
                            "t": format(l.init_int(), "x")} for l in L])
        head = ({"type": "groupsum", "num_classes": int(self.num_classes),
                 "tau": float(self.meta.get("tau", 1.0))} if self.num_classes else {"type": "bits"})
        return {"format": FORMAT, "version": VERSION, "name": self.name,
                "n_inputs": int(self.n_inputs), "input_threshold": self.input_threshold,
                "layers": layers, "head": head, "meta": self.meta}

    @classmethod
    def from_dict(cls, d):
        if d.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} file")
        if int(d.get("version", 0)) > VERSION:
            raise ValueError(f"netlist version {d['version']} newer than supported {VERSION}")
        layers = []
        for L in d["layers"]:
            lay = []
            for e in L:
                k, v = len(e["p"]), int(e["t"], 16)
                lay.append(LUT(tuple(e["p"]), np.array([(v >> a) & 1 for a in range(1 << k)], bool)))
            layers.append(lay)
        h = d.get("head", {"type": "bits"})
        meta = dict(d.get("meta", {}))
        if h["type"] == "groupsum":
            meta.setdefault("tau", h.get("tau", 1.0))
        return cls(int(d["n_inputs"]), layers, h.get("num_classes") if h["type"] == "groupsum" else None,
                   d.get("input_threshold"), d.get("name", "logic_network"), meta)

    def __repr__(self):
        s = self.stats()
        return (f"Netlist({self.name}: {self.n_inputs} in, "
                f"{[len(L) for L in self.layers]} LUTs/layer, "
                f"{s['luts_k>=2']} non-trivial, classes={self.num_classes})")


# --------------------------------------------------------------- extraction
def extract_netlist(model, name=None):
    """Build a :class:`Netlist` from a Keras model made of
    ``[BinaryThreshold] [Flatten] LUTDense+ [GroupSum]``."""
    layers, n_in, thr, ncls, tau = [], None, None, None, 1.0
    passthrough = (keras.layers.InputLayer, keras.layers.Flatten,
                   keras.layers.Reshape)
    for lay in model.layers:
        if isinstance(lay, LUTDense):
            if n_in is None:
                n_in = lay.input_dim
            idx, tab = lay.get_luts()
            layers.append([LUT(tuple(int(p) for p in idx[u]), tab[u].copy())
                           for u in range(lay.units)])
        elif isinstance(lay, BinaryThreshold):
            if layers:
                raise ValueError("BinaryThreshold must precede all LUTDense layers")
            thr = lay.threshold
        elif isinstance(lay, GroupSum):
            ncls, tau = lay.num_classes, lay.tau
        elif isinstance(lay, passthrough) and not layers:
            continue
        else:
            raise ValueError(f"layer {lay.name} ({type(lay).__name__}) not supported")
    if not layers:
        raise ValueError("model has no LUTDense layers")
    return Netlist(n_in, layers, ncls, thr, name or model.name, {"tau": float(tau)})


def save_netlist(net, path):
    with open(path, "w") as f:
        json.dump(net.to_dict(), f, separators=(",", ":"))


def load_netlist(path):
    with open(path) as f:
        return Netlist.from_dict(json.load(f))


# ---------------------------------------------------------------- simulation
def _pack_layer(L):
    K = max(1, max(l.k for l in L))
    U = len(L)
    idx = np.zeros((U, K), np.int64)
    tab = np.zeros((U, 1 << K), bool)
    a = np.arange(1 << K)
    for u, l in enumerate(L):
        if l.k:
            idx[u, :l.k] = l.pins
            idx[u, l.k:] = l.pins[0]
        tab[u] = l.table[a & ((1 << l.k) - 1)]
    return idx, tab, K


def simulate(net, x, batch=2048, return_bits=False):
    """Bit-exact evaluation.

    x: (N, n_inputs) bits (or raw values if the netlist has an input
    threshold).  Returns ``(scores (N, C) int, cls (N,) int)`` for GroupSum
    networks, otherwise the output bits (N, n_outputs).
    """
    x = np.asarray(x)
    if net.input_threshold is not None and x.dtype != bool:
        x = x > net.input_threshold
    x = x.astype(bool).reshape(len(x), -1)
    packed = [_pack_layer(L) for L in net.layers]
    outs = []
    for s in range(0, len(x), batch):
        h = x[s:s + batch]
        for idx, tab, K in packed:
            addr = h[:, idx].astype(np.int64) @ (1 << np.arange(K))
            h = tab[np.arange(len(idx))[None, :], addr]
        outs.append(h)
    bits = np.concatenate(outs)
    if not net.num_classes or return_bits:
        return bits
    scores = bits.reshape(len(bits), net.num_classes, -1).sum(-1)
    return scores, scores.argmax(-1)
