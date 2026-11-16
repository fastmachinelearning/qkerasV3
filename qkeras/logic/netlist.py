"""Boolean netlist of a trained LUT network: extraction, neutral file format,
bit-exact simulation.

HDL / C generation, netlist simplification, synthesis and place & route live in the
separate ``logicnet-fpga`` backend, which reads the file written by :func:`save_netlist`.

A :class:`Netlist` is strictly layered: LUTs of layer ``L`` read outputs of
layer ``L-1`` (layer ``-1`` = primary input wires).  Each LUT is ``(pins, table)``:
``pins[j]`` is the source index driving address bit ``j`` and ``table[a]`` the
output for address ``a``.  ``k = len(pins)`` may differ per LUT (k = 0 is a
constant).  LUTs produced by :mod:`qkeras.logic.convert` can be *wide*
(k up to 16) and carry a ``care`` mask: ``care[a] = False`` marks address ``a``
as a don't-care (never observed during range analysis); the backend may
assign any value there when it decomposes the LUT into device LUTs.

Heads (how the last layer is read):

* ``groupsum``: popcount of ``n_outputs / num_classes`` consecutive bits per
  class (LUTDense + GroupSum networks).
* ``words``: the last layer holds ``num_classes`` unsigned integers of
  ``word_bits`` bits each (class ``c`` = bits ``[c*B, (c+1)*B)``, LSB first);
  used for converted QKeras teachers whose logits are exact integers.
* ``bits``: raw output bits.

Input encoders (``Netlist.input_encoder``) map raw features to input wires:

* ``{"type": "threshold", "threshold": t}``: wire i = x_i > t.
* ``{"type": "uniform", "scale": [...], "zmin": [...], "bits": [...]}``:
  feature f gives ``bits[f]`` wires (LSB first) of the code
  ``clip(rint(x_f * scale_f) - zmin_f, 0, 2**bits_f - 1)`` (``rint`` rounds
  half to even, like ``keras.ops.round``).
* ``{"type": "thermometer", "feature": [...], "threshold": [...]}``: wire i =
  x[feature_i] > threshold_i.
"""

import json
from dataclasses import dataclass, field

import numpy as np
import keras

from .layers import BinaryThreshold, GroupSum, LUTDense

__all__ = ["LUT", "Netlist", "extract_netlist", "simulate", "save_netlist", "load_netlist",
           "encode_inputs", "FORMAT", "VERSION"]

FORMAT, VERSION = "logicnet-netlist", 2
MAX_LUT_INPUTS = 20


def _hex_to_bits(h, n):
    v = int(h, 16)
    if n <= 4096:
        return np.array([(v >> a) & 1 for a in range(n)], bool)
    b = np.frombuffer(v.to_bytes((n + 7) // 8, "little"), np.uint8)
    return np.unpackbits(b, bitorder="little")[:n].astype(bool)


def _bits_to_hex(t):
    t = np.asarray(t, bool)
    if t.size <= 4096:
        return format(int(sum(1 << int(a) for a in np.flatnonzero(t))), "x")
    b = np.packbits(t, bitorder="little").tobytes()
    return format(int.from_bytes(b, "little"), "x")


@dataclass
class LUT:
    pins: tuple
    table: np.ndarray            # bool, length 2**len(pins)
    care: np.ndarray = None      # bool, length 2**len(pins); None = fully specified

    @property
    def k(self):
        return len(self.pins)

    def init_int(self):
        """Truth table as integer, bit a = output for address a."""
        return int(_bits_to_hex(self.table), 16)

    def is_const(self):
        return self.k == 0

    @property
    def n_care(self):
        return int(self.care.sum()) if self.care is not None else 1 << self.k
@dataclass
class Netlist:
    n_inputs: int
    layers: list                       # list[list[LUT]]
    num_classes: int = None            # GroupSum / words head, None = raw output bits
    input_threshold: float = None      # BinaryThreshold value if present
    name: str = "logic_network"
    meta: dict = field(default_factory=dict)
    head_type: str = None              # "groupsum" | "words" | "bits"; None = infer
    word_bits: int = None              # words head: bits per class word
    input_encoder: dict = None         # see module docstring
    linear: dict = None                # linear head: {"terms": [...], "bias": [...]}

    def __post_init__(self):
        if self.head_type is None:
            self.head_type = "groupsum" if self.num_classes else "bits"
        if self.input_encoder is None and self.input_threshold is not None:
            self.input_encoder = {"type": "threshold", "threshold": float(self.input_threshold)}

    @property
    def n_outputs(self):
        return len(self.layers[-1])

    @property
    def group_size(self):
        if self.head_type == "groupsum" and self.num_classes:
            return self.n_outputs // self.num_classes
        return None

    def linear_bounds(self):
        """(zmin, zmax) over all classes of the linear head's integer score."""
        lo = [int(b) + sum(min(0, int(w)) for _, w in t)
              for t, b in zip(self.linear["terms"], self.linear["bias"])]
        hi = [int(b) + sum(max(0, int(w)) for _, w in t)
              for t, b in zip(self.linear["terms"], self.linear["bias"])]
        return min(lo), max(hi)

    @property
    def max_k(self):
        return max((l.k for L in self.layers for l in L), default=0)

    @property
    def has_dont_cares(self):
        return any(l.care is not None and not l.care.all() for L in self.layers for l in L)

    def stats(self):
        per_layer = []
        for L in self.layers:
            ks = np.array([l.k for l in L])
            per_layer.append({int(k): int((ks == k).sum()) for k in np.unique(ks)})
        tot = sum(len(L) for L in self.layers)
        real = sum(1 for L in self.layers for l in L if l.k >= 2)
        out = {"luts": tot, "luts_k>=2": real, "per_layer": per_layer,
               "depth": len(self.layers), "max_k": self.max_k}
        if self.has_dont_cares:
            care = sum(l.n_care for L in self.layers for l in L)
            full = sum(1 << l.k for L in self.layers for l in L)
            out["care_fraction"] = care / full
        return out

    def _needs_v2(self):
        return (self.head_type in ("words", "linear") or self.max_k > 6 or self.has_dont_cares
                or (self.input_encoder is not None and self.input_encoder.get("type") != "threshold"))

    # ------------------------------------------------------------ file format
    def to_dict(self):
        layers = []
        for L in self.layers:
            lay = []
            for l in L:
                e = {"p": [int(p) for p in l.pins], "t": _bits_to_hex(l.table)}
                if l.care is not None and not l.care.all():
                    e["c"] = _bits_to_hex(l.care)
                lay.append(e)
            layers.append(lay)
        if self.head_type == "groupsum":
            head = {"type": "groupsum", "num_classes": int(self.num_classes),
                    "tau": float(self.meta.get("tau", 1.0))}
        elif self.head_type == "words":
            head = {"type": "words", "num_classes": int(self.num_classes),
                    "word_bits": int(self.word_bits)}
        elif self.head_type == "linear":
            head = {"type": "linear", "num_classes": int(self.num_classes),
                    "word_bits": int(self.word_bits),
                    "terms": [[[int(p), int(w)] for p, w in t] for t in self.linear["terms"]],
                    "bias": [int(b) for b in self.linear["bias"]]}
        else:
            head = {"type": "bits"}
        d = {"format": FORMAT, "version": 2 if self._needs_v2() else 1, "name": self.name,
             "n_inputs": int(self.n_inputs), "input_threshold": self.input_threshold,
             "layers": layers, "head": head, "meta": self.meta}
        if self.input_encoder is not None and self.input_encoder.get("type") != "threshold":
            d["input_encoder"] = self.input_encoder
        return d

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
                n = 1 << len(e["p"])
                care = _hex_to_bits(e["c"], n) if "c" in e else None
                lay.append(LUT(tuple(e["p"]), _hex_to_bits(e["t"], n), care))
            layers.append(lay)
        h = d.get("head", {"type": "bits"})
        meta = dict(d.get("meta", {}))
        if h["type"] == "groupsum":
            meta.setdefault("tau", h.get("tau", 1.0))
        lin = ({"terms": [[(int(p), int(w)) for p, w in t] for t in h["terms"]],
                "bias": [int(b) for b in h["bias"]]} if h["type"] == "linear" else None)
        return cls(int(d["n_inputs"]), layers,
                   h.get("num_classes") if h["type"] in ("groupsum", "words", "linear") else None,
                   d.get("input_threshold"), d.get("name", "logic_network"), meta,
                   head_type=h["type"], word_bits=h.get("word_bits"),
                   input_encoder=d.get("input_encoder"), linear=lin)

    def __repr__(self):
        s = self.stats()
        return (f"Netlist({self.name}: {self.n_inputs} in, "
                f"{[len(L) for L in self.layers]} LUTs/layer, "
                f"{s['luts_k>=2']} non-trivial, max k={self.max_k}, head={self.head_type}, "
                f"classes={self.num_classes})")


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
def encode_inputs(net, x):
    """Raw features -> input wires according to ``net.input_encoder``.
    Boolean ``x`` is taken as wires already."""
    x = np.asarray(x)
    enc = net.input_encoder
    if x.dtype == bool or enc is None:
        return x.astype(bool).reshape(len(x), -1)
    x = x.reshape(len(x), -1)
    t = enc["type"]
    if t == "threshold":
        return x > enc["threshold"]
    if t == "thermometer":
        return x[:, np.asarray(enc["feature"])] > np.asarray(enc["threshold"], x.dtype)
    if t == "uniform":
        scale = np.asarray(enc["scale"], np.float32)
        zmin = np.asarray(enc["zmin"], np.int64)
        bits = np.asarray(enc["bits"], np.int64)
        code = np.rint(x.astype(np.float32) * scale).astype(np.int64) - zmin
        code = np.clip(code, 0, (1 << bits) - 1)
        cols = [(code[:, f:f + 1] >> np.arange(bits[f])) & 1 for f in range(len(bits))]
        return np.concatenate(cols, 1).astype(bool)
    raise ValueError(f"unknown input encoder {t!r}")


def _pack_groups(L):
    """Group the LUTs of one layer by k: list of (unit_idx, pins (U,k), table (U,2^k), k)."""
    ks = np.array([l.k for l in L])
    groups = []
    for k in np.unique(ks):
        u = np.flatnonzero(ks == k)
        k = int(k)
        idx = np.array([L[i].pins for i in u], np.int64).reshape(len(u), k)
        tab = np.stack([L[i].table for i in u]).reshape(len(u), 1 << k)
        groups.append((u, idx, tab, k))
    return groups


def simulate(net, x, batch=2048, return_bits=False):
    """Bit-exact evaluation.

    x: (N, n_inputs) input wires (bool) or raw features (encoded with
    ``net.input_encoder``).  Returns ``(scores (N, C) int, cls (N,) int)`` for
    ``groupsum`` (popcounts) and ``words`` (integer words) heads, otherwise the
    output bits (N, n_outputs).  Don't-care addresses evaluate to the stored
    table value.
    """
    x = encode_inputs(net, x)
    packed = [(_pack_groups(L), len(L)) for L in net.layers]
    outs = []
    for s in range(0, len(x), batch):
        h = x[s:s + batch]
        for groups, U in packed:
            out = np.zeros((len(h), U), bool)
            for u, idx, tab, k in groups:
                if k == 0:
                    out[:, u] = tab[:, 0]
                    continue
                addr = h[:, idx].astype(np.int64) @ (1 << np.arange(k, dtype=np.int64))
                out[:, u] = tab[np.arange(len(u))[None, :], addr]
            h = out
        outs.append(h)
    bits = np.concatenate(outs)
    if return_bits or net.head_type == "bits":
        return bits
    C = net.num_classes
    if net.head_type == "words":
        B = net.word_bits
        scores = (bits.reshape(len(bits), C, B).astype(np.int64)
                  * (1 << np.arange(B, dtype=np.int64))).sum(-1)
    elif net.head_type == "linear":
        zmin, _ = net.linear_bounds()
        scores = np.zeros((len(bits), C), np.int64)
        for c, (t, b) in enumerate(zip(net.linear["terms"], net.linear["bias"])):
            if t:
                src = np.array([p for p, _ in t], np.int64)
                w = np.array([w for _, w in t], np.int64)
                scores[:, c] = bits[:, src].astype(np.int64) @ w
            scores[:, c] += int(b) - zmin
    else:
        scores = bits.reshape(len(bits), C, -1).sum(-1)
    return scores, scores.argmax(-1)
