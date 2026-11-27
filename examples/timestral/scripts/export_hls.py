from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # examples/timestral

import argparse
import re
from pathlib import Path

import numpy as np
from qkeras import get_quantizer
from qkeras.range_analysis import optimize_bitwidths
from safetensors.numpy import load_file

from timestral_kd.legacy import legacy_factory
from timestral_kd.model import (
    TimestralSpec,
    build_timestral_multivariate,
    forecast_mask,
    load_timestral_weights,
    quantile_increment_widths,
    rope_tables,
    scale_context_mv,
)
from forecast_csv import (
    CONTEXT_PATCHES,
    COVARIATE_COLUMNS,
    DEFAULT_CSV,
    HORIZON_PATCHES,
    MODEL_DIR,
    TARGET_COLUMNS,
    build_sliding_windows,
    load_process_csv,
)


def parse_q_spec(q_obj) -> tuple[int, int, bool]:
    """Extract (total_bits, int_bits, is_signed) from a QKeras quantizer instance or string."""
    if q_obj is None:
        return (16, 6, True)
    q_str = str(q_obj)
    m = re.search(r"quantized_bits\(\s*(\d+)\s*,\s*(-?\d+)", q_str)
    keep_neg = "keep_negative=False" not in q_str and "keep_negative=0" not in q_str
    if m:
        bits = int(m.group(1))
        int_bits = int(m.group(2))
        return bits, int_bits, keep_neg
    bits = getattr(q_obj, "bits", 16)
    int_bits = getattr(q_obj, "integer", 6)
    keep_neg = bool(getattr(q_obj, "keep_negative", True))
    return int(bits), int(int_bits), keep_neg


def ap_type_str(q_obj, default: str = "ap_fixed<16, 6, AP_RND, AP_SAT>") -> str:
    """Convert QKeras quantizer spec to Vitis HLS ap_fixed / ap_ufixed typedef string."""
    if q_obj is None:
        return default
    bits, int_bits, is_signed = parse_q_spec(q_obj)
    # In ap_fixed<W, I>, I includes the sign bit for signed types
    hls_int = int_bits + (1 if is_signed else 0)
    hls_int = max(1, min(bits, hls_int))
    prefix = "ap_fixed" if is_signed else "ap_ufixed"
    return f"{prefix}<{bits}, {hls_int}, AP_RND, AP_SAT>"


def quantize_array(arr: np.ndarray, q_obj) -> np.ndarray:
    """Apply QKeras quantizer to a numpy weight/bias array."""
    if q_obj is None:
        return arr.astype(np.float32)
    q_fn = get_quantizer(q_obj) if isinstance(q_obj, str) else q_obj
    return np.asarray(q_fn(arr.astype(np.float32)), dtype=np.float32)


def format_c_array(name: str, c_type: str, arr: np.ndarray, per_line: int = 12) -> str:
    """Format a flattened numpy array as a static const C++ array."""
    flat = arr.ravel()
    lines = [f"static const {c_type} {name}[{flat.size}] = {{"]
    for i in range(0, flat.size, per_line):
        chunk = ", ".join(f"{float(v):.8f}" for v in flat[i : i + per_line])
        lines.append(f"    {chunk},")
    lines.append("};")
    return "".join(lines)


def generate_lut(func_name: str, q_in, q_out, table_size: int = 256) -> tuple[str, float, float]:
    """Generate uniform-grid fixed-point LUT for SiLU or Sigmoid."""
    _, int_bits, is_signed = parse_q_spec(q_in)
    bound = float(2 ** int_bits)
    x_min = -bound if is_signed else 0.0
    x_max = bound
    xs = np.linspace(x_min, x_max, table_size, dtype=np.float32)
    if func_name == "silu":
        ys = xs / (1.0 + np.exp(-np.clip(xs, -20.0, 20.0)))
    elif func_name == "sigmoid":
        ys = 1.0 / (1.0 + np.exp(-np.clip(xs, -20.0, 20.0)))
    else:
        raise ValueError(f"Unsupported unary LUT function: {func_name}")
    ys_q = quantize_array(ys, q_out)
    out_type = ap_type_str(q_out)
    decl = format_c_array(f"LUT_{func_name.upper()}", out_type, ys_q)
    return decl, x_min, x_max


def export_hls_project(
    model,
    spec: TimestralSpec,
    num_variates: int,
    context_patches: int,
    horizon_patches: int,
    sample_ctx: np.ndarray,
    golden_pred: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    P, D, V, L, h = spec.patch_size, spec.d_model, num_variates, context_patches, horizon_patches
    S = L + h
    H, hd = spec.num_heads, spec.head_dim
    eh = spec.embed_hidden or 4 * D
    hh = spec.head_hidden or 4 * D

    cos, sin = rope_tables(spec, S)
    time_mask = forecast_mask(L, h).astype(np.uint8)

    # 1. Build weights_and_luts.h
    header_lines = [
        "#ifndef TIMESTRAL_WEIGHTS_AND_LUTS_H",
        "#define TIMESTRAL_WEIGHTS_AND_LUTS_H",
        "",
        "#include <ap_fixed.h>",
        "#include <ap_int.h>",
        "#include <hls_math.h>",
        "",
        f"constexpr int V_DIM = {V};",
        f"constexpr int L_PATCHES = {L};",
        f"constexpr int H_PATCHES = {h};",
        f"constexpr int S_TOKENS = {S};",
        f"constexpr int P_SIZE = {P};",
        f"constexpr int D_MODEL = {D};",
        f"constexpr int N_HEADS = {H};",
        f"constexpr int HEAD_DIM = {hd};",
        f"constexpr int ROPE_HALF = {cos.shape[-1]};",
        f"constexpr int EMBED_HIDDEN = {eh};",
        f"constexpr int MLP_HIDDEN = {spec.mlp_hidden_dim};",
        f"constexpr int HEAD_HIDDEN = {hh};",
        f"constexpr int N_QUANTILES = {spec.num_quantiles};",
        f"constexpr int MEDIAN_IDX = {spec.quantile_knots.index(0.5)};",
        f"constexpr float NORM_EPS = {spec.norm_eps}f;",
        f"constexpr float WIDTH_FACTOR = {spec.quantile_width_factor}f;",
        "",
        "// Fixed-point types derived from QKeras range_analysis",
        "typedef ap_fixed<16, 6, AP_RND, AP_SAT> act_t;",
        "typedef ap_fixed<32, 12, AP_RND, AP_SAT> acc_t;",
        "typedef ap_fixed<12, 4, AP_RND, AP_SAT> weight_t;",
        "typedef ap_fixed<16, 6, AP_RND, AP_SAT> bias_t;",
        "",
    ]

    # Precomputed RoPE & masks
    header_lines.append(format_c_array("ROPE_COS", "act_t", cos))
    header_lines.append(format_c_array("ROPE_SIN", "act_t", sin))
    header_lines.append(format_c_array("TIME_MASK", "ap_uint<1>", time_mask, per_line=32))
    # width_factor * knot gap per quantile (see keras_timestral.quantile_increment_widths)
    header_lines.append(
        "const float QUANTILE_WIDTH[N_QUANTILES] = {"
        + ", ".join(f"{w:.8g}f" for w in quantile_increment_widths(spec.quantile_knots, spec.quantile_width_factor))
        + "};"
    )
    header_lines.append("")

    # Unary LUTs from first QUnary layers
    silu_lut_code, silu_min, silu_max = generate_lut(
        "silu", "quantized_bits(12, 4, symmetric=1)", "quantized_bits(12, 3, symmetric=1)"
    )
    sig_lut_code, sig_min, sig_max = generate_lut(
        "sigmoid", "quantized_bits(12, 4, symmetric=1)", "quantized_bits(12, 3, symmetric=1)"
    )
    header_lines.extend([
        f"constexpr float SILU_MIN = {silu_min}f, SILU_MAX = {silu_max}f;",
        silu_lut_code,
        f"constexpr float SIG_MIN = {sig_min}f, SIG_MAX = {sig_max}f;",
        sig_lut_code,
        "",
    ])

    # Extract quantized parameters from every QDense / RMSNorm / AddToken layer
    for layer in model.layers:
        lname = layer.name.upper()
        if hasattr(layer, "kernel"):
            w_np = layer.kernel.numpy()
            wq = getattr(layer, "kernel_quantizer_internal", None)
            w_quant = quantize_array(w_np, wq)
            w_ctype = ap_type_str(wq, "weight_t")
            header_lines.append(f"typedef {w_ctype} {lname}_W_T;")
            header_lines.append(format_c_array(f"{lname}_W", f"{lname}_W_T", w_quant))
            if getattr(layer, "bias", None) is not None:
                b_np = layer.bias.numpy()
                bq = getattr(layer, "bias_quantizer_internal", None)
                b_quant = quantize_array(b_np, bq)
                b_ctype = ap_type_str(bq, "bias_t")
                header_lines.append(f"typedef {b_ctype} {lname}_B_T;")
                header_lines.append(format_c_array(f"{lname}_B", f"{lname}_B_T", b_quant))
            header_lines.append("")
        elif hasattr(layer, "scale"):
            s_np = layer.scale.numpy()
            gq = getattr(layer, "gain_quantizer_internal", None)
            s_quant = quantize_array(s_np, gq)
            header_lines.append(format_c_array(f"{lname}_SCALE", "weight_t", s_quant))
            header_lines.append("")
        elif hasattr(layer, "token"):
            t_np = layer.token.numpy().reshape(-1)
            header_lines.append(format_c_array(f"{lname}_TOKEN", "act_t", t_np))
            header_lines.append("")

    header_lines.append("#endif // TIMESTRAL_WEIGHTS_AND_LUTS_H")
    (out_dir / "weights_and_luts.h").write_text("".join(header_lines))

    # 2. Build timestral_hls.h
    hls_h = """#ifndef TIMESTRAL_HLS_H
#define TIMESTRAL_HLS_H

#include "weights_and_luts.h"

// Top-level Vitis HLS synthesizable inference kernel
// x_in : [V_DIM][L_PATCHES * P_SIZE] causal-scaled sensor input
// y_out: [V_DIM][H_PATCHES * P_SIZE][N_QUANTILES] normalized monotone quantiles
void timestral_infer(
    const act_t x_in[V_DIM][L_PATCHES * P_SIZE],
    act_t y_out[V_DIM][H_PATCHES * P_SIZE][N_QUANTILES]
);

#endif // TIMESTRAL_HLS_H
"""
    (out_dir / "timestral_hls.h").write_text(hls_h)

    # 3. Build timestral_hls.cpp
    block_calls = []
    for i, kind in enumerate(spec.block_schedule):
        block_calls.append(f"    // --- Transformer Block {i} ({kind}) ---")
        block_calls.append(
            f"    run_time_attention<B{i}_ATTN0_WQKV_W_T, B{i}_ATTN0_GATE_W_T, B{i}_ATTN0_WO_W_T>("
            f"x, B{i}_ATTN_NORM0_SCALE, B{i}_ATTN0_WQKV_W, B{i}_ATTN0_GATE_W, B{i}_ATTN0_WO_W);"
        )
        if kind == "time_variate":
            block_calls.append(
                f"    run_variate_attention<B{i}_ATTN1_WQKV_W_T, B{i}_ATTN1_GATE_W_T, B{i}_ATTN1_WO_W_T>("
                f"x, B{i}_ATTN_NORM1_SCALE, B{i}_ATTN1_WQKV_W, B{i}_ATTN1_GATE_W, B{i}_ATTN1_WO_W);"
            )
        block_calls.append(
            f"    run_swiglu_mlp<B{i}_W12_W_T, B{i}_W3_W_T>("
            f"x, B{i}_MLP_NORM_SCALE, B{i}_W12_W, B{i}_W3_W);"
        )

    hls_cpp = f"""#include "timestral_hls.h"

static inline act_t lookup_silu(act_t x) {{
#pragma HLS INLINE
    float xf = (float)x;
    if (xf <= SILU_MIN) return LUT_SILU[0];
    if (xf >= SILU_MAX) return LUT_SILU[255];
    int idx = (int)(((xf - SILU_MIN) / (SILU_MAX - SILU_MIN)) * 255.0f);
    return LUT_SILU[idx];
}}

static inline act_t lookup_sigmoid(act_t x) {{
#pragma HLS INLINE
    float xf = (float)x;
    if (xf <= SIG_MIN) return LUT_SIGMOID[0];
    if (xf >= SIG_MAX) return LUT_SIGMOID[255];
    int idx = (int)(((xf - SIG_MIN) / (SIG_MAX - SIG_MIN)) * 255.0f);
    return LUT_SIGMOID[idx];
}}

static void rms_norm_vec(const act_t in[D_MODEL], const weight_t scale[D_MODEL], act_t out[D_MODEL]) {{
#pragma HLS INLINE off
    acc_t sq_sum = 0;
    for (int d = 0; d < D_MODEL; ++d) {{
#pragma HLS PIPELINE II=1
        sq_sum += (acc_t)in[d] * (acc_t)in[d];
    }}
    float ms = (float)sq_sum / (float)D_MODEL;
    act_t inv_rms = (act_t)(1.0f / hls::sqrtf(ms + NORM_EPS));
    for (int d = 0; d < D_MODEL; ++d) {{
#pragma HLS PIPELINE II=1
        out[d] = in[d] * inv_rms * scale[d];
    }}
}}

template <int IN_D, int OUT_D, typename W_T>
static void linear_no_bias(const act_t in[IN_D], const W_T W[IN_D * OUT_D], act_t out[OUT_D]) {{
#pragma HLS INLINE off
    for (int o = 0; o < OUT_D; ++o) {{
#pragma HLS PIPELINE II=1
        acc_t sum = 0;
        for (int i = 0; i < IN_D; ++i) {{
            sum += (acc_t)in[i] * (acc_t)W[i * OUT_D + o];
        }}
        out[o] = (act_t)sum;
    }}
}}

template <int IN_D, int OUT_D, typename W_T, typename B_T>
static void linear_bias(
    const act_t in[IN_D], const W_T W[IN_D * OUT_D], const B_T B[OUT_D], act_t out[OUT_D]
) {{
#pragma HLS INLINE off
    for (int o = 0; o < OUT_D; ++o) {{
#pragma HLS PIPELINE II=1
        acc_t sum = (acc_t)B[o];
        for (int i = 0; i < IN_D; ++i) {{
            sum += (acc_t)in[i] * (acc_t)W[i * OUT_D + o];
        }}
        out[o] = (act_t)sum;
    }}
}}

static void apply_rope_token(act_t vec[N_HEADS][HEAD_DIM], int pos) {{
#pragma HLS INLINE
    for (int h = 0; h < N_HEADS; ++h) {{
        for (int r = 0; r < ROPE_HALF; ++r) {{
#pragma HLS UNROLL
            // rotate-half pairing of the released weights: r <-> r + HEAD_DIM/2
            act_t x1 = vec[h][r];
            act_t x2 = vec[h][r + HEAD_DIM / 2];
            act_t c = ROPE_COS[pos * ROPE_HALF + r];
            act_t s = ROPE_SIN[pos * ROPE_HALF + r];
            vec[h][r]                = x1 * c - x2 * s;
            vec[h][r + HEAD_DIM / 2] = x2 * c + x1 * s;
        }}
    }}
}}

template <typename WQKV_T, typename WG_T, typename WO_T>
static void run_time_attention(
    act_t x[V_DIM][S_TOKENS][D_MODEL],
    const weight_t norm_scale[D_MODEL],
    const WQKV_T w_qkv[D_MODEL * 3 * D_MODEL],
    const WG_T w_gate[D_MODEL * D_MODEL],
    const WO_T w_o[D_MODEL * D_MODEL]
) {{
    const float inv_sqrt_hd = 1.0f / hls::sqrtf((float)HEAD_DIM);
    for (int v = 0; v < V_DIM; ++v) {{
        act_t q[S_TOKENS][N_HEADS][HEAD_DIM];
        act_t k[S_TOKENS][N_HEADS][HEAD_DIM];
        act_t val[S_TOKENS][N_HEADS][HEAD_DIM];
        act_t gate[S_TOKENS][D_MODEL];

        for (int s = 0; s < S_TOKENS; ++s) {{
            act_t n_vec[D_MODEL];
            act_t qkv_flat[3 * D_MODEL];
            act_t g_raw[D_MODEL];
            rms_norm_vec(x[v][s], norm_scale, n_vec);
            linear_no_bias<D_MODEL, 3 * D_MODEL, WQKV_T>(n_vec, w_qkv, qkv_flat);
            linear_no_bias<D_MODEL, D_MODEL, WG_T>(n_vec, w_gate, g_raw);

            for (int d = 0; d < D_MODEL; ++d) {{
#pragma HLS PIPELINE II=1
                gate[s][d] = lookup_sigmoid(g_raw[d]);
            }}
            for (int hd = 0; hd < HEAD_DIM; ++hd) {{
                for (int h = 0; h < N_HEADS; ++h) {{
#pragma HLS UNROLL
                    q[s][h][hd]   = qkv_flat[(0 * HEAD_DIM + hd) * N_HEADS + h];
                    k[s][h][hd]   = qkv_flat[(1 * HEAD_DIM + hd) * N_HEADS + h];
                    val[s][h][hd] = qkv_flat[(2 * HEAD_DIM + hd) * N_HEADS + h];
                }}
            }}
            apply_rope_token(q[s], s);
            apply_rope_token(k[s], s);
        }}

        for (int sq = 0; sq < S_TOKENS; ++sq) {{
            act_t attn_out[D_MODEL];
            for (int h = 0; h < N_HEADS; ++h) {{
                float scores[S_TOKENS];
                float max_s = -1e9f;
                for (int sk = 0; sk < S_TOKENS; ++sk) {{
                    if (!TIME_MASK[sq * S_TOKENS + sk]) {{
                        scores[sk] = -1e9f;
                    }} else {{
                        acc_t dot = 0;
                        for (int hd = 0; hd < HEAD_DIM; ++hd) {{
                            dot += (acc_t)q[sq][h][hd] * (acc_t)k[sk][h][hd];
                        }}
                        scores[sk] = (float)dot * inv_sqrt_hd;
                    }}
                    if (scores[sk] > max_s) max_s = scores[sk];
                }}
                float exp_sum = 0.0f;
                for (int sk = 0; sk < S_TOKENS; ++sk) {{
                    scores[sk] = TIME_MASK[sq * S_TOKENS + sk] ? hls::expf(scores[sk] - max_s) : 0.0f;
                    exp_sum += scores[sk];
                }}
                float inv_sum = 1.0f / (exp_sum + 1e-9f);
                for (int hd = 0; hd < HEAD_DIM; ++hd) {{
                    acc_t wv = 0;
                    for (int sk = 0; sk < S_TOKENS; ++sk) {{
                        wv += (acc_t)(scores[sk] * inv_sum) * (acc_t)val[sk][h][hd];
                    }}
                    attn_out[h * HEAD_DIM + hd] = (act_t)wv * gate[sq][h * HEAD_DIM + hd];
                }}
            }}
            act_t proj[D_MODEL];
            linear_no_bias<D_MODEL, D_MODEL, WO_T>(attn_out, w_o, proj);
            for (int d = 0; d < D_MODEL; ++d) {{
#pragma HLS PIPELINE II=1
                x[v][sq][d] += proj[d];
            }}
        }}
    }}
}}

template <typename WQKV_T, typename WG_T, typename WO_T>
static void run_variate_attention(
    act_t x[V_DIM][S_TOKENS][D_MODEL],
    const weight_t norm_scale[D_MODEL],
    const WQKV_T w_qkv[D_MODEL * 3 * D_MODEL],
    const WG_T w_gate[D_MODEL * D_MODEL],
    const WO_T w_o[D_MODEL * D_MODEL]
) {{
    const float inv_sqrt_hd = 1.0f / hls::sqrtf((float)HEAD_DIM);
    for (int s = 0; s < S_TOKENS; ++s) {{
        act_t q[V_DIM][N_HEADS][HEAD_DIM];
        act_t k[V_DIM][N_HEADS][HEAD_DIM];
        act_t val[V_DIM][N_HEADS][HEAD_DIM];
        act_t gate[V_DIM][D_MODEL];

        for (int v = 0; v < V_DIM; ++v) {{
            act_t n_vec[D_MODEL];
            act_t qkv_flat[3 * D_MODEL];
            act_t g_raw[D_MODEL];
            rms_norm_vec(x[v][s], norm_scale, n_vec);
            linear_no_bias<D_MODEL, 3 * D_MODEL, WQKV_T>(n_vec, w_qkv, qkv_flat);
            linear_no_bias<D_MODEL, D_MODEL, WG_T>(n_vec, w_gate, g_raw);

            for (int d = 0; d < D_MODEL; ++d) {{
#pragma HLS PIPELINE II=1
                gate[v][d] = lookup_sigmoid(g_raw[d]);
            }}
            for (int hd = 0; hd < HEAD_DIM; ++hd) {{
                for (int h = 0; h < N_HEADS; ++h) {{
#pragma HLS UNROLL
                    q[v][h][hd]   = qkv_flat[(0 * HEAD_DIM + hd) * N_HEADS + h];
                    k[v][h][hd]   = qkv_flat[(1 * HEAD_DIM + hd) * N_HEADS + h];
                    val[v][h][hd] = qkv_flat[(2 * HEAD_DIM + hd) * N_HEADS + h];
                }}
            }}
        }}

        for (int vq = 0; vq < V_DIM; ++vq) {{
            act_t attn_out[D_MODEL];
            for (int h = 0; h < N_HEADS; ++h) {{
                float scores[V_DIM];
                float max_s = -1e9f;
                for (int vk = 0; vk < V_DIM; ++vk) {{
                    acc_t dot = 0;
                    for (int hd = 0; hd < HEAD_DIM; ++hd) {{
                        dot += (acc_t)q[vq][h][hd] * (acc_t)k[vk][h][hd];
                    }}
                    scores[vk] = (float)dot * inv_sqrt_hd;
                    if (scores[vk] > max_s) max_s = scores[vk];
                }}
                float exp_sum = 0.0f;
                for (int vk = 0; vk < V_DIM; ++vk) {{
                    scores[vk] = hls::expf(scores[vk] - max_s);
                    exp_sum += scores[vk];
                }}
                float inv_sum = 1.0f / (exp_sum + 1e-9f);
                for (int hd = 0; hd < HEAD_DIM; ++hd) {{
                    acc_t wv = 0;
                    for (int vk = 0; vk < V_DIM; ++vk) {{
                        wv += (acc_t)(scores[vk] * inv_sum) * (acc_t)val[vk][h][hd];
                    }}
                    attn_out[h * HEAD_DIM + hd] = (act_t)wv * gate[vq][h * HEAD_DIM + hd];
                }}
            }}
            act_t proj[D_MODEL];
            linear_no_bias<D_MODEL, D_MODEL, WO_T>(attn_out, w_o, proj);
            for (int d = 0; d < D_MODEL; ++d) {{
#pragma HLS PIPELINE II=1
                x[vq][s][d] += proj[d];
            }}
        }}
    }}
}}

template <typename W12_T, typename W3_T>
static void run_swiglu_mlp(
    act_t x[V_DIM][S_TOKENS][D_MODEL],
    const weight_t norm_scale[D_MODEL],
    const W12_T w12[D_MODEL * 2 * MLP_HIDDEN],
    const W3_T w3[MLP_HIDDEN * D_MODEL]
) {{
    for (int v = 0; v < V_DIM; ++v) {{
        for (int s = 0; s < S_TOKENS; ++s) {{
            act_t n_vec[D_MODEL];
            act_t w12_out[2 * MLP_HIDDEN];
            act_t swiglu[MLP_HIDDEN];
            act_t mlp_out[D_MODEL];

            rms_norm_vec(x[v][s], norm_scale, n_vec);
            linear_no_bias<D_MODEL, 2 * MLP_HIDDEN, W12_T>(n_vec, w12, w12_out);
            for (int m = 0; m < MLP_HIDDEN; ++m) {{
#pragma HLS PIPELINE II=1
                swiglu[m] = w12_out[m] * lookup_silu(w12_out[MLP_HIDDEN + m]);
            }}
            linear_no_bias<MLP_HIDDEN, D_MODEL, W3_T>(swiglu, w3, mlp_out);
            for (int d = 0; d < D_MODEL; ++d) {{
#pragma HLS PIPELINE II=1
                x[v][s][d] += mlp_out[d];
            }}
        }}
    }}
}}

void timestral_infer(
    const act_t x_in[V_DIM][L_PATCHES * P_SIZE],
    act_t y_out[V_DIM][H_PATCHES * P_SIZE][N_QUANTILES]
) {{
#pragma HLS INTERFACE m_axi port=x_in  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=y_out offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=return bundle=control

    act_t x[V_DIM][S_TOKENS][D_MODEL];

    // 1. Patch embedding + forecast query tokens
    for (int v = 0; v < V_DIM; ++v) {{
        for (int l = 0; l < L_PATCHES; ++l) {{
            act_t patch_in[2 * P_SIZE];
            for (int p = 0; p < P_SIZE; ++p) {{
#pragma HLS PIPELINE II=1
                patch_in[p]          = x_in[v][l * P_SIZE + p];
                patch_in[P_SIZE + p] = (act_t)0;
            }}
            act_t h_emb[EMBED_HIDDEN];
            act_t o_emb[D_MODEL];
            act_t r_emb[D_MODEL];
            linear_bias<2 * P_SIZE, EMBED_HIDDEN, EMBED_HIDDEN_W_T, EMBED_HIDDEN_B_T>(
                patch_in, EMBED_HIDDEN_W, EMBED_HIDDEN_B, h_emb
            );
            for (int e = 0; e < EMBED_HIDDEN; ++e) {{
#pragma HLS PIPELINE II=1
                h_emb[e] = lookup_silu(h_emb[e]);
            }}
            linear_bias<EMBED_HIDDEN, D_MODEL, EMBED_OUTPUT_W_T, EMBED_OUTPUT_B_T>(
                h_emb, EMBED_OUTPUT_W, EMBED_OUTPUT_B, o_emb
            );
            linear_bias<2 * P_SIZE, D_MODEL, EMBED_RESIDUAL_W_T, EMBED_RESIDUAL_B_T>(
                patch_in, EMBED_RESIDUAL_W, EMBED_RESIDUAL_B, r_emb
            );
            for (int d = 0; d < D_MODEL; ++d) {{
#pragma HLS PIPELINE II=1
                x[v][l][d] = o_emb[d] + r_emb[d];
            }}
        }}
        for (int fh = 0; fh < H_PATCHES; ++fh) {{
            for (int d = 0; d < D_MODEL; ++d) {{
#pragma HLS PIPELINE II=1
                x[v][L_PATCHES + fh][d] = FORECAST_QUERY_TOKEN[d];
            }}
        }}
    }}

    // 2. Transformer backbone
{chr(10).join(block_calls)}

    // 3. Output norm + quantile projection head + MonotoneQuantiles
    for (int v = 0; v < V_DIM; ++v) {{
        for (int fh = 0; fh < H_PATCHES; ++fh) {{
            act_t n_dec[D_MODEL];
            act_t h_head[HEAD_HIDDEN];
            act_t o_head[P_SIZE * N_QUANTILES];
            act_t r_head[P_SIZE * N_QUANTILES];

            rms_norm_vec(x[v][L_PATCHES + fh], OUT_NORM_SCALE, n_dec);
            linear_no_bias<D_MODEL, HEAD_HIDDEN, HEAD_HIDDEN_W_T>(n_dec, HEAD_HIDDEN_W, h_head);
            for (int hh_i = 0; hh_i < HEAD_HIDDEN; ++hh_i) {{
#pragma HLS PIPELINE II=1
                h_head[hh_i] = lookup_silu(h_head[hh_i]);
            }}
            linear_no_bias<HEAD_HIDDEN, P_SIZE * N_QUANTILES, HEAD_OUTPUT_W_T>(
                h_head, HEAD_OUTPUT_W, o_head
            );
            linear_no_bias<D_MODEL, P_SIZE * N_QUANTILES, HEAD_RESIDUAL_W_T>(
                n_dec, HEAD_RESIDUAL_W, r_head
            );

            for (int p = 0; p < P_SIZE; ++p) {{
                float raw_q[N_QUANTILES];
                for (int q = 0; q < N_QUANTILES; ++q) {{
                    raw_q[q] = (float)(o_head[p * N_QUANTILES + q] + r_head[p * N_QUANTILES + q]);
                }}
                float med = raw_q[MEDIAN_IDX];
                y_out[v][fh * P_SIZE + p][MEDIAN_IDX] = (act_t)med;

                float acc_lo = med;
                for (int q = MEDIAN_IDX - 1; q >= 0; --q) {{
                    float sp = hls::logf(1.0f + hls::expf(raw_q[q])) * QUANTILE_WIDTH[q];
                    acc_lo -= sp;
                    y_out[v][fh * P_SIZE + p][q] = (act_t)acc_lo;
                }}
                float acc_hi = med;
                for (int q = MEDIAN_IDX + 1; q < N_QUANTILES; ++q) {{
                    float sp = hls::logf(1.0f + hls::expf(raw_q[q])) * QUANTILE_WIDTH[q];
                    acc_hi += sp;
                    y_out[v][fh * P_SIZE + p][q] = (act_t)acc_hi;
                }}
            }}
        }}
    }}
}}
"""
    (out_dir / "timestral_hls.cpp").write_text(hls_cpp)

    # 4. Build C-sim verification testbench tb_timestral.cpp
    tb_lines = [
        "#include <cmath>",
        "#include <cstdio>",
        '#include "timestral_hls.h"',
        "",
        format_c_array("TB_X_IN", "act_t", sample_ctx),
        format_c_array("TB_GOLDEN_Y", "float", golden_pred),
        "",
        "int main() {",
        "    static act_t x_in[V_DIM][L_PATCHES * P_SIZE];",
        "    static act_t y_out[V_DIM][H_PATCHES * P_SIZE][N_QUANTILES];",
        "",
        "    for (int v = 0; v < V_DIM; ++v)",
        "        for (int t = 0; t < L_PATCHES * P_SIZE; ++t)",
        "            x_in[v][t] = TB_X_IN[v * (L_PATCHES * P_SIZE) + t];",
        "",
        "    timestral_infer(x_in, y_out);",
        "",
        "    double max_err = 0.0, sum_err = 0.0;",
        "    int count = V_DIM * H_PATCHES * P_SIZE * N_QUANTILES;",
        "    for (int v = 0; v < V_DIM; ++v) {",
        "        for (int t = 0; t < H_PATCHES * P_SIZE; ++t) {",
        "            for (int q = 0; q < N_QUANTILES; ++q) {",
        "                int idx = (v * H_PATCHES * P_SIZE + t) * N_QUANTILES + q;",
        "                double diff = std::fabs((double)y_out[v][t][q] - (double)TB_GOLDEN_Y[idx]);",
        "                if (diff > max_err) max_err = diff;",
        "                sum_err += diff;",
        "            }",
        "        }",
        "    }",
        '    std::printf("HLS C-sim vs QKeras golden | MAE = %.6f | MaxErr = %.6f\n", sum_err / count, max_err);',
        "    return (max_err < 0.15) ? 0 : 1;",
        "}",
    ]
    (out_dir / "tb_timestral.cpp").write_text("".join(tb_lines))

    # 5. Build Vitis HLS Tcl script
    tcl = """open_project -reset timestral_fpga_prj
set_top timestral_infer
add_files timestral_hls.cpp
add_files -tb tb_timestral.cpp
open_solution -reset "solution1" -flow_target vivado
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 5.0 -name default
csim_design
csynth_design
exit
"""
    (out_dir / "run_hls.tcl").write_text(tcl)
    print(f"Exported Vitis HLS C++ project to {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Quantized Multivariate Timestral to Vitis HLS C++")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Calibration CSV path")
    parser.add_argument("--weights", type=Path, default=None, help="Optional fine-tuned .weights.h5 path")
    parser.add_argument("--include-covariates", action="store_true", help="Include process covariates")
    parser.add_argument("--out-dir", type=Path, default=Path("fpga_timestral_hls"), help="Output HLS directory")
    args = parser.parse_args()

    spec = TimestralSpec.from_json(MODEL_DIR / "config.json")
    ctx_len = CONTEXT_PATCHES * spec.patch_size
    hor_len = HORIZON_PATCHES * spec.patch_size

    covs = COVARIATE_COLUMNS if args.include_covariates else []
    df, active_cols = load_process_csv(args.csv, TARGET_COLUMNS, covs)
    num_variates = len(active_cols)
    data_matrix = df[active_cols].to_numpy(dtype=np.float32)
    ctx_windows, _, _ = build_sliding_windows(data_matrix, ctx_len, hor_len, stride=32)

    model = build_timestral_multivariate(
        spec,
        num_variates=num_variates,
        context_patches=CONTEXT_PATCHES,
        horizon_patches=HORIZON_PATCHES,
        factory=legacy_factory(),
        name="timestral_mv_fpga",
    )

    if args.weights is not None and args.weights.exists():
        model.load_weights(args.weights)
        print(f"Loaded fine-tuned weights from {args.weights}")
    else:
        weights = load_file(MODEL_DIR / "model.safetensors")
        load_timestral_weights(model, weights, spec)
        s_calib, _, _ = scale_context_mv(
            ctx_windows[:32], spec.patch_size, spec.minimum_scale, spec.use_asinh
        )
        optimize_bitwidths(model, data=s_calib, policy="keep_fractional", headroom=1, verbose=1)

    sample_s, _, _ = scale_context_mv(
        ctx_windows[:1], spec.patch_size, spec.minimum_scale, spec.use_asinh
    )
    golden_q = model(sample_s, training=False).numpy()[0]

    export_hls_project(
        model=model,
        spec=spec,
        num_variates=num_variates,
        context_patches=CONTEXT_PATCHES,
        horizon_patches=HORIZON_PATCHES,
        sample_ctx=sample_s[0],
        golden_pred=golden_q,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
