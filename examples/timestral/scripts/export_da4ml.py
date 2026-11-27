from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # examples/timestral

import argparse
import faulthandler
import json
import multiprocessing as mp
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

faulthandler.enable()


def qkeras_to_da4ml_qformat(bits: int, int_bits: int, is_signed: bool) -> tuple[int, int, int]:
    """Convert QKeras (bits, integer, keep_negative) to da4ml (k, i, f) tuple."""
    k = 1 if is_signed else 0
    i = max(0, min(bits - k, int_bits))
    f = max(0, bits - k - i)
    return (k, i, f)


def _compile_single_tile_task(task: dict) -> dict:
    """Isolated single-tile da4ml compilation task (runs in a fresh worker process)."""
    import da4ml.codegen as da4ml_cg
    from da4ml.trace import FixedVariableArray, comb_trace
    from da4ml.trace.ops import einsum

    w_tile_int = task["w_tile_int"]
    frac_shift = task["frac_shift"]
    k_in, i_in, f_in = task["in_qformat"]
    mod_name = task["mod_name"]
    layer_dir = Path(task["layer_dir"])
    latency_cutoff = task["latency_cutoff"]
    test_x_tile = task["test_x_tile"]

    t_in, _ = w_tile_int.shape
    inp = FixedVariableArray.from_lhs(
        np.full(t_in, k_in, dtype=np.int32),
        np.full(t_in, i_in, dtype=np.int32),
        np.full(t_in, f_in, dtype=np.int32),
    )
    out = einsum("i,io->o", inp, w_tile_int.astype(np.float64))
    if frac_shift != 0:
        out = out * (2.0 ** (-frac_shift))

    comb_model = comb_trace(inp, out)
    sol = comb_model.solve(latency_cutoff=latency_cutoff) if hasattr(comb_model, "solve") else comb_model

    mod_dir = layer_dir / mod_name
    if hasattr(da4ml_cg, "RTLModel"):
        rtl = da4ml_cg.RTLModel(sol, mod_name, str(mod_dir), latency_cutoff=latency_cutoff)
        rtl.write()
        v_path = mod_dir / f"{mod_name}.v"
        raw_lat = getattr(rtl, "latency", getattr(sol, "latency", latency_cutoff))
        latency = int(max(raw_lat)) if isinstance(raw_lat, (tuple, list)) else int(raw_lat)
    elif hasattr(da4ml_cg, "to_verilog"):
        v_path = layer_dir / f"{mod_name}.v"
        v_path.write_text(da4ml_cg.to_verilog(sol, module_name=mod_name))
        raw_lat = getattr(sol, "latency", latency_cutoff)
        latency = int(max(raw_lat)) if isinstance(raw_lat, (tuple, list)) else int(raw_lat)
    else:
        raise AttributeError(f"Unsupported da4ml.codegen exports: {dir(da4ml_cg)}")

    n_adders = int(getattr(sol, "n_adders", getattr(sol, "cost", len(getattr(sol, "ops", [])))))
    pred_out = np.asarray(sol.predict(test_x_tile), dtype=np.float32)

    return {
        "o0": task["o0"],
        "o1": task["o1"],
        "v_path": str(v_path),
        "adders": n_adders,
        "latency": latency,
        "pred_out": pred_out,
    }


def run_da4ml_worker(
    npz_path: Path,
    meta_path: Path,
    out_dir: Path,
    tile_size: int,
    latency_cutoff: int,
    target_layers: list[str] | None,
    jobs: int,
) -> None:
    """Pure-NumPy + da4ml parallel worker with fresh process per tile (max_tasks_per_child=1)."""
    meta = json.loads(meta_path.read_text())
    arrays = np.load(npz_path)
    in_qfmt = tuple(meta["in_qformat"])
    layers_info = meta["layers"]
    if target_layers:
        wanted = set(target_layers)
        layers_info = [item for item in layers_info if item["name"] in wanted]

    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    print(
        f"Compiling {len(layers_info)} QDense layers (tile={tile_size}x{tile_size}, jobs={jobs}) -> {out_dir}/",
        flush=True,
    )

    results = []
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, max_tasks_per_child=1) as pool:
        for item in layers_info:
            lname = item["name"]
            w_quant = arrays[f"{lname}__w"]
            b_quant = arrays[f"{lname}__b"] if f"{lname}__b" in arrays else None
            w_frac = int(item["w_frac_bits"])
            scale_w = 2.0 ** w_frac
            w_int = np.round(w_quant * scale_w).astype(np.int64)

            in_dim, out_dim = w_quant.shape
            _, _, f_in = in_qfmt
            layer_dir = out_dir / lname
            layer_dir.mkdir(parents=True, exist_ok=True)

            rng = np.random.default_rng(42)
            test_x = rng.uniform(-1.0, 1.0, size=(8, in_dim)).astype(np.float32)
            scale_in = 2.0 ** (-f_in)
            test_x_q = np.round(test_x / scale_in) * scale_in
            da4ml_accum = np.zeros((8, out_dim), dtype=np.float32)
            if b_quant is not None:
                da4ml_accum += b_quant[None, :]

            tasks = []
            for i0 in range(0, in_dim, tile_size):
                i1 = min(i0 + tile_size, in_dim)
                for o0 in range(0, out_dim, tile_size):
                    o1 = min(o0 + tile_size, out_dim)
                    w_tile_int = w_int[i0:i1, o0:o1]
                    if not np.any(w_tile_int):
                        continue
                    tasks.append({
                        "w_tile_int": w_tile_int,
                        "frac_shift": w_frac,
                        "in_qformat": in_qfmt,
                        "mod_name": f"da4ml_{lname}_i{i0}_o{o0}",
                        "layer_dir": str(layer_dir),
                        "latency_cutoff": latency_cutoff,
                        "test_x_tile": test_x_q[:, i0:i1],
                        "o0": o0,
                        "o1": o1,
                    })

            total_adders = 0
            max_latency = 0
            tile_paths = []
            futures = [pool.submit(_compile_single_tile_task, t) for t in tasks]
            for fut in as_completed(futures):
                res = fut.result()
                total_adders += res["adders"]
                max_latency = max(max_latency, res["latency"])
                tile_paths.append(Path(res["v_path"]))
                da4ml_accum[:, res["o0"] : res["o1"]] += res["pred_out"]

            tile_paths.sort()
            ref_y = test_x_q @ w_quant + (b_quant if b_quant is not None else 0.0)
            max_err = float(np.max(np.abs(ref_y - da4ml_accum)))

            manifest_path = layer_dir / f"da4ml_{lname}_tiles.vh"
            manifest_path.write_text("".join(f'`include "{p.relative_to(layer_dir)}"' for p in tile_paths))

            nnz = int(np.count_nonzero(w_quant))
            sparsity = 100.0 * (1.0 - nnz / w_quant.size)
            info = {
                "layer": lname,
                "shape": f"{in_dim}x{out_dim}",
                "tiles": len(tasks),
                "sparsity_pct": sparsity,
                "adders": total_adders,
                "pipeline_stages": max_latency,
                "max_err": max_err,
                "manifest": str(manifest_path),
            }
            results.append(info)
            print(
                f"  {info['layer']:<22} | {info['shape']:<10} | tiles={info['tiles']:3d} | "
                f"sparsity={info['sparsity_pct']:5.1f}% | adders={info['adders']:6d} | "
                f"stages={info['pipeline_stages']:2d} | max_err={info['max_err']:.2e}",
                flush=True,
            )

    top_vh = out_dir / "timestral_da4ml_includes.vh"
    top_vh.write_text("".join(f'`include "{Path(r["manifest"]).relative_to(out_dir)}"' for r in results))
    print(f"Done. Generated {len(results)} tiled da4ml layer packages in {out_dir}/", flush=True)


def extract_quantized_weights(args) -> tuple[Path, Path]:
    """Phase 1: Load QKeras model, quantize weights, and serialize to .npz + .json."""
    from qkeras.range_analysis import optimize_bitwidths
    from safetensors.numpy import load_file

    from export_hls import parse_q_spec, quantize_array
    from timestral_kd.legacy import legacy_factory
    from timestral_kd.model import (
        TimestralSpec,
        build_timestral_multivariate,
        load_timestral_weights,
        scale_context_mv,
    )
    from forecast_csv import (
        CONTEXT_PATCHES,
        COVARIATE_COLUMNS,
        HORIZON_PATCHES,
        MODEL_DIR,
            TARGET_COLUMNS,
        build_sliding_windows,
        load_process_csv,
    )

    spec = TimestralSpec.from_json(MODEL_DIR / "config.json")
    ctx_len = CONTEXT_PATCHES * spec.patch_size
    hor_len = HORIZON_PATCHES * spec.patch_size

    covs = COVARIATE_COLUMNS if args.include_covariates else []
    df, active_cols = load_process_csv(args.csv, TARGET_COLUMNS, covs)
    num_variates = len(active_cols)
    data_matrix = df[active_cols].to_numpy(dtype=np.float32)
    ctx_windows, _, _ = build_sliding_windows(data_matrix, ctx_len, hor_len, stride=32)

    factory = legacy_factory()
    model = build_timestral_multivariate(
        spec,
        num_variates=num_variates,
        context_patches=CONTEXT_PATCHES,
        horizon_patches=HORIZON_PATCHES,
        factory=factory,
        name="timestral_mv_da4ml",
    )

    if args.weights is not None and args.weights.exists():
        model.load_weights(args.weights)
        print(f"Loaded fine-tuned/pruned weights from {args.weights}")
    else:
        weights = load_file(MODEL_DIR / "model.safetensors")
        load_timestral_weights(model, weights, spec)
        s_calib, _, _ = scale_context_mv(
            ctx_windows[:32], spec.patch_size, spec.minimum_scale, spec.use_asinh
        )
        optimize_bitwidths(model, data=s_calib, policy="keep_fractional", headroom=1, verbose=1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    in_bits, in_int, in_signed = parse_q_spec(factory.p["a"])
    in_qfmt = qkeras_to_da4ml_qformat(in_bits, in_int, in_signed)

    npz_dict = {}
    layers_meta = []
    for lyr in model.layers:
        if not hasattr(lyr, "kernel"):
            continue
        wq = getattr(lyr, "kernel_quantizer_internal", None)
        w_bits, w_int, w_signed = parse_q_spec(wq)
        _, _, w_frac = qkeras_to_da4ml_qformat(w_bits, w_int, w_signed)
        w_quant = quantize_array(lyr.kernel.numpy(), wq)
        npz_dict[f"{lyr.name}__w"] = w_quant

        if getattr(lyr, "bias", None) is not None:
            bq = getattr(lyr, "bias_quantizer_internal", None)
            b_quant = quantize_array(lyr.bias.numpy(), bq)
            npz_dict[f"{lyr.name}__b"] = b_quant

        layers_meta.append({"name": lyr.name, "w_frac_bits": w_frac})

    npz_path = args.out_dir / "_quantized_weights.npz"
    meta_path = args.out_dir / "_quantized_meta.json"
    np.savez(npz_path, **npz_dict)
    meta_path.write_text(json.dumps({"in_qformat": list(in_qfmt), "layers": layers_meta}, indent=2))
    return npz_path, meta_path


def main() -> None:
    from forecast_csv import DEFAULT_CSV

    parser = argparse.ArgumentParser(description="Compile Quantized Timestral Projections to HDL via da4ml")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Calibration CSV path")
    parser.add_argument("--weights", type=Path, default=None, help="Optional fine-tuned/pruned .weights.h5 path")
    parser.add_argument("--include-covariates", action="store_true", help="Include process covariates")
    parser.add_argument("--tile-size", type=int, default=16, help="Max tile dimension for da4ml DAG solver")
    parser.add_argument("--latency-cutoff", type=int, default=2, help="da4ml pipeline register stage cutoff")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 4), help="Parallel worker processes")
    parser.add_argument("--layers", nargs="*", default=None, help="Optional subset of layer names to compile")
    parser.add_argument("--out-dir", type=Path, default=Path("fpga_timestral_da4ml"), help="Output Verilog directory")
    parser.add_argument("--worker-npz", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-meta", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Phase 2: Clean worker process with zero TensorFlow/Keras loaded
    if args.worker_npz is not None and args.worker_meta is not None:
        run_da4ml_worker(
            npz_path=args.worker_npz,
            meta_path=args.worker_meta,
            out_dir=args.out_dir,
            tile_size=args.tile_size,
            latency_cutoff=args.latency_cutoff,
            target_layers=args.layers,
            jobs=args.jobs,
        )
        return

    # Phase 1: Extract quantized weights via QKeras, then spawn isolated da4ml worker
    npz_path, meta_path = extract_quantized_weights(args)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-npz",
        str(npz_path),
        "--worker-meta",
        str(meta_path),
        "--out-dir",
        str(args.out_dir),
        "--tile-size",
        str(args.tile_size),
        "--latency-cutoff",
        str(args.latency_cutoff),
        "--jobs",
        str(args.jobs),
    ]
    if args.layers:
        cmd.extend(["--layers", *args.layers])

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
