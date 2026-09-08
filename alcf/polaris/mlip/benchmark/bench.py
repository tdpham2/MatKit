#!/usr/bin/env python3
"""Throughput benchmark for MatKit MLIP backends on a single GPU.

Unlike ``smoke.py`` (correctness/integration evidence), this script measures
performance: how many structures per second a backend sustains, how timing
scales with structure size, and how NVIDIA ALCHEMI native batching compares to
sequential ASE MACE. Results are hardware- and checkpoint-specific; record the
GPU model, MatKit commit, and package versions alongside any numbers.

Each sweep runs ``run_mlip_batch`` in-process and reads timings from the
returned manifest. One backend per process keeps GPU runtime state isolated.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import subprocess
import sys
import time
from pathlib import Path

from ase.build import make_supercell
from ase.io import read, write
import numpy as np

from matkit.mlip import (
    ASEMACEConfig,
    MLIPCalculationConfig,
    NVAlchemiMACEConfig,
    run_mlip_batch,
)


def _backend_config(name, checkpoint, dtype):
    """Build a backend config with GPU defaults matched across backends."""
    if name == "ase-mace":
        return ASEMACEConfig(
            checkpoint=checkpoint or "medium", device="cuda", dtype=dtype
        )
    if name == "nvalchemi-mace":
        return NVAlchemiMACEConfig(
            checkpoint=checkpoint or "medium", device="cuda", dtype=dtype
        )
    raise ValueError(f"Unsupported benchmark backend: {name}")


def _gpu_memory_mib():
    """Best-effort peak used GPU memory via nvidia-smi; None if unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    values = [
        int(line) for line in result.stdout.split("\n") if line.strip().isdigit()
    ]
    return max(values) if values else None


def _gpu_name():
    """Best-effort GPU model name via nvidia-smi; None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _run_once(input_files, backend, calculation, output_dir, batch_size):
    """Run one batch and summarize timing from the manifest and results."""
    started = time.perf_counter()
    summary = run_mlip_batch(
        input_files,
        backend,
        calculation=calculation,
        output_dir=output_dir,
        batch_size=batch_size,
    )
    elapsed = time.perf_counter() - started
    results = summary["results"]
    calc_times = [
        r["calculation_time_s"]
        for r in results
        if r.get("success") and r.get("calculation_time_s") is not None
    ]
    succeeded = summary["succeeded"]
    throughput = succeeded / elapsed if elapsed > 0 else 0.0
    return {
        "status": summary["status"],
        "total": summary["total"],
        "succeeded": succeeded,
        "failed": summary["failed"],
        "setup_time_s": summary["setup_time_s"],
        "wall_time_s": summary["wall_time_s"],
        "elapsed_s": elapsed,
        "throughput_structs_per_s": throughput,
        "calc_time_s_mean": float(np.mean(calc_times)) if calc_times else None,
        "calc_time_s_max": float(np.max(calc_times)) if calc_times else None,
        "peak_gpu_mem_mib": _gpu_memory_mib(),
        "manifest_file": summary["manifest_file"],
    }


def _write_row(handle, row):
    handle.write(json.dumps(row, allow_nan=False) + "\n")
    handle.flush()


def _prepare_replicas(base_atoms, count, work_dir):
    """Write ``count`` copies of a base structure; return their paths."""
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = work_dir / f"struct_{index:04d}.extxyz"
        write(path, base_atoms)
        paths.append(str(path))
    return paths


def _prepare_sizes(base_atoms, factors, work_dir):
    """Write supercells scaled by cubic diagonal factors; return (path, n)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for factor in factors:
        supercell = make_supercell(base_atoms, np.diag([factor, factor, factor]))
        path = work_dir / f"size_{len(supercell):06d}.extxyz"
        write(path, supercell)
        entries.append((str(path), len(supercell)))
    return entries


def _sweep_batch_size(args, base_atoms, out, handle):
    """Fixed structure set; vary ALCHEMI batch size (ASE ignores batch)."""
    input_files = _prepare_replicas(
        base_atoms, args.n_structures, out / "inputs_batch"
    )
    for backend_name in args.backends:
        backend = _backend_config(backend_name, args.checkpoint, args.dtype)
        calculation = MLIPCalculationConfig(driver=args.driver, steps=args.steps)
        sizes = args.batch_sizes if backend_name == "nvalchemi-mace" else [1]
        for batch_size in sizes:
            run_dir = out / "runs" / f"batch_{backend_name}_bs{batch_size}"
            metrics = _run_once(
                input_files, backend, calculation, run_dir, batch_size
            )
            _write_row(
                handle,
                {
                    "sweep": "batch_size",
                    "backend": backend_name,
                    "driver": args.driver,
                    "checkpoint": args.checkpoint,
                    "dtype": args.dtype,
                    "batch_size": batch_size,
                    "n_structures": args.n_structures,
                    "n_atoms": len(base_atoms),
                    **metrics,
                },
            )
            print(
                f"[batch_size] {backend_name} bs={batch_size}: "
                f"{metrics['throughput_structs_per_s']:.2f} struct/s"
            )


def _sweep_structure_size(args, base_atoms, out, handle):
    """One structure per size; measure time vs atom count."""
    entries = _prepare_sizes(base_atoms, args.size_factors, out / "inputs_size")
    for backend_name in args.backends:
        backend = _backend_config(backend_name, args.checkpoint, args.dtype)
        calculation = MLIPCalculationConfig(driver=args.driver, steps=args.steps)
        for path, n_atoms in entries:
            run_dir = out / "runs" / f"size_{backend_name}_{n_atoms}"
            metrics = _run_once([path], backend, calculation, run_dir, 1)
            _write_row(
                handle,
                {
                    "sweep": "structure_size",
                    "backend": backend_name,
                    "driver": args.driver,
                    "checkpoint": args.checkpoint,
                    "dtype": args.dtype,
                    "batch_size": 1,
                    "n_structures": 1,
                    "n_atoms": n_atoms,
                    **metrics,
                },
            )
            print(
                f"[structure_size] {backend_name} n_atoms={n_atoms}: "
                f"{metrics['calc_time_s_mean']} s"
            )


def _package_versions():
    packages = {}
    for name in ("matkit", "ase", "mace-torch", "nvalchemi-toolkit", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return packages


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--sweeps",
        nargs="+",
        default=["batch_size", "structure_size"],
        choices=["batch_size", "structure_size"],
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["nvalchemi-mace", "ase-mace"],
        choices=["nvalchemi-mace", "ase-mace"],
        help="ase-mace serves as the sequential baseline.",
    )
    parser.add_argument("--checkpoint", default="medium")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--driver", default="energy", choices=["energy", "opt"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--n-structures",
        type=int,
        default=64,
        help="Structure count for the batch-size sweep.",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64],
    )
    parser.add_argument(
        "--size-factors",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Cubic supercell factors for the structure-size sweep.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.n_structures < 1 or args.steps < 1:
        raise SystemExit("--n-structures and --steps must be positive")
    if any(size < 1 for size in args.batch_sizes):
        raise SystemExit("--batch-sizes must be positive")
    if any(factor < 1 for factor in args.size_factors):
        raise SystemExit("--size-factors must be positive")
    out = args.output_dir.resolve()
    if out.exists():
        raise SystemExit("--output-dir must be a new directory")
    out.mkdir(parents=True)

    base_atoms = read(args.input)
    meta = {
        "input_file": str(args.input.resolve()),
        "base_n_atoms": len(base_atoms),
        "packages": _package_versions(),
        "gpu": _gpu_name(),
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
    }
    (out / "bench_meta.json").write_text(json.dumps(meta, indent=2))

    results_path = out / "bench_results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        if "batch_size" in args.sweeps:
            _sweep_batch_size(args, base_atoms, out, handle)
        if "structure_size" in args.sweeps:
            _sweep_structure_size(args, base_atoms, out, handle)

    print(f"Benchmark results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
