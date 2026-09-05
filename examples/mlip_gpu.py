#!/usr/bin/env python3
"""Run one MatKit MLIP backend on one or more structures using a GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from matkit.mlip import (
    ASEMACEConfig,
    MLIPCalculationConfig,
    NVAlchemiMACEConfig,
    RootstockConfig,
    run_mlip_batch,
)
from matkit.mlip.config import _validate_explicit_options


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run direct ASE MACE, Rootstock, or NVIDIA ALCHEMI MACE on a GPU. "
            "Use one backend per process to keep GPU runtime state isolated."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Structure files readable by ASE.",
    )
    parser.add_argument(
        "--backend",
        choices=("ase-mace", "rootstock", "nvalchemi-mace"),
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        help=(
            "Model alias or checkpoint path. Defaults to 'medium' for MACE and "
            "ALCHEMI, or 'mace-mp-0-medium' for Rootstock."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Result directory (default: mlip_gpu_results/<backend>).",
    )
    parser.add_argument("--driver", choices=("energy", "opt"), default="energy")
    parser.add_argument(
        "--fmax", type=float, help="Optimization only (default: 0.01)."
    )
    parser.add_argument(
        "--steps", type=int, help="Optimization only (default: 1000)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Native batch size for ALCHEMI only (default: 16).",
    )
    parser.add_argument(
        "--cluster",
        help="Rootstock cluster name (defaults to 'polaris').",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Rootstock deployment root instead of a named cluster.",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        default=None,
        help="Enable model compilation for NVIDIA ALCHEMI MACE.",
    )
    parser.add_argument(
        "--enable-cueq",
        action="store_true",
        default=None,
        help="Enable CuEquivariance for NVIDIA ALCHEMI MACE.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.cluster and args.root:
        parser.error("--cluster and --root are mutually exclusive")

    provided = {
        "root_path" if name == "root" else name
        for name, value in vars(args).items()
        if value is not None
    }
    try:
        _validate_explicit_options(
            args.backend, args.driver, "mace_mp", provided
        )
        calculation = MLIPCalculationConfig(
            driver=args.driver,
            fmax=args.fmax if args.fmax is not None else 0.01,
            steps=args.steps if args.steps is not None else 1000,
        )
        if args.batch_size is not None and args.batch_size < 1:
            raise ValueError("--batch-size must be at least 1")
    except ValueError as exc:
        parser.error(str(exc))

    if args.backend == "ase-mace":
        backend = ASEMACEConfig(
            checkpoint=args.checkpoint or "medium",
            device="cuda",
            dtype="float32",
        )
    elif args.backend == "rootstock":
        backend = RootstockConfig(
            checkpoint=args.checkpoint or "mace-mp-0-medium",
            cluster=None if args.root else (args.cluster or "polaris"),
            root=str(args.root) if args.root else None,
            device="cuda",
        )
    else:
        backend = NVAlchemiMACEConfig(
            checkpoint=args.checkpoint or "medium",
            device="cuda",
            dtype="float32",
            compile_model=bool(args.compile_model),
            enable_cueq=bool(args.enable_cueq),
        )

    output_dir = args.output_dir or Path("mlip_gpu_results") / args.backend
    try:
        summary = run_mlip_batch(
            args.inputs,
            backend,
            calculation,
            output_dir=output_dir,
            batch_size=args.batch_size if args.batch_size is not None else 16,
        )
    except Exception as exc:
        print(json.dumps({"status": "failure", "error": str(exc)}, indent=2))
        print(str(exc), file=sys.stderr)
        return 1
    unconverged = (
        sum(
            result["success"] and not result["converged"]
            for result in summary["results"]
        )
        if calculation.driver == "opt"
        else 0
    )

    print(
        json.dumps(
            {
                "status": summary["status"],
                "backend": args.backend,
                "total": summary["total"],
                "succeeded": summary["succeeded"],
                "failed": summary["failed"],
                "unconverged": unconverged,
                "manifest_file": summary["manifest_file"],
            },
            indent=2,
        )
    )
    if summary["failed"] or unconverged:
        print(
            f"{summary['failed']} failed, {unconverged} unconverged; "
            "available results retained.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
