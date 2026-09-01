#!/usr/bin/env python3
"""Run one MatKit MLIP backend on one or more structures using a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matkit.mlip import (
    ASEMACEConfig,
    MLIPCalculationConfig,
    NVAlchemiMACEConfig,
    RootstockConfig,
    run_mlip_batch,
)


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
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Native batch size for ALCHEMI; accepted by all backends.",
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
        help="Enable model compilation for NVIDIA ALCHEMI MACE.",
    )
    parser.add_argument(
        "--enable-cueq",
        action="store_true",
        help="Enable CuEquivariance for NVIDIA ALCHEMI MACE.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.cluster and args.root:
        parser.error("--cluster and --root are mutually exclusive")

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
            compile_model=args.compile_model,
            enable_cueq=args.enable_cueq,
        )

    calculation = MLIPCalculationConfig(
        driver=args.driver,
        fmax=args.fmax,
        steps=args.steps,
    )
    output_dir = args.output_dir or Path("mlip_gpu_results") / args.backend
    summary = run_mlip_batch(
        args.inputs,
        backend,
        calculation,
        output_dir=output_dir,
        batch_size=args.batch_size,
    )

    print(
        json.dumps(
            {
                "status": summary["status"],
                "backend": args.backend,
                "total": summary["total"],
                "succeeded": summary["succeeded"],
                "failed": summary["failed"],
                "manifest_file": summary["manifest_file"],
            },
            indent=2,
        )
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
