#!/usr/bin/env python3
"""Evaluate or relax structures with MACE, Rootstock, or ALCHEMI."""

import argparse
from pathlib import Path
import sys

from matkit.api import (
    AlchemiAdapter,
    EvaluateRequest,
    ExecutionConfig,
    MACEAdapter,
    MLIPMethod,
    RelaxRequest,
    RootstockAdapter,
    StructureRef,
    evaluate,
    relax,
    run_batch,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", help="One structure per input file."
    )
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--execution", type=Path, help="Execution profile JSON."
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=("ase-mace", "rootstock", "nvalchemi-mace"),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--driver", choices=("energy", "relax"), default="energy"
    )
    parser.add_argument("--dtype", choices=("float32", "float64"))
    location = parser.add_mutually_exclusive_group()
    location.add_argument("--cluster", help="Rootstock deployment cluster.")
    location.add_argument("--root", help="Rootstock deployment directory.")
    parser.add_argument(
        "--batch-size", type=int, help="ALCHEMI native chunk size."
    )
    parser.add_argument(
        "--fmax", type=float, help="Relaxation force tolerance (eV/A)."
    )
    parser.add_argument("--steps", type=int, help="Maximum relaxation steps.")
    parser.add_argument(
        "--property",
        action="append",
        choices=("potential_energy", "forces", "stress"),
    )
    args = parser.parse_args(argv)
    if args.backend != "rootstock" and (args.cluster or args.root):
        parser.error("--cluster and --root require --backend rootstock")
    if args.backend == "rootstock" and args.dtype:
        parser.error("Rootstock precision is configured in its deployment")
    if args.backend != "nvalchemi-mace" and args.batch_size is not None:
        parser.error("--batch-size requires --backend nvalchemi-mace")
    if args.driver == "energy" and (
        args.fmax is not None or args.steps is not None
    ):
        parser.error("--fmax and --steps require --driver relax")
    if args.driver == "relax" and args.property:
        parser.error("Relaxation requests energy and forces automatically")

    try:
        execution = (
            ExecutionConfig.model_validate_json(args.execution.read_text())
            if args.execution
            else ExecutionConfig()
        ).model_copy(update={"mode": "subprocess"})
        if args.backend == "ase-mace":
            adapter = MACEAdapter(dtype=args.dtype)
        elif args.backend == "rootstock":
            adapter = RootstockAdapter(cluster=args.cluster, root=args.root)
        else:
            adapter = AlchemiAdapter(
                dtype=args.dtype or "float32",
                batch_size=args.batch_size
                if args.batch_size is not None
                else 16,
            )
        method = MLIPMethod(checkpoint=args.checkpoint)
        if args.driver == "relax":
            request_type, operation = RelaxRequest, relax
            settings = {
                "fmax": args.fmax if args.fmax is not None else 0.01,
                "steps": args.steps if args.steps is not None else 1000,
            }
        else:
            request_type, operation = EvaluateRequest, evaluate
            settings = {
                "properties": args.property or ["potential_energy", "forces"]
            }
        requests = [
            request_type(
                structure=StructureRef(path=path),
                method=method,
                adapter=adapter,
                **settings,
            )
            for path in args.inputs
        ]
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    try:
        if len(requests) == 1:
            result = operation(
                requests[0], output_dir=args.outdir, execution=execution
            )
        else:
            result = run_batch(
                requests, output_dir=args.outdir, execution=execution
            )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2))
    return 0 if result.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
