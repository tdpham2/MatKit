#!/usr/bin/env python3
"""Prepare or run pure-component gRASPA at one or more pressures."""

import argparse
from pathlib import Path
import sys

from matkit.api import (
    AdsorptionRequest,
    ExecutionConfig,
    StructureRef,
    prepare_adsorption,
    run_adsorption,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cif", help="Atom-mapped, charged, periodic CIF.")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--execution", type=Path, help="Execution profile JSON."
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--adsorbate", default="CO2")
    parser.add_argument("--temperature-k", type=float, default=298.0)
    parser.add_argument(
        "--pressure-pa", action="append", type=float, help="Repeat for a sweep."
    )
    parser.add_argument("--cutoff", type=float, default=12.8)
    parser.add_argument("--initialization-cycles", type=int, default=1000)
    parser.add_argument("--equilibration-cycles", type=int, default=1000)
    parser.add_argument("--production-cycles", type=int, default=10000)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--net-charge", type=float, default=0.0)
    parser.add_argument(
        "--unit", choices=("mol/kg", "mg/g", "g/L"), default="mol/kg"
    )
    parser.add_argument(
        "--template-dir", help="Complete custom template directory."
    )
    args = parser.parse_args(argv)
    if args.prepare_only and args.execution:
        parser.error("Preparation does not use an execution profile")

    try:
        execution = (
            ExecutionConfig.model_validate_json(args.execution.read_text())
            if args.execution
            else ExecutionConfig()
        ).model_copy(update={"mode": "subprocess"})
        requests = [
            AdsorptionRequest(
                structure=StructureRef(path=args.cif),
                adsorbate=args.adsorbate,
                temperature_K=args.temperature_k,
                pressure_Pa=pressure,
                cutoff_angstrom=args.cutoff,
                initialization_cycles=args.initialization_cycles,
                equilibration_cycles=args.equilibration_cycles,
                production_cycles=args.production_cycles,
                number_of_blocks=args.blocks,
                fugacity_coefficient="PR-EOS",
                net_charge=args.net_charge,
                unit=args.unit,
                template_dir=args.template_dir,
            )
            for pressure in args.pressure_pa or [100000.0]
        ]
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    try:
        args.outdir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f"Use a fresh output directory: {exc}", file=sys.stderr)
        return 1

    succeeded = True
    # Different pressures cannot share a homogeneous MatKit batch.
    # Each public API call owns preparation, execution, logs, and results.
    for index, request in enumerate(requests):
        bundle = args.outdir / f"{index:05d}"
        try:
            if args.prepare_only:
                result = prepare_adsorption(request, output_dir=bundle)
            else:
                result = run_adsorption(
                    request, output_dir=bundle, execution=execution
                )
            print(result.model_dump_json())  # One run record per stdout line.
            succeeded = succeeded and (
                result.state == "prepared"
                if args.prepare_only
                else result.accepted
            )
        except Exception as exc:
            print(f"{bundle}: {exc}", file=sys.stderr)
            succeeded = False
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
