#!/usr/bin/env python3
"""Run Zeo++ pore analyses for one structure or a homogeneous batch."""

import argparse
from pathlib import Path
import sys

from matkit.api import (
    ExecutionConfig,
    PoreRequest,
    StructureRef,
    analyze_pores,
    run_batch,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Periodic structure files.")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--execution", type=Path, help="Execution profile JSON."
    )
    parser.add_argument(
        "--analysis",
        action="append",
        choices=("res", "sa", "vol", "psd", "chan"),
    )
    parser.add_argument("--probe-radius", type=float, default=1.86)
    parser.add_argument("--channel-radius", type=float, default=1.86)
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--radii", help="Defaults to MatKit's bundled UFF.rad.")
    args = parser.parse_args(argv)

    try:
        execution = (
            ExecutionConfig.model_validate_json(args.execution.read_text())
            if args.execution
            else ExecutionConfig()
        ).model_copy(update={"mode": "subprocess"})
        requests = [
            PoreRequest(
                structure=StructureRef(path=path),
                analyses=args.analysis or ["res", "sa", "vol", "psd", "chan"],
                probe_radius=args.probe_radius,
                channel_radius=args.channel_radius,
                num_samples=args.num_samples,
                radii_file=args.radii,
            )
            for path in args.inputs
        ]
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    try:
        if len(requests) == 1:
            result = analyze_pores(
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
