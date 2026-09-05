#!/usr/bin/env python3
"""Prepare a CO2/N2 mixture with explicit fractions; no engine execution."""

import argparse
from pathlib import Path
import sys

from matkit.graspa import setup_simulation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cif", help="Charged periodic framework CIF.")
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        args.outdir.mkdir(parents=True, exist_ok=False)
        setup_simulation(
            cif=args.cif,
            outpath=str(args.outdir),
            adsorbates=[
                {"MoleculeName": "CO2", "MolFraction": 0.15},
                {"MoleculeName": "N2", "MolFraction": 0.85},
            ],
            temperature=298.0,
            pressure=100000.0,
            cutoff=12.8,
            n_cycle=10000,
            template_dir="template",
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Prepared mixture input: {args.outdir / 'simulation.input'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
