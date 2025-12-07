#!/usr/bin/env python
"""
Minimal PubChem JSON → XYZ converter using ASE.

Assumes a PubChem PC_Compounds JSON with a 2D/3D conformer.
"""

import json
import sys
from pathlib import Path

from ase import Atoms
from ase.io import write


def pubchem_json_to_atoms(json_file: str | Path) -> Atoms:
    """Load PubChem PC_Compounds JSON and return an ASE Atoms object."""
    with open(json_file, "r") as f:
        data = json.load(f)

    compound = data["PC_Compounds"][0]

    # atomic numbers (ASE understands these directly)
    numbers = compound["atoms"]["element"]

    # coordinates (take first coord set and first conformer)
    coord_block = compound["coords"][0]
    conf = coord_block["conformers"][0]

    x = conf["x"]
    y = conf["y"]
    z = conf.get("z", [0.0] * len(x))  # PubChem 2D → z = 0

    positions = list(zip(x, y, z))

    atoms = Atoms(numbers=numbers, positions=positions)

    # minimal metadata
    try:
        atoms.info["pubchem_cid"] = compound["id"]["id"]["cid"]
    except Exception:
        pass

    return atoms


def main():
    if len(sys.argv) != 3:
        print("Usage: convert_pubchem_json.py input.json output.xyz")
        sys.exit(1)

    json_file = sys.argv[1]
    xyz_file = sys.argv[2]

    atoms = pubchem_json_to_atoms(json_file)
    write(xyz_file, atoms)


if __name__ == "__main__":
    main()
