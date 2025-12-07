#!/usr/bin/env python
"""
Minimal SMILES → XYZ converter using RDKit + ASE.

Flow:
    SMILES → RDKit Mol → 3D coordinates → ASE Atoms → ase.io.write
"""

import sys
from typing import List

from ase import Atoms
from ase.io import write

from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_atoms(smiles: str) -> Atoms:
    """Convert a SMILES string to an ASE Atoms object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Add hydrogens and generate 3D geometry
    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()

    numbers: List[int] = []
    positions = []

    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        numbers.append(atom.GetAtomicNum())
        positions.append((pos.x, pos.y, pos.z))

    return Atoms(numbers=numbers, positions=positions)


def main():
    if len(sys.argv) != 3:
        print("Usage: convert_smiles_to_xyz.py '<SMILES>' output.xyz")
        sys.exit(1)

    smiles = sys.argv[1]
    output = sys.argv[2]

    atoms = smiles_to_atoms(smiles)
    write(output, atoms)


if __name__ == "__main__":
    main()
