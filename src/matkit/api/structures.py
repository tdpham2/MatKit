"""Structure handoffs retain original files and explicit atom correspondence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import dict2constraint
from ase.io import read, write
from ase.io.cif import parse_cif
from ase.spacegroup import Spacegroup

from .models import StructureData, StructureRef


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable(value):
    """Convert supported scientific metadata; never silently drop an object."""
    if isinstance(value, np.ndarray):
        return portable(value.tolist())
    if isinstance(value, np.generic):
        return portable(value.item())
    if isinstance(value, Spacegroup):
        return {"number": value.no, "symbol": value.symbol}
    if isinstance(value, dict):
        return {str(k): portable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [portable(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        json.dumps(value, allow_nan=False)
        return value
    raise ValueError(f"Unsupported structure metadata: {type(value).__name__}")


def to_atoms(data: StructureData) -> Atoms:
    atoms = Atoms(
        numbers=data.atomic_numbers,
        positions=data.positions,
        cell=data.cell,
        pbc=data.pbc,
    )
    for key, value in data.arrays.items():
        if key in {"numbers", "positions"}:
            raise ValueError(
                "Species and positions cannot be overridden by arrays"
            )
        atoms.new_array(key, np.asarray(value))
    atoms.info = dict(data.info)
    try:
        atoms.set_constraint([dict2constraint(c) for c in data.constraints])
    except Exception as exc:
        raise ValueError(f"Unsupported constraint handoff: {exc}") from exc
    return atoms


def validate_geometry(atoms: Atoms) -> None:
    if (
        not len(atoms)
        or (atoms.numbers < 1).any()
        or (atoms.numbers > 118).any()
    ):
        raise ValueError("Structure must contain supported atomic species")
    if (
        not np.isfinite(atoms.positions).all()
        or not np.isfinite(atoms.cell.array).all()
    ):
        raise ValueError("Structure coordinates and cell must be finite")
    periodic = atoms.cell.array[atoms.pbc]
    if len(periodic) and np.linalg.matrix_rank(periodic) != len(periodic):
        raise ValueError(
            "Periodic cell vectors must be nonzero and independent"
        )


def load_structure(ref: StructureRef) -> tuple[Atoms, StructureData, str]:
    path = Path(ref.path).expanduser().resolve()
    digest = sha256(path)
    if ref.sha256 is not None and ref.sha256 != digest:
        raise ValueError("Structure hash does not match the supplied reference")
    frames = read(path, index=":")
    if len(frames) != 1:
        raise ValueError("Provide a single structure, not a trajectory")
    atoms = frames[0]
    validate_geometry(atoms)
    metadata = ref.metadata
    sidecar = path.with_suffix(".metadata.json")
    if metadata is None and sidecar.is_file():
        metadata = StructureData.model_validate_json(sidecar.read_text())
    if metadata is not None:
        if (
            metadata.atomic_numbers != atoms.numbers.tolist()
            or metadata.pbc != tuple(atoms.pbc)
            or not np.allclose(
                metadata.positions, atoms.positions, rtol=0, atol=1e-7
            )
            or not np.allclose(
                metadata.cell, atoms.cell.array, rtol=0, atol=1e-7
            )
        ):
            raise ValueError(
                "Structure metadata does not match geometry or atom order"
            )
        if any(source != digest for source in metadata.derived_from.values()):
            raise ValueError(
                "Derived results were invalidated by a structural change; "
                "recompute them"
            )
        return to_atoms(metadata), metadata, digest

    labels = None
    if path.suffix.lower() == ".cif":
        blocks = list(parse_cif(str(path)))
        if len(blocks) != 1:
            raise ValueError("Provide a single CIF data block")
        tags = dict(blocks[0])
        occupancy = tags.get("_atom_site_occupancy", [])
        if occupancy and not np.allclose(occupancy, 1):
            raise ValueError(
                "Disordered/partially occupied CIF sites are unsupported"
            )
        labels = tags.get("_atom_site_label")
        charges = tags.get("_atom_site_charge")
        if labels is not None and len(labels) != len(atoms):
            raise ValueError(
                "CIF symmetry expansion cannot preserve site correspondence; "
                "provide an explicit P1 structure"
            )
        if charges is not None:
            if len(charges) != len(atoms):
                raise ValueError(
                    "CIF charges cannot be mapped to expanded atoms"
                )
            atoms.set_initial_charges(charges)
        atoms.info["cif_tags"] = tags

    constraints = [portable(c.todict()) for c in atoms.constraints]
    data = StructureData(
        atomic_numbers=atoms.numbers.tolist(),
        positions=atoms.positions.tolist(),
        cell=atoms.cell.array.tolist(),
        pbc=atoms.pbc.tolist(),
        atom_ids=[f"{digest[:16]}:{i}" for i in range(len(atoms))],
        arrays={
            key: portable(value)
            for key, value in atoms.arrays.items()
            if key not in {"numbers", "positions"}
        },
        info=portable(atoms.info),
        constraints=constraints,
        labels=labels,
        derived_from={"charges": digest}
        if "initial_charges" in atoms.arrays
        else {},
    )
    # Round-trip constraints now, before accepting the structure.
    return to_atoms(data), data, digest


def final_structure(
    atoms: Atoms, original: StructureData, source_hash: str, path: Path
) -> StructureData:
    """Persist final geometry with metadata; invalidate inherited properties."""
    validate_geometry(atoms)
    if atoms.numbers.tolist() != original.atomic_numbers:
        raise ValueError("Calculator changed species or atom correspondence")
    changed = (
        not np.array_equal(atoms.positions, original.positions)
        or not np.array_equal(atoms.cell.array, original.cell)
        or tuple(atoms.pbc) != original.pbc
    )
    values = original.model_dump()
    values.update(
        positions=atoms.positions.tolist(),
        cell=atoms.cell.array.tolist(),
        pbc=atoms.pbc.tolist(),
        parent_sha256=source_hash,
    )
    if changed:
        values["derived_from"] = {}
        for key in ("initial_charges", "forces", "energies", "stresses"):
            values["arrays"].pop(key, None)
        for key in (
            "energy",
            "potential_energy",
            "forces",
            "stress",
            "charges",
            "pore_properties",
            "cif_tags",
            "spacegroup",
            "occupancy",
        ):
            values["info"].pop(key, None)
    # The sidecar carries arrays and constraints even when the file cannot.
    geometry = Atoms(
        numbers=atoms.numbers,
        positions=atoms.positions,
        cell=atoms.cell,
        pbc=atoms.pbc,
    )
    write(path, geometry, format="extxyz")
    if not changed:
        values["derived_from"] = {
            name: sha256(path) for name in original.derived_from
        }
    result = StructureData.model_validate(values)
    path.with_suffix(".metadata.json").write_text(
        result.model_dump_json(indent=2)
    )
    return result
