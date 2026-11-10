"""Read physical geometries from gRASPA's LAMMPS full-style movies.

Atom identities come from pseudo-atom names, never inferred atomic masses.
Simulation charges and molecule identifiers remain provenance, not ASE arrays.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.geometry import find_mic
from ase.io import read
from scipy.spatial import cKDTree


_ALIASES = {
    "Ow": "O",
    "Hw": "H",
    "C_co2": "C",
    "O_co2": "O",
    "N_n2": "N",
    "H_h2": "H",
}
_VIRTUAL = {"Lw", "N_com", "H_com"}


@dataclass(frozen=True)
class MovieSite:
    """A source site; list order in MovieSnapshot matches the ASE atom order."""

    atom_id: int
    molecule_id: int
    type_id: int
    pseudo_atom: str
    component: str | None
    charge: float
    image_flags: tuple[int, int, int]


@dataclass
class MovieSnapshot:
    atoms: Atoms
    sites: list[MovieSite]
    removed_sites: list[MovieSite]
    snapshot_atoms: int
    origin: list[float]
    source_sha256: str
    water_molecules: int

    @property
    def component_counts(self) -> dict[str, int]:
        molecules = defaultdict(set)
        for site in self.sites:
            molecules[site.component or "unknown"].add(site.molecule_id)
        return {name: len(ids) for name, ids in sorted(molecules.items())}

    def provenance(self) -> dict:
        def record(site):
            return {**asdict(site), "image_flags": list(site.image_flags)}

        return {
            "snapshot_sha256": self.source_sha256,
            "snapshot_atoms": self.snapshot_atoms,
            "physical_atoms": len(self.atoms),
            "water_molecules": self.water_molecules,
            "dummy_sites_removed": len(self.removed_sites),
            "component_counts": self.component_counts,
            "origin": self.origin,
            "sites": [record(site) for site in self.sites],
            "removed_sites": [record(site) for site in self.removed_sites],
        }


def _data_lines(path):
    return [
        fields
        for line in Path(path).read_text().splitlines()
        if (fields := line.partition("#")[0].split())
    ]


def _pseudo_definitions(path):
    if path is None:
        return {}
    lines = _data_lines(path)
    if not lines or len(lines[0]) != 1:
        raise ValueError("Invalid pseudo_atoms.def header")
    count = int(lines[0][0])
    definitions = {}
    for fields in lines[1:]:
        if len(fields) < 7 or fields[1].lower() not in {"yes", "no"}:
            raise ValueError("Invalid pseudo_atoms.def row")
        name = fields[0]
        if name in definitions:
            raise ValueError(f"Duplicate pseudo-atom definition: {name}")
        mass, charge = float(fields[5]), float(fields[6])
        if not np.isfinite([mass, charge]).all() or mass < 0:
            raise ValueError(f"Invalid pseudo-atom mass or charge: {name}")
        definitions[name] = (fields[3], mass)
    if count != len(definitions):
        raise ValueError("Incomplete pseudo_atoms.def table")
    return definitions


def _type_map(value):
    if value is None:
        return {}
    if not isinstance(value, dict) or any(
        not isinstance(name, str)
        or not name
        or (
            symbol is not None
            and (
                not isinstance(symbol, str)
                or symbol not in atomic_numbers
                or symbol == "X"
            )
        )
        for name, symbol in value.items()
    ):
        raise ValueError("type_map must map pseudo-atom names to elements/null")
    return dict(value)


def _element(name, mass, definitions, overrides):
    definition = definitions.get(name)
    if definitions and definition is None:
        raise ValueError(f"Missing pseudo-atom definition: {name}")
    if definition and not np.isclose(mass, definition[1], atol=0.01, rtol=1e-4):
        raise ValueError(f"Movie and pseudo-atom masses disagree: {name}")
    # Mass is a consistency check, never a way of choosing an element.
    if re.match(r"^(?:CH[1-4]|NH[1-4])(?:_|$)", name):
        raise ValueError(f"Unsupported united-atom representation: {name}")
    virtual = name in _VIRTUAL or (
        definition is not None and definition[0] == "-" and mass == 0
    )
    if virtual:
        if mass != 0 or overrides.get(name) is not None:
            raise ValueError(f"Inconsistent virtual site: {name}")
        return None
    if name in overrides:
        symbol = overrides[name]
    elif definition:
        symbol = definition[0]
    else:
        symbol = _ALIASES.get(name, name)
    if symbol is None:
        if mass != 0:
            raise ValueError(f"Cannot discard a massive physical site: {name}")
        return None
    if symbol not in atomic_numbers or symbol == "X" or mass <= 0:
        raise ValueError(f"Unsupported physical species: {name}")
    expected_mass = atomic_masses[atomic_numbers[symbol]]
    if abs(mass - expected_mass) > max(0.5, expected_mass * 0.02):
        raise ValueError(
            f"Mass inconsistent with element {symbol}: {name}; "
            "united-atom/coarse-grained sites are unsupported"
        )
    return symbol


def _parse_table(text):
    bounds, tilt, types, rows = {}, [0.0, 0.0, 0.0], {}, {}
    count = type_count = None
    section = None
    seen_sections = set()
    for line_number, raw in enumerate(text.splitlines(), 1):
        body, _, comment = raw.partition("#")
        fields = body.split()
        if not fields:
            continue
        try:
            if len(fields) == 2 and fields[1] == "atoms":
                if count is not None:
                    raise ValueError("Duplicate atom count")
                count = int(fields[0])
            elif fields[-2:] == ["atom", "types"]:
                type_count = int(fields[0])
            elif fields[-2:] in (
                ["xlo", "xhi"],
                ["ylo", "yhi"],
                ["zlo", "zhi"],
            ):
                axis = fields[-2][0]
                if axis in bounds or len(fields) != 4:
                    raise ValueError("Invalid or duplicate cell bounds")
                bounds[axis] = tuple(map(float, fields[:2]))
            elif fields[-3:] == ["xy", "xz", "yz"]:
                if len(fields) != 6:
                    raise ValueError("Invalid tilt factors")
                tilt = list(map(float, fields[:3]))
            elif fields[0][0].isalpha():
                section = (
                    fields[0] if fields[0] in {"Masses", "Atoms"} else None
                )
                if section:
                    if section in seen_sections:
                        raise ValueError(f"Duplicate {section} section")
                    seen_sections.add(section)
                    if section == "Atoms" and comment.strip() not in {
                        "",
                        "full",
                    }:
                        raise ValueError(
                            "Only full-style atom tables supported"
                        )
            elif section == "Masses":
                if len(fields) != 2 or not comment.split():
                    raise ValueError("Missing pseudo-atom name in Masses")
                type_id, mass = int(fields[0]), float(fields[1])
                if type_id <= 0 or type_id in types:
                    raise ValueError("Invalid or duplicate atom type")
                if not np.isfinite(mass) or mass < 0:
                    raise ValueError("Invalid atom mass")
                types[type_id] = (comment.split()[0], mass)
            elif section == "Atoms":
                if len(fields) not in {7, 10}:
                    raise ValueError("Expected full-style atom rows")
                atom_id, molecule, type_id = map(int, fields[:3])
                if atom_id <= 0 or molecule < 0 or atom_id in rows:
                    raise ValueError("Invalid or duplicate atom ID/molecule ID")
                values = list(map(float, fields[3:7]))
                if not np.isfinite(values).all():
                    raise ValueError("Non-finite coordinates or charge")
                images = (
                    tuple(map(int, fields[7:]))
                    if len(fields) == 10
                    else (0, 0, 0)
                )
                rows[atom_id] = (
                    molecule,
                    type_id,
                    values,
                    images,
                    comment.split(),
                )
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Line {line_number}: {exc}") from exc
    if count is None or count < 1 or len(rows) != count:
        raise ValueError("Incomplete snapshot atom table")
    if type_count is None or type_count != len(types):
        raise ValueError("Incomplete atom type table")
    if set(bounds) != {"x", "y", "z"}:
        raise ValueError("Missing cell bounds")
    origin = np.array([bounds[axis][0] for axis in "xyz"])
    lengths = [bounds[axis][1] - bounds[axis][0] for axis in "xyz"]
    lx, ly, lz = lengths
    xy, xz, yz = tilt
    cell = np.array([[lx, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, lz]])
    if (
        not np.isfinite(cell).all()
        or not np.isfinite(origin).all()
        or min(lengths) <= 0
        or np.linalg.matrix_rank(cell) != 3
    ):
        raise ValueError("Invalid cell")
    return count, origin, cell, types, rows


def _check_water(atoms, sites, removed):
    groups = defaultdict(list)
    for index, site in enumerate(sites):
        groups[site.molecule_id].append((index, site))
    for site in removed:
        groups[site.molecule_id].append((None, site))
    waters = 0
    for molecule, entries in groups.items():
        components = {site.component for _, site in entries}
        if len(components) != 1:
            raise ValueError("Different components share a molecule ID")
        names = Counter(site.pseudo_atom for _, site in entries)
        component = next(iter(components))
        water = component == "TIP4P" or (
            component is None and bool(set(names) & {"Ow", "Hw", "Lw"})
        )
        if not water:
            continue
        if (
            names["Ow"] != 1
            or names["Hw"] != 2
            or names["Lw"] not in {0, 1}
            or set(names) - {"Ow", "Hw", "Lw"}
        ):
            raise ValueError(f"Incomplete TIP4P molecule {molecule}")
        oxygen = next(i for i, s in entries if s.pseudo_atom == "Ow")
        hydrogens = [i for i, s in entries if s.pseudo_atom == "Hw"]
        vectors, lengths = find_mic(
            atoms.positions[hydrogens] - atoms.positions[oxygen],
            atoms.cell,
            True,
        )
        if not np.allclose(lengths, 0.9572, atol=0.02, rtol=0):
            raise ValueError(f"Invalid O-H distances in water {molecule}")
        cosine = np.dot(*vectors) / np.prod(lengths)
        angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        if abs(angle - 104.52) > 1:
            raise ValueError(f"Invalid H-O-H angle in water {molecule}")
        frac = np.linalg.solve(atoms.cell.array.T, atoms.positions[oxygen])
        atoms.positions[oxygen] -= np.floor(frac) @ atoms.cell.array
        atoms.positions[hydrogens] = atoms.positions[oxygen] + vectors
        waters += 1
    return waters


def read_movie_snapshot(path, *, pseudo_atoms=None, type_map=None):
    """Read one full-style data file into a geometry-only MovieSnapshot.

    ``pseudo_atoms`` is an optional pseudo_atoms.def path. ``type_map`` maps
    pseudo-atom names to element symbols, or null for massless virtual sites.
    Image flags are preserved as provenance; coordinates are periodic and
    TIP4P molecules are made whole around their oxygen. No charges, rigid
    constraints, or provenance arrays are attached to the returned Atoms.
    """
    content = Path(path).read_bytes()
    count, origin, cell, types, rows = _parse_table(content.decode())
    definitions = _pseudo_definitions(pseudo_atoms)
    overrides = _type_map(type_map)
    symbols, positions, sites, removed = [], [], [], []
    elements = {}
    for atom_id, (molecule, type_id, values, images, comment) in sorted(
        rows.items()
    ):
        if type_id not in types:
            raise ValueError(f"Undefined atom type {type_id}")
        name, mass = types[type_id]
        if comment and comment[-1] != name:
            raise ValueError(f"Atom {atom_id} comment disagrees with its type")
        component = " ".join(comment[:-1]) if len(comment) > 1 else None
        site = MovieSite(
            atom_id, molecule, type_id, name, component, values[0], images
        )
        if type_id not in elements:
            elements[type_id] = _element(name, mass, definitions, overrides)
        symbol = elements[type_id]
        if symbol is None:
            removed.append(site)
        else:
            symbols.append(symbol)
            positions.append(np.array(values[1:]) - origin)
            sites.append(site)
    if not sites:
        raise ValueError("Snapshot has no physical atoms")
    atoms = Atoms(symbols, positions=positions, cell=cell, pbc=True)
    waters = _check_water(atoms, sites, removed)
    return MovieSnapshot(
        atoms,
        sites,
        removed,
        count,
        origin.tolist(),
        hashlib.sha256(content).hexdigest(),
        waters,
    )


def validate_framework(atoms, source, replication, *, framework_mask):
    """Validate a rigid framework and return its reference atom indices.

    A periodic fractional-space search supplies candidates; acceptance uses
    Cartesian minimum-image distances and a one-to-one atom mapping.
    """
    if len(replication) != 3 or any(
        type(value) is not int or value < 1 for value in replication
    ):
        raise ValueError("Replication must contain three positive integers")
    mask = np.asarray(framework_mask, dtype=bool)
    if mask.shape != (len(atoms),):
        raise ValueError("Framework mask does not match atom count")
    reference = read(source).repeat(replication)
    framework = atoms[mask]
    if Counter(framework.numbers) != Counter(reference.numbers):
        raise ValueError(
            "Framework composition or supercell atom count changed"
        )
    if not np.allclose(atoms.cell, reference.cell, atol=1e-3, rtol=0):
        raise ValueError("Snapshot cell does not match the GCMC supercell")
    observed_frac = framework.get_scaled_positions(wrap=True) % 1
    reference_frac = reference.get_scaled_positions(wrap=True) % 1
    mapping = np.empty(len(framework), dtype=int)
    # A bound on the fractional norm of an accepted Cartesian displacement.
    rounding = np.sqrt(3) * np.linalg.norm(
        atoms.cell.array - reference.cell.array
    )
    radius = (0.002 + rounding) / np.linalg.svd(reference.cell)[1].min()
    for number in set(reference.numbers):
        actual = np.flatnonzero(framework.numbers == number)
        expected = np.flatnonzero(reference.numbers == number)
        tree = cKDTree(reference_frac[expected], boxsize=1)
        candidates = tree.query_ball_point(observed_frac[actual], radius)
        counts = np.array([len(group) for group in candidates])
        if (counts == 0).any():
            raise ValueError("Framework coordinates changed")
        owners = np.repeat(np.arange(len(actual)), counts)
        possible = expected[np.concatenate(candidates).astype(int)]
        _, distances = find_mic(
            framework.positions[actual[owners]] - reference.positions[possible],
            reference.cell,
            True,
        )
        valid = distances <= 0.002
        matches = np.bincount(owners[valid], minlength=len(actual))
        if (matches == 0).any():
            raise ValueError("Framework coordinates changed")
        if (matches != 1).any():
            raise ValueError("Framework atom mapping is not one-to-one")
        mapping[actual] = possible[valid]
    if len(set(mapping)) != len(reference):
        raise ValueError("Framework atom mapping is not one-to-one")
    return mapping.tolist()


def validate_molecule_definition(
    snapshot, component, path, *, pseudo_atoms=None, type_map=None
):
    """Check physical site counts against a rigid RASPA molecule definition.

    Virtual sites may already be omitted by the movie writer. Flexible
    definition layouts are rejected rather than interpreted as rigid sites.
    """
    lines = _data_lines(path)
    try:
        total, groups = int(lines[3][0]), int(lines[4][0])
        offset, names = 5, []
        for _ in range(groups):
            if lines[offset] != ["rigid"]:
                raise ValueError("Only rigid molecule definitions supported")
            size = int(lines[offset + 1][0])
            for row in lines[offset + 2 : offset + 2 + size]:
                if len(row) != 5:
                    raise ValueError("Invalid molecular site row")
                int(row[0])
                if not np.isfinite(list(map(float, row[2:]))).all():
                    raise ValueError("Non-finite molecular site")
                names.append(row[1])
            offset += 2 + size
        if len(names) != total or total < 1 or groups < 1:
            raise ValueError("Incomplete molecule definition")
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid molecule definition {path}: {exc}") from exc
    definitions = _pseudo_definitions(pseudo_atoms)
    overrides = _type_map(type_map)
    expected = Counter()
    for name in names:
        if name not in definitions:
            raise ValueError(f"Missing pseudo-atom definition: {name}")
        if _element(name, definitions[name][1], definitions, overrides):
            expected[name] += 1
    if not expected:
        raise ValueError("Molecule definition contains no physical atoms")
    molecules = defaultdict(Counter)
    for site in snapshot.sites + snapshot.removed_sites:
        if site.component == component:
            molecules[site.molecule_id][site.pseudo_atom] += 1
    for molecule, observed in molecules.items():
        if observed - Counter(names) or any(
            observed[name] != count for name, count in expected.items()
        ):
            raise ValueError(f"Incomplete {component} molecule {molecule}")
