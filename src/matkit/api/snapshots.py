"""Validated geometry export from completed gRASPA calculation bundles."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import write
from ase.io.cif import parse_cif
from pydantic import Field, JsonValue
from scipy.spatial import cKDTree

from matkit.graspa.snapshots import (
    _type_map,
    read_movie_snapshot,
    validate_framework,
    validate_molecule_definition,
)

from .bundles import (
    artifact,
    atomic_json,
    contained_path,
    inspect_run,
    staged_request,
)
from .models import (
    AdsorptionRequest,
    Artifact,
    Model,
    StructureData,
    StructureRef,
)
from .structures import sha256


class SnapshotFrameResult(Model):
    cycle: int
    source_snapshot: str
    status: Literal["ready", "exported", "exported_dry", "failed"]
    structure: StructureRef | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    details: dict[str, JsonValue] = Field(default_factory=dict)
    error: str | None = None


class SnapshotExportResult(Model):
    schema_name: Literal["matkit.snapshot_export"] = "matkit.snapshot_export"
    schema_version: Literal[1] = 1
    run_id: str
    source_bundle: str
    check_only: bool
    selection: str | list[int]
    provenance: dict[str, JsonValue]
    items: list[SnapshotFrameResult] = Field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return bool(self.items) and all(
            item.status != "failed" for item in self.items
        )


def _input_value(text, key):
    values = re.findall(rf"^{re.escape(key)}\s+([^#\n]+)", text, re.M)
    if len(values) != 1:
        raise ValueError(f"Expected one {key} in simulation.input")
    return values[0].split()


def _bundle_context(root):
    record = inspect_run(root)
    if record.operation != "adsorption":
        raise ValueError("Expected an adsorption bundle")
    if (root / ".matkit.lock").exists():
        raise ValueError("Cannot export an actively owned bundle")
    if record.state not in {"completed", "failed"}:
        raise ValueError(f"Run is {record.state}; engine completion required")
    if record.state == "failed" and record.failure.stage not in {
        "calculation",
        "collection",
    }:
        raise ValueError("Only completed-engine statistics failures can export")
    exit_info = json.loads((root / "exit.json").read_text())
    if type(exit_info.get("returncode")) is not int or (
        exit_info["returncode"] != 0
    ):
        raise ValueError("Engine did not report a zero exit code")
    log = (root / "work/raspa.log").read_text()
    if not all(
        marker in log
        for marker in (
            "PRODUCTION PHASE ENDS",
            "END OF SIMULATION!",
            "Work took",
        )
    ):
        raise ValueError("Missing engine completion markers")
    hashes = record.provenance.get("prepared_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Missing prepared input hash inventory")
    for name, expected in hashes.items():
        path = contained_path(root, name)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Staged input changed or is missing: {name}")
    request = staged_request(root)
    if not isinstance(request, AdsorptionRequest):
        raise ValueError("Expected a gRASPA adsorption request")
    source = Path(request.structure.path)
    simulation = contained_path(
        root, record.resolved.get("simulation_input", "work/simulation.input")
    )
    pseudo = root / "work/pseudo_atoms.def"
    molecule = root / "work" / f"{request.adsorbate}.def"
    for path in (source, simulation, pseudo, molecule, root / "request.json"):
        if str(path.relative_to(root)) not in hashes:
            raise ValueError(f"Input is not covered by prepared hashes: {path}")
    if request.structure.sha256 != sha256(source):
        raise ValueError("Staged framework does not match the request hash")
    text = simulation.read_text()
    if _input_value(text, "NumberOfSimulations") != ["1"] or _input_value(
        text, "SingleSimulation"
    ) != ["yes"]:
        raise ValueError("Bundle export supports one simulation system")
    if _input_value(text, "RestartFile") != ["no"]:
        raise ValueError("Restarted movie sequences are unsupported")
    components = re.findall(
        r"^Component\s+(\d+)\s+MoleculeName\s+(\S+)", text, re.M
    )
    if components != [("0", request.adsorbate)]:
        raise ValueError("Bundle export requires one matching adsorbate")
    cycles = int(_input_value(text, "NumberOfProductionCycles")[0])
    if cycles != request.production_cycles:
        raise ValueError("Production cycle count disagrees with request")
    replication = record.resolved.get("unit_cells")
    unit_cells = list(map(int, _input_value(text, "UnitCells")))
    if unit_cells != [0, *(replication or [])]:
        raise ValueError("UnitCells disagrees with resolved replication")
    framework = _input_value(text, "FrameworkName")[0]
    return record, request, source, pseudo, molecule, replication, framework


def _select_frames(root, frames, production_cycles):
    movie_dir = root / "work/Movies/System_0"
    if isinstance(frames, str):
        if frames == "final":
            return [production_cycles - 1]
        if frames != "all":
            raise ValueError("frames must be 'final', 'all', or cycle indices")
        indices = []
        for path in movie_dir.glob("result_*.data"):
            match = re.fullmatch(r"result_(0|[1-9][0-9]*)\.data", path.name)
            if match:
                indices.append(int(match[1]))
        if not indices:
            raise ValueError("No production snapshots available")
        return sorted(indices)
    try:
        indices = list(frames)
    except TypeError as exc:
        raise ValueError("frames must be a sequence of cycle indices") from exc
    if not indices or any(type(i) is not int or i < 0 for i in indices):
        raise ValueError("Frame indices must be nonnegative integers")
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate requested frame index")
    return sorted(indices)


def _cif_bytes(snapshot):
    stream = BytesIO()
    write(stream, snapshot.atoms, format="cif")
    encoded = stream.getvalue()
    # These files contain the full P1 cell. Read the explicit atom table
    # through ASE's CIF parser, avoiding its quadratic symmetry expansion
    # on thousands of atoms. Reject sites that the ordinary ASE reader
    # would merge at its default fractional-coordinate tolerance.
    blocks = list(parse_cif(BytesIO(encoded)))
    if len(blocks) != 1:
        raise ValueError("Expected one CIF data block")
    block = blocks[0]
    if block.get_spacegroup(subtrans_included=True).no != 1 or not np.allclose(
        block["_atom_site_occupancy"], 1
    ):
        raise ValueError("Expected fully occupied P1 output")
    fractional = (
        np.column_stack([block[f"_atom_site_fract_{axis}"] for axis in "xyz"])
        % 1
    )
    pairs = cKDTree(fractional, boxsize=1).query_pairs(0.001, p=np.inf)
    if pairs:
        pairs = np.array(sorted(pairs))
        diff = fractional[pairs[:, 0]] - fractional[pairs[:, 1]]
        diff -= np.rint(diff)
        if (np.max(np.abs(diff), axis=1) < 0.001).any():
            raise ValueError("CIF reader would merge overlapping atom sites")
    restored = Atoms(
        symbols=block["_atom_site_type_symbol"],
        scaled_positions=fractional,
        cell=block.get_cell(),
        pbc=True,
    )
    atoms = snapshot.atoms
    if not np.array_equal(restored.numbers, atoms.numbers) or not np.allclose(
        restored.cell, atoms.cell, atol=1e-7, rtol=0
    ):
        raise ValueError("CIF round trip changed atom order/count or cell")
    _, distances = find_mic(
        restored.positions - atoms.positions, atoms.cell, True
    )
    if max(distances, default=0) > 1e-7:
        raise ValueError("CIF round trip changed periodic coordinates")
    # CIF reading generates ASE bookkeeping arrays. The geometry-only
    # sidecar deliberately excludes those, as well as simulation charges.
    metadata = StructureData(
        atomic_numbers=restored.numbers.tolist(),
        positions=restored.positions.tolist(),
        cell=restored.cell.array.tolist(),
        pbc=restored.pbc.tolist(),
        atom_ids=[
            f"{snapshot.source_sha256}:{site.atom_id}"
            for site in snapshot.sites
        ],
        parent_sha256=snapshot.source_sha256,
    )
    return encoded, metadata


def _publish_frame(output, cycle, encoded, metadata):
    directory = output / "System_0"
    cif = directory / f"snapshot_{cycle}.cif"
    sidecar = cif.with_suffix(".metadata.json")
    with TemporaryDirectory(prefix=".snapshot-", dir=directory) as temp:
        stage = Path(temp)
        (stage / cif.name).write_bytes(encoded)
        atomic_json(stage / sidecar.name, metadata)
        try:
            (stage / sidecar.name).rename(sidecar)
            (stage / cif.name).rename(cif)
            artifacts = [
                artifact(output, cif, "structure"),
                artifact(output, sidecar, "structure_metadata"),
            ]
        except OSError:
            cif.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
            raise
    return StructureRef(path=str(cif), sha256=artifacts[0].sha256), artifacts


def export_adsorption_snapshots(
    bundle,
    *,
    output_dir,
    frames="final",
    type_map=None,
    check_only=False,
) -> SnapshotExportResult:
    """Export selected production frames without modifying the run bundle.

    ``final`` requires the exact final production cycle; ``all`` selects all
    saved frames numerically. Explicit cycle indices report missing frames
    individually. ``check_only`` performs the same validation, including a
    CIF round trip in memory, without creating the output directory.

    Fresh output directories are required. Each successful CIF has a clean
    StructureData sidecar; source site records live in export_manifest.json.
    Successful frames survive other frames failing. Inspect ``accepted`` and
    individual item errors; this result does not certify sampling quality.
    """
    root = Path(bundle).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.is_relative_to(root):
        raise ValueError("Export outside the immutable source run bundle")
    overrides = _type_map(type_map)
    record, request, source, pseudo, molecule, replication, framework = (
        _bundle_context(root)
    )
    indices = _select_frames(root, frames, request.production_cycles)
    report = SnapshotExportResult(
        run_id=record.run_id,
        source_bundle=str(root),
        check_only=check_only,
        selection=frames if isinstance(frames, str) else indices,
        provenance={
            "matkit_state": record.state,
            "matkit_failure": record.failure.model_dump()
            if record.failure
            else None,
            "temperature_K": request.temperature_K,
            "pressure_Pa": request.pressure_Pa,
            "unit_cells": replication,
            "source_framework_sha256": sha256(source),
            "prepared_hashes": record.provenance["prepared_hashes"],
            "type_map": overrides,
            "fixed_charges_exported": False,
            "sampling_quality": "not established by export",
        },
    )
    if not check_only:
        output.mkdir(parents=True, exist_ok=False)
        (output / "System_0").mkdir()
    recorded_hashes = {ref.path: ref.sha256 for ref in record.artifacts}
    for cycle in indices:
        relative = f"work/Movies/System_0/result_{cycle}.data"
        item = SnapshotFrameResult(
            cycle=cycle, source_snapshot=relative, status="failed"
        )
        try:
            if cycle >= request.production_cycles:
                raise ValueError("Frame index is outside the production run")
            path = contained_path(root, relative)
            if not path.is_file():
                raise ValueError(f"Missing production snapshot: {relative}")
            snapshot = read_movie_snapshot(
                path, pseudo_atoms=pseudo, type_map=overrides
            )
            if (
                relative in recorded_hashes
                and recorded_hashes[relative] != snapshot.source_sha256
            ):
                raise ValueError(
                    "Snapshot changed since run artifact inventory"
                )
            allowed = {framework, f"{framework}.cif"}
            all_sites = snapshot.sites + snapshot.removed_sites
            if any(
                s.component not in allowed | {request.adsorbate}
                for s in all_sites
            ):
                raise ValueError("Missing or unrecognized component labels")
            mask = [site.component in allowed for site in snapshot.sites]
            reference_indices = validate_framework(
                snapshot.atoms, source, replication, framework_mask=mask
            )
            validate_molecule_definition(
                snapshot,
                request.adsorbate,
                molecule,
                pseudo_atoms=pseudo,
                type_map=overrides,
            )
            encoded, metadata = _cif_bytes(snapshot)
            item.details = {
                **snapshot.provenance(),
                "framework_reference_indices": reference_indices,
                "framework_atoms": sum(mask),
                "adsorbate_molecules": snapshot.component_counts.get(
                    request.adsorbate, 0
                ),
                "checks": [
                    "atom_table",
                    "physical_species",
                    "framework",
                    "molecule_completeness",
                    "cif_roundtrip",
                ]
                + (["tip4p_geometry"] if snapshot.water_molecules else []),
            }
            if check_only:
                item.status = "ready"
            else:
                item.structure, item.artifacts = _publish_frame(
                    output, cycle, encoded, metadata
                )
                item.status = (
                    "exported"
                    if item.details["adsorbate_molecules"]
                    else "exported_dry"
                )
        except (OSError, ValueError, KeyError) as exc:
            item.error = str(exc)
        report.items.append(item)
        if not check_only:
            atomic_json(output / "export_manifest.json", report)
    return report
