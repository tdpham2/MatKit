"""Relocatable calculation bundles and atomic run records."""

from __future__ import annotations

from contextlib import contextmanager
from importlib import metadata, resources
import json
import mimetypes
import os
from pathlib import Path
import platform
import re
import shutil
import tempfile
from uuid import uuid4

import numpy as np
from ase.io import write
from ase.io.cif import parse_cif

from .models import (
    AdsorptionRequest,
    Artifact,
    CalculatorRequest,
    Failure,
    PoreRequest,
    RunResult,
    StructureRef,
    parse_request,
)
from .structures import load_structure, sha256


def atomic_json(path: Path, value) -> None:
    data = (
        value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    )
    encoded = json.dumps(data, allow_nan=False, indent=2)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def claim(root: Path):
    lock = root / ".matkit.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def contained_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact paths must stay within the run bundle")
    return path


def artifact(root: Path, path: Path, role: str) -> Artifact:
    relative = str(path.relative_to(root))
    contained_path(root, relative)
    return Artifact(
        path=relative,
        sha256=sha256(path),
        size_bytes=path.stat().st_size,
        media_type=mimetypes.guess_type(path.name)[0]
        or "application/octet-stream",
        role=role,
    )


def collect_artifacts(root: Path) -> list[Artifact]:
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink artifacts are unsupported: {path.name}")
        if (
            path.is_file()
            and not path.name.startswith(".")
            and path.name not in {"run.json", "result.json"}
        ):
            relative = path.relative_to(root)
            role = (
                "input"
                if relative.parts[0] == "inputs"
                or relative.name == "request.json"
                else "output"
            )
            result.append(artifact(root, path, role))
    return result


def verify_inputs(root: Path, record: RunResult) -> None:
    # Verify the full prepared inventory, including generated work files.
    for ref in record.artifacts:
        path = contained_path(root, ref.path)
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256(path) != ref.sha256
        ):
            raise ValueError(
                f"Staged artifact changed or is missing: {ref.path}"
            )


def inspect_run(path: str | Path) -> RunResult:
    root = Path(path).expanduser().resolve()
    # The committed result is authoritative after a manifest-update failure.
    result_path = root / "result.json"
    if not result_path.exists():
        result_path = root / "run.json"
    result = RunResult.model_validate_json(result_path.read_text())
    if result_path.name == "result.json" and (root / "run.json").exists():
        try:
            manifest = RunResult.model_validate_json(
                (root / "run.json").read_text()
            )
        except (OSError, ValueError):
            return result
        if manifest.run_id == result.run_id and manifest.state in {
            "failed",
            "interrupted",
        }:
            return manifest
    return result


def commit_result(root: Path, result: RunResult) -> RunResult:
    # Validate updates as model_copy deliberately skips validation.
    result = RunResult.model_validate(result.model_dump(mode="json"))
    atomic_json(root / "result.json", result)
    atomic_json(root / "run.json", result)
    return result


def refresh_artifacts(root: Path, record: RunResult) -> RunResult:
    """Update closed worker logs without replacing a committed outcome.

    The manifest can report interruption after the scientific result was
    committed. Keep those states distinct while refreshing both inventories.
    """
    artifacts = collect_artifacts(root)
    record = RunResult.model_validate(
        record.model_copy(update={"artifacts": artifacts}).model_dump()
    )
    result_path = root / "result.json"
    if result_path.exists():
        committed = RunResult.model_validate_json(result_path.read_text())
        if committed.run_id != record.run_id:
            raise ValueError("Committed result does not match run manifest")
        atomic_json(
            result_path, committed.model_copy(update={"artifacts": artifacts})
        )
    atomic_json(root / "run.json", record)
    return record


def environment_versions() -> dict:
    versions = {}
    for package in (
        "matkit",
        "ase",
        "numpy",
        "pydantic",
        "mace-torch",
        "rootstock",
        "nvalchemi-toolkit",
        "torch",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "runtime_settings": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
                "MKL_NUM_THREADS",
            )
        },
    }


def _copy_tree(source, destination: Path):
    destination.mkdir()
    for entry in source.iterdir():
        if isinstance(entry, Path) and entry.is_symlink():
            raise ValueError("Template symlinks are unsupported")
        target = destination / entry.name
        if entry.is_dir():
            _copy_tree(entry, target)
        else:
            target.write_bytes(entry.read_bytes())


def _set_input(path: Path, key: str, value) -> None:
    content, count = re.subn(
        rf"^{re.escape(key)}\s+.*$",
        f"{key} {value}",
        path.read_text(),
        flags=re.M,
    )
    if count != 1:
        raise ValueError(f"Template must define {key} exactly once")
    path.write_text(content)


def _prepare_adsorption(root, request, atoms):
    from matkit.graspa import setup_simulation
    from matkit.utils import calculate_cell_size

    source = root / request.structure.path
    if (
        source.suffix.lower() != ".cif"
        or not atoms.pbc.all()
        or atoms.cell.rank != 3
    ):
        raise ValueError("gRASPA requires a fully periodic CIF")
    if atoms.constraints:
        raise ValueError("gRASPA constraint conversion is unsupported")
    tags = dict(list(parse_cif(str(source)))[0])
    charges = np.asarray(tags.get("_atom_site_charge", []), dtype=float)
    if charges.shape != (len(atoms),) or not np.isfinite(charges).all():
        raise ValueError(
            "gRASPA requires finite, atom-mapped _atom_site_charge "
            "values in the CIF"
        )
    if not np.isclose(charges.sum(), request.net_charge, atol=1e-4, rtol=0):
        raise ValueError("CIF charges do not sum to requested net_charge")
    if request.number_of_blocks > request.production_cycles:
        raise ValueError("number_of_blocks exceeds production_cycles")
    template = root / request.template_dir
    for name in (
        "simulation.input",
        "pseudo_atoms.def",
        "force_field.def",
        "force_field_mixing_rules.def",
        f"{request.adsorbate}.def",
    ):
        if not (template / name).is_file():
            raise ValueError(f"Missing simulation definition: {name}")
    sizes = calculate_cell_size(atoms, request.cutoff_angstrom)
    setup_simulation(
        str(source),
        str(root / "work"),
        [
            {
                "MoleculeName": request.adsorbate,
                "FugacityCoefficient": request.fugacity_coefficient,
            }
        ],
        temperature=request.temperature_K,
        pressure=request.pressure_Pa,
        cutoff=request.cutoff_angstrom,
        n_cycle=request.production_cycles,
        template_dir=str(template),
        cell_size=sizes,
    )
    input_file = root / "work" / "simulation.input"
    for key, value in {
        "NumberOfInitializationCycles": request.initialization_cycles,
        "NumberOfEquilibrationCycles": request.equilibration_cycles,
        "NumberOfProductionCycles": request.production_cycles,
        "NumberOfBlocks": request.number_of_blocks,
    }.items():
        _set_input(input_file, key, value)
    text = input_file.read_text()
    if len(re.findall(r"^Component\s+\d+\s+MoleculeName", text, re.M)) != 1:
        raise ValueError(
            "Unified gRASPA execution supports exactly one component"
        )
    for key, expected in {
        "UseChargesFromCIFFile": "yes",
        "NumberOfSimulations": "1",
        "SingleSimulation": "yes",
        "RestartFile": "no",
    }.items():
        match = re.search(rf"^{key}\s+(\S+)", text, re.M)
        if not match or match[1].lower() != expected:
            raise ValueError(
                f"Unified gRASPA execution requires {key} {expected}"
            )
    return {
        "unit_cells": sizes,
        "simulation_input": "work/simulation.input",
        "random_seed": None,
        "uncertainty_method": "unknown; engine-reported",
    }


def prepare(request, *, output_dir: str | Path) -> RunResult:
    request = parse_request(request)
    atoms, structure, digest = load_structure(request.structure)
    if (
        isinstance(request, CalculatorRequest)
        and request.adapter.type == "nvalchemi-mace"
    ):
        if atoms.constraints or structure.arrays or structure.bonds:
            raise ValueError(
                "Native ALCHEMI does not support this structure's "
                "arrays, bonds or constraints"
            )
    if isinstance(request, PoreRequest) and (
        not atoms.pbc.all() or atoms.cell.rank != 3
    ):
        raise ValueError("Zeo++ requires a fully periodic cell")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with claim(root):
        if any(p.name != ".matkit.lock" for p in root.iterdir()):
            raise FileExistsError(
                "Use a fresh output directory; resume is not supported"
            )
        record = RunResult(
            run_id=uuid4().hex,
            operation=request.operation,
            state="prepared",
            requested=request.model_dump(mode="json"),
            provenance={
                "input_sha256": digest,
                "model_sha256": None,
                "preparation_environment": environment_versions(),
            },
        )
        atomic_json(root / "run.json", record)
        try:
            (root / "inputs").mkdir()
            (root / "work").mkdir()
            source = Path(request.structure.path).expanduser().resolve()
            staged = root / "inputs" / f"structure{source.suffix.lower()}"
            shutil.copyfile(source, staged)
            atomic_json(staged.with_suffix(".metadata.json"), structure)
            values = request.model_dump(mode="json")
            values["structure"] = StructureRef(
                path=str(staged.relative_to(root)), sha256=digest
            ).model_dump(mode="json")
            if isinstance(request, CalculatorRequest):
                checkpoint = Path(request.method.checkpoint).expanduser()
                if checkpoint.is_file():
                    target = root / "inputs" / f"model{checkpoint.suffix}"
                    shutil.copyfile(checkpoint, target)
                    values["method"]["checkpoint"] = str(
                        target.relative_to(root)
                    )
                    record.provenance["model_sha256"] = sha256(target)
                elif checkpoint.is_absolute():
                    raise FileNotFoundError(
                        f"Checkpoint file does not exist: {checkpoint}"
                    )
                if (
                    request.adapter.type == "rootstock"
                    and request.adapter.weights
                ):
                    weights = Path(request.adapter.weights).expanduser()
                    target = root / "inputs" / f"weights{weights.suffix}"
                    shutil.copyfile(weights, target)
                    values["adapter"]["weights"] = str(target.relative_to(root))
                    record.provenance["model_sha256"] = sha256(target)
            if isinstance(request, PoreRequest):
                radii = root / "inputs" / "radii.rad"
                if request.radii_file:
                    shutil.copyfile(
                        Path(request.radii_file).expanduser(), radii
                    )
                else:
                    radii.write_bytes(
                        resources.files("matkit.zeopp")
                        .joinpath("files/UFF.rad")
                        .read_bytes()
                    )
                values["radii_file"] = str(radii.relative_to(root))
                if source.suffix.lower() == ".cif":
                    shutil.copyfile(source, root / "work" / "structure.cif")
                else:
                    write(root / "work" / "structure.cif", atoms)
            if isinstance(request, AdsorptionRequest):
                template = (
                    Path(request.template_dir).expanduser()
                    if request.template_dir
                    else resources.files("matkit.graspa").joinpath(
                        "files/template"
                    )
                )
                _copy_tree(template, root / "inputs" / "template")
                values["template_dir"] = "inputs/template"
                staged_request = parse_request(values)
                record.resolved.update(
                    _prepare_adsorption(root, staged_request, atoms)
                )
            atomic_json(root / "request.json", values)
            record.resolved["request"] = values
            record.artifacts = collect_artifacts(root)
            record.provenance["prepared_hashes"] = {
                a.path: a.sha256 for a in record.artifacts
            }
            atomic_json(root / "run.json", record)
            return record
        except Exception as exc:
            failed = record.model_copy(
                update={
                    "state": "failed",
                    "failure": Failure(
                        code=type(exc).__name__,
                        stage="preparation",
                        message=str(exc),
                    ),
                }
            )
            atomic_json(root / "run.json", failed)
            raise


def staged_request(root: Path):
    request = parse_request(json.loads((root / "request.json").read_text()))
    values = request.model_dump(mode="json")
    values["structure"]["path"] = str(
        contained_path(root, request.structure.path)
    )
    if isinstance(request, CalculatorRequest):
        checkpoint = request.method.checkpoint
        if checkpoint.startswith("inputs/"):
            values["method"]["checkpoint"] = str(
                contained_path(root, checkpoint)
            )
        if request.adapter.type == "rootstock" and request.adapter.weights:
            values["adapter"]["weights"] = str(
                contained_path(root, request.adapter.weights)
            )
    if isinstance(request, PoreRequest):
        values["radii_file"] = str(contained_path(root, request.radii_file))
    if isinstance(request, AdsorptionRequest):
        values["template_dir"] = str(contained_path(root, request.template_dir))
    return parse_request(values)
