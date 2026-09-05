"""Scientific request validation, structure metadata and compatibility."""

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import write

from matkit.api import (
    AlchemiAdapter,
    EvaluateRequest,
    MLIPMethod,
    PoreRequest,
    RelaxRequest,
    RootstockAdapter,
    RunResult,
    StructureRef,
    list_capabilities,
    parse_request,
    prepare,
)
from matkit.api.models import REQUEST_ADAPTER
from matkit.api.structures import final_structure, load_structure, to_atoms


def test_request_round_trip_and_schema():
    for adapter in (RootstockAdapter(), AlchemiAdapter()):
        request = EvaluateRequest(
            structure=StructureRef(path="input.cif"),
            method=MLIPMethod(checkpoint="medium"),
            adapter=adapter,
        )
        assert parse_request(json.loads(request.model_dump_json())) == request
    schema = REQUEST_ADAPTER.json_schema()
    assert set(schema["discriminator"]["mapping"]) == {
        "evaluate",
        "relax",
        "pores",
        "adsorption",
    }


@pytest.mark.parametrize(
    "changes",
    [{"fmax": float("nan")}, {"steps": True}, {"steps": 0}, {"cell_opt": True}],
)
def test_invalid_relaxation_settings(changes):
    with pytest.raises(ValueError):
        RelaxRequest(
            structure=StructureRef(path="input.cif"),
            method=MLIPMethod(checkpoint="medium"),
            **changes,
        )


def test_unknown_result_schema_is_rejected():
    with pytest.raises(ValueError):
        RunResult.model_validate(
            {"schema_version": 1, "success": True, "energy": 0}
        )


def test_nonfinite_nested_configuration():
    with pytest.raises(ValueError):
        RootstockAdapter(setup_kwargs={"nested": {"bad": float("inf")}})


@pytest.mark.parametrize("value", [True, "1.0", -1, float("nan")])
def test_scientific_numbers_require_finite_numeric_values(value):
    with pytest.raises(ValueError):
        PoreRequest(
            structure=StructureRef(path="input.cif"), probe_radius=value
        )


def test_discovery_does_not_import_optional_engines(monkeypatch):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch",
            "mace",
            "rootstock",
            "nvalchemi",
            "mcp",
        }:
            raise AssertionError(f"Discovery imported {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert len(list_capabilities()) == 5
    assert all(c["status"] == "experimental" for c in list_capabilities())


def test_structure_handoff_and_invalidation(tmp_path):
    path = tmp_path / "input.extxyz"
    atoms = Atoms(
        "Cu2", positions=[[0, 0, 0], [3, 0, 0]], cell=[10] * 3, pbc=True
    )
    atoms.set_initial_charges([0.2, -0.2])
    atoms.set_constraint(FixAtoms(indices=[0]))
    write(path, atoms)
    original, data, digest = load_structure(StructureRef(path=str(path)))
    data.labels = ["Cu1", "Cu2"]
    data.bonds = [(data.atom_ids[0], data.atom_ids[1])]
    recovered = to_atoms(data)
    np.testing.assert_allclose(recovered.get_initial_charges(), [0.2, -0.2])
    assert len(recovered.constraints) == 1
    original.positions[1, 0] -= 0.1
    output = tmp_path / "final.extxyz"
    result = final_structure(original, data, digest, output)
    assert result.atom_ids == data.atom_ids
    assert result.labels == data.labels
    assert result.bonds == data.bonds
    assert result.parent_sha256 == digest
    assert "initial_charges" not in result.arrays
    assert not result.derived_from
    _, restored, _ = load_structure(StructureRef(path=str(output)))
    assert restored == result
    with pytest.raises(ValueError, match="metadata does not match"):
        load_structure(StructureRef(path=str(output), metadata=data))


def test_wrong_structure_digest(sample_cif):
    with pytest.raises(ValueError, match="hash"):
        load_structure(StructureRef(path=sample_cif, sha256="0" * 64))


def test_stale_charges_rejected(tmp_path):
    path = tmp_path / "input.extxyz"
    write(path, Atoms("Cu", cell=[10] * 3))
    _, data, _ = load_structure(StructureRef(path=str(path)))
    data.derived_from = {"charges": "0" * 64}
    with pytest.raises(ValueError, match="invalidated"):
        load_structure(StructureRef(path=str(path), metadata=data))


def test_preparation_is_engine_independent(sample_cif, tmp_path):
    record = prepare(
        PoreRequest(structure=StructureRef(path=sample_cif)),
        output_dir=tmp_path / "run",
    )
    assert record.state == "prepared"
    assert any(a.path == "inputs/radii.rad" for a in record.artifacts)
    assert all(not Path(a.path).is_absolute() for a in record.artifacts)
