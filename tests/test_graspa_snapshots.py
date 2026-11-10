"""Portable movie fixtures and public CIF/MLIP handoff regression tests."""

from collections import Counter
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write
from click.testing import CliRunner

from matkit.api import (
    AdsorptionRequest,
    AlchemiAdapter,
    EvaluateRequest,
    MLIPMethod,
    SnapshotExportResult,
    StructureRef,
    export_adsorption_snapshots,
    prepare,
)
from matkit.api.models import AdsorptionPayload, RunResult
from matkit.api.structures import load_structure, sha256
from matkit.cli import main
from matkit.graspa import read_movie_snapshot
from matkit.graspa.snapshots import (
    validate_framework,
    validate_molecule_definition,
)


CELL = np.array([[10.0, 0.0, 0.0], [1.2, 9.0, 0.0], [-0.5, 0.8, 8.0]])
TYPES = {1: ("C", 12.011), 2: ("Ow", 15.9994), 3: ("Hw", 1.008), 4: ("Lw", 0.0)}
WATER_DEF = """647.096
22064000
0.344
4
1
rigid
4
0 Ow 0 0 0
1 Lw 0 .15 0
2 Hw .75695 .58588 0
3 Hw -.75695 .58588 0
"""


def movie(
    path,
    *,
    dummy=True,
    water=True,
    types=None,
    rows=None,
    origin=(0.0, 0.0, 0.0),
    cell=CELL,
    images=False,
):
    types = TYPES if types is None else types
    if rows is None:
        rows = [(1, 1, 1, 0.2, [2.0, 3.0, 4.0], "framework.cif")]
        if water:
            rows += [
                (2, 2, 2, 0.0, [9.9, 4.0, 4.0], "TIP4P"),
                (3, 2, 3, 0.52, [0.65695, 4.58588, 4.0], "TIP4P"),
                (4, 2, 3, 0.52, [9.14305, 4.58588, 4.0], "TIP4P"),
            ]
            if dummy:
                rows.append((5, 2, 4, -1.04, [9.9, 4.15, 4.0], "TIP4P"))
    header = [
        "# portable gRASPA movie",
        f"{len(rows)} atoms",
        "0 bonds",
        f"{len(types)} atom types",
    ]
    for index, axis in enumerate("xyz"):
        lo = origin[index]
        header.append(f"{lo} {lo + cell[index, index]} {axis}lo {axis}hi")
    header += [
        f"{cell[1, 0]} {cell[2, 0]} {cell[2, 1]} xy xz yz",
        "",
        "Masses",
        "",
    ]
    header += [f"{i} {mass} # {name}" for i, (name, mass) in types.items()]
    header += ["", "Pair Coeffs", "", "1 0 0", "", "Atoms # full", ""]
    for atom_id, mol, kind, charge, position, component in rows:
        xyz = " ".join(str(v) for v in np.array(position) + origin)
        flags = " 1 -1 0" if images else ""
        comment = f"{component} " if component else ""
        header.append(
            f"{atom_id} {mol} {kind} {charge} {xyz}{flags} "
            f"# {comment}{types[kind][0]}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header) + "\n")
    return path


def pseudo(path, types=None, overrides=None):
    definitions = {
        "C": ("C", "yes"),
        "Ow": ("O", "yes"),
        "Hw": ("H", "yes"),
        "Lw": ("-", "no"),
    }
    definitions.update(overrides or {})
    types = TYPES if types is None else types
    lines = [str(len(types))]
    for name, mass in types.values():
        symbol, printable = definitions.get(name, (name, "yes"))
        lines.append(f"{name} {printable} {symbol} {symbol} 0 {mass} 0")
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def bundle_factory(tmp_path):
    counter = 0

    def create(*, indices=(0, 9), water=True, dummy=False, cycles=10):
        nonlocal counter
        counter += 1
        root = tmp_path / f"bundle{counter}"
        (root / "inputs").mkdir(parents=True)
        (root / "work").mkdir()
        source = root / "inputs/structure.cif"
        write(source, Atoms("C", positions=[[2, 3, 4]], cell=CELL, pbc=True))
        request = AdsorptionRequest(
            structure=StructureRef(
                path="inputs/structure.cif", sha256=sha256(source)
            ),
            adsorbate="TIP4P",
            temperature_K=298.0,
            pressure_Pa=3140.0,
            production_cycles=cycles,
            template_dir="inputs/template",
        )
        (root / "request.json").write_text(request.model_dump_json())
        (root / "work/simulation.input").write_text(
            "NumberOfSimulations 1\nSingleSimulation yes\nRestartFile no\n"
            f"NumberOfProductionCycles {cycles}\nUnitCells 0 1 1 1\n"
            "FrameworkName framework\nComponent 0 MoleculeName TIP4P\n"
        )
        pseudo(root / "work/pseudo_atoms.def")
        (root / "work/TIP4P.def").write_text(WATER_DEF)
        hashes = {
            str(p.relative_to(root)): sha256(p)
            for p in root.rglob("*")
            if p.is_file()
        }
        for index in indices:
            movie(
                root / f"work/Movies/System_0/result_{index}.data",
                water=water,
                dummy=dummy,
            )
        (root / "work/raspa.log").write_text(
            "PRODUCTION PHASE ENDS\nWork took 1 seconds\nEND OF SIMULATION!"
        )
        (root / "exit.json").write_text('{"returncode": 0}')
        record = RunResult(
            run_id=f"test-{counter}",
            operation="adsorption",
            state="completed",
            numerical_validity="valid",
            requested=request.model_dump(),
            resolved={"unit_cells": [1, 1, 1]},
            provenance={"prepared_hashes": hashes},
            payload=AdsorptionPayload(
                component="TIP4P",
                uptake=1.0,
                unit="mol/kg",
                uncertainty=0.0,
                uncertainty_method="fixture",
                heat_of_adsorption=-40.0,
                heat_uncertainty=0.0,
            ),
        )
        (root / "run.json").write_text(record.model_dump_json())
        return root

    return create


@pytest.mark.parametrize("dummy", [True, False])
def test_tip4p_dummy_present_or_omitted(tmp_path, dummy):
    path = movie(tmp_path / "water.data", dummy=dummy)
    snapshot = read_movie_snapshot(path)
    assert snapshot.atoms.get_chemical_formula() == "CH2O"
    assert snapshot.water_molecules == 1
    assert len(snapshot.removed_sites) == int(dummy)
    assert set(snapshot.atoms.arrays) == {"numbers", "positions"}
    assert not snapshot.atoms.constraints
    np.testing.assert_allclose(snapshot.atoms.cell, CELL)
    assert snapshot.atoms.positions[2, 0] == pytest.approx(10.65695)
    assert snapshot.component_counts == {"TIP4P": 1, "framework.cif": 1}
    assert snapshot.source_sha256 == sha256(path)


def test_origin_image_flags_and_noncontiguous_ids(tmp_path):
    path = movie(tmp_path / "origin.data", origin=(2, -3, 4), images=True)
    text = path.read_text().replace("5 2 4 -1.04", "50 2 4 -1.04")
    path.write_text(text)
    snapshot = read_movie_snapshot(path)
    np.testing.assert_allclose(snapshot.atoms.positions[0], [2, 3, 4])
    assert snapshot.origin == [2, -3, 4]
    assert snapshot.sites[0].image_flags == (1, -1, 0)
    assert snapshot.removed_sites[0].atom_id == 50


@pytest.mark.parametrize(
    "change,match",
    [
        (("5 atoms", "6 atoms"), "Incomplete snapshot"),
        (("4 atom types", "5 atom types"), "Incomplete atom type"),
        (("5 2 4 -1.04", "4 2 4 -1.04"), "duplicate atom ID"),
        (("5 2 4 -1.04", "5 2 99 -1.04"), "Undefined atom type"),
        (("2.0 3.0 4.0", "nan 3.0 4.0"), "Non-finite"),
        (("1 1 1 0.2", "1 1 1 nan"), "Non-finite"),
        (("Atoms # full", "Atoms # atomic"), "full-style"),
        (("0.0 10.0 xlo xhi", "0.0 0.0 xlo xhi"), "Invalid cell"),
        (("# TIP4P Lw", "# TIP4P Ow"), "comment disagrees"),
        (("5 2 4 -1.04", "5 1 4 -1.04"), "components share"),
    ],
)
def test_invalid_movie_rejected(tmp_path, change, match):
    path = movie(tmp_path / "bad.data")
    assert change[0] in path.read_text()
    path.write_text(path.read_text().replace(*change))
    with pytest.raises(ValueError, match=match):
        read_movie_snapshot(path)


def test_partial_water_and_bad_geometry(tmp_path):
    path = movie(tmp_path / "partial.data")
    lines = path.read_text().splitlines()
    path.write_text(
        "\n".join(
            line.replace("5 atoms", "4 atoms")
            for line in lines
            if not line.startswith("4 2 3")
        )
    )
    with pytest.raises(ValueError, match="Incomplete TIP4P"):
        read_movie_snapshot(path)
    path = movie(tmp_path / "geometry.data")
    path.write_text(path.read_text().replace("0.65695", "1.65695"))
    with pytest.raises(ValueError, match="O-H distances"):
        read_movie_snapshot(path)
    path = movie(tmp_path / "angle.data")
    path.write_text(path.read_text().replace("9.14305", "0.65695"))
    with pytest.raises(ValueError, match="H-O-H angle"):
        read_movie_snapshot(path)


def test_generic_mapping_and_print_flag(tmp_path):
    types = {1: ("oxygen_custom", 15.9994)}
    path = movie(
        tmp_path / "custom.data",
        types=types,
        rows=[
            (1, 1, 1, -0.5, [2, 3, 4], "guest"),
        ],
    )
    with pytest.raises(ValueError, match="Unsupported physical species"):
        read_movie_snapshot(path)
    snapshot = read_movie_snapshot(path, type_map={"oxygen_custom": "O"})
    assert snapshot.atoms.get_chemical_symbols() == ["O"]
    definitions = pseudo(
        tmp_path / "pseudo_atoms.def",
        types,
        overrides={"oxygen_custom": ("O", "no")},
    )
    assert len(read_movie_snapshot(path, pseudo_atoms=definitions).atoms) == 1
    with pytest.raises(ValueError, match="Cannot discard"):
        read_movie_snapshot(path, type_map={"oxygen_custom": None})


def test_united_atom_mapping_cannot_fabricate_missing_hydrogen(tmp_path):
    types = {1: ("CH4_sp3", 16.04246)}
    path = movie(
        tmp_path / "methane.data",
        types=types,
        rows=[
            (1, 1, 1, 0, [2, 3, 4], "methane"),
        ],
    )
    with pytest.raises(ValueError, match="united-atom"):
        read_movie_snapshot(path, type_map={"CH4_sp3": "C"})
    path.write_text(path.read_text().replace("CH4_sp3", "custom_bead"))
    with pytest.raises(ValueError, match="coarse-grained"):
        read_movie_snapshot(path, type_map={"custom_bead": "C"})


def test_mixture_and_same_element_in_framework_and_guest(tmp_path):
    types = {
        1: ("C", 12.011),
        2: ("O_co2", 15.9994),
        3: ("N_n2", 14.0067),
        4: ("N_com", 0.0),
    }
    rows = [
        (1, 1, 1, 0, [2, 3, 4], "framework.cif"),
        (2, 2, 1, 0.7, [4, 4, 4], "CO2"),
        (3, 2, 2, -0.35, [4, 4, 5.16], "CO2"),
        (4, 2, 2, -0.35, [4, 4, 2.84], "CO2"),
        (5, 3, 3, -0.482, [6, 6, 6.55], "N2"),
        (6, 3, 3, -0.482, [6, 6, 5.45], "N2"),
        (7, 3, 4, 0.964, [6, 6, 6], "N2"),
    ]
    path = movie(tmp_path / "mixture.data", types=types, rows=rows)
    snapshot = read_movie_snapshot(path)
    assert Counter(snapshot.atoms.symbols) == {"C": 2, "O": 2, "N": 2}
    assert snapshot.component_counts == {"CO2": 1, "N2": 1, "framework.cif": 1}
    source = tmp_path / "source.cif"
    write(source, Atoms("C", positions=[[2, 3, 4]], cell=CELL, pbc=True))
    mask = [s.component == "framework.cif" for s in snapshot.sites]
    assert validate_framework(
        snapshot.atoms, source, [1, 1, 1], framework_mask=mask
    ) == [0]


def test_molecule_definition_and_missing_nonprinting_physical_site(tmp_path):
    path = movie(tmp_path / "water.data", dummy=False)
    snapshot = read_movie_snapshot(path)
    defs = pseudo(tmp_path / "pseudo_atoms.def")
    molecule = tmp_path / "TIP4P.def"
    molecule.write_text(WATER_DEF)
    validate_molecule_definition(snapshot, "TIP4P", molecule, pseudo_atoms=defs)
    molecule.write_text(WATER_DEF.replace("1 Lw", "1 Hw"))
    with pytest.raises(ValueError, match="Incomplete TIP4P"):
        validate_molecule_definition(
            snapshot, "TIP4P", molecule, pseudo_atoms=defs
        )
    molecule.write_text(WATER_DEF.replace("rigid", "flexible"))
    with pytest.raises(ValueError, match="Only rigid"):
        validate_molecule_definition(
            snapshot, "TIP4P", molecule, pseudo_atoms=defs
        )


def test_virtual_only_guest_is_not_silently_discarded(tmp_path):
    types = {1: ("C", 12.011), 2: ("N_n2", 14.0067), 3: ("N_com", 0.0)}
    path = movie(
        tmp_path / "incomplete.data",
        types=types,
        rows=[
            (1, 1, 1, 0, [2, 3, 4], "framework.cif"),
            (2, 2, 3, 0.964, [5, 5, 5], "N2"),
        ],
    )
    defs = pseudo(
        tmp_path / "pseudo_atoms.def",
        types,
        overrides={
            "N_n2": ("N", "yes"),
            "N_com": ("-", "no"),
        },
    )
    snapshot = read_movie_snapshot(path, pseudo_atoms=defs)
    molecule = tmp_path / "N2.def"
    molecule.write_text(
        "126.192\n3395800\n.0372\n3\n1\nrigid\n3\n"
        "0 N_n2 0 0 .55\n1 N_com 0 0 0\n2 N_n2 0 0 -.55\n"
    )
    with pytest.raises(ValueError, match="Incomplete N2"):
        validate_molecule_definition(
            snapshot, "N2", molecule, pseudo_atoms=defs
        )


def test_moved_framework_rejected(tmp_path):
    snapshot = read_movie_snapshot(movie(tmp_path / "movie.data"))
    source = tmp_path / "source.cif"
    write(source, Atoms("C", positions=[[2.1, 3, 4]], cell=CELL, pbc=True))
    with pytest.raises(ValueError, match="Framework coordinates changed"):
        validate_framework(
            snapshot.atoms,
            source,
            [1, 1, 1],
            framework_mask=[True, False, False, False],
        )


def test_export_cif_roundtrip_and_native_mlip_handoff(bundle_factory, tmp_path):
    root = bundle_factory()
    before = {p: sha256(p) for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "export"
    report = export_adsorption_snapshots(root, output_dir=output)
    assert report.accepted
    (frame,) = report.items
    assert frame.cycle == 9 and frame.status == "exported"
    assert frame.details["adsorbate_molecules"] == 1
    assert len(frame.artifacts) == 2
    restored, data, digest = load_structure(frame.structure)
    original = read_movie_snapshot(root / frame.source_snapshot).atoms
    assert restored.get_chemical_symbols() == original.get_chemical_symbols()
    np.testing.assert_allclose(restored.cell, CELL, atol=1e-8)
    assert set(restored.arrays) == {"numbers", "positions"}
    assert data.arrays == {} and data.constraints == [] and data.bonds == []
    assert digest == frame.structure.sha256
    for i in range(len(original)):
        for j in range(i):
            assert restored.get_distance(i, j, mic=True) == pytest.approx(
                original.get_distance(i, j, mic=True), abs=1e-7
            )
    # Exercise ALCHEMI's array restrictions without loading a model.
    prepared = prepare(
        EvaluateRequest(
            structure=frame.structure,
            method=MLIPMethod(checkpoint="medium"),
            adapter=AlchemiAdapter(),
        ),
        output_dir=tmp_path / "mlip",
    )
    assert prepared.state == "prepared"
    reread = SnapshotExportResult.model_validate_json(
        (output / "export_manifest.json").read_text()
    )
    assert reread == report
    assert before == {p: sha256(p) for p in root.rglob("*") if p.is_file()}


def test_cif_is_standalone_p1_geometry(bundle_factory, tmp_path):
    report = export_adsorption_snapshots(
        bundle_factory(), output_dir=tmp_path / "export"
    )
    path = Path(report.items[0].structure.path)
    text = path.read_text()
    assert "_atom_site_charge" not in text
    assert "P 1" in text
    assert (
        read(BytesIO(path.read_bytes()), format="cif").get_chemical_formula()
        == "CH2O"
    )


def test_dry_frame_explicit(bundle_factory, tmp_path):
    report = export_adsorption_snapshots(
        bundle_factory(water=False), output_dir=tmp_path / "dry"
    )
    assert report.accepted
    assert report.items[0].status == "exported_dry"
    assert report.items[0].details["adsorbate_molecules"] == 0


def test_check_only_and_numeric_selection(bundle_factory, tmp_path):
    root = bundle_factory(indices=(10, 2, 0), cycles=11)
    output = tmp_path / "check"
    for selection, expected in [("all", [0, 2, 10]), ([10, 2], [2, 10])]:
        report = export_adsorption_snapshots(
            root, output_dir=output, frames=selection, check_only=True
        )
        assert report.accepted
        assert [item.cycle for item in report.items] == expected
        assert all(item.status == "ready" for item in report.items)
        assert not output.exists()


def test_missing_final_never_substituted(bundle_factory, tmp_path):
    report = export_adsorption_snapshots(
        bundle_factory(indices=(0, 8)), output_dir=tmp_path / "missing"
    )
    assert not report.accepted
    assert report.items[0].cycle == 9
    assert "Missing production snapshot" in report.items[0].error
    assert not list((tmp_path / "missing").rglob("*.cif"))


def test_partial_export_reports_all_requested_frames(bundle_factory, tmp_path):
    root = bundle_factory(indices=(0, 9))
    report = export_adsorption_snapshots(
        root, output_dir=tmp_path / "partial", frames=[0, 2, 9, 10]
    )
    assert not report.accepted
    assert [i.status for i in report.items] == [
        "exported",
        "failed",
        "exported",
        "failed",
    ]
    assert "outside the production" in report.items[-1].error
    assert all(
        Path(i.structure.path).exists() for i in report.items if i.structure
    )


def test_bad_frame_does_not_publish_cif_or_block_other_frames(
    bundle_factory,
    tmp_path,
):
    root = bundle_factory()
    path = root / "work/Movies/System_0/result_9.data"
    path.write_text(path.read_text().replace("4 atoms", "5 atoms"))
    output = tmp_path / "partial"
    report = export_adsorption_snapshots(root, output_dir=output, frames="all")
    assert [item.status for item in report.items] == ["exported", "failed"]
    assert not (output / "System_0/snapshot_9.cif").exists()
    assert not (output / "System_0/snapshot_9.metadata.json").exists()
    assert (
        SnapshotExportResult.model_validate_json(
            (output / "export_manifest.json").read_text()
        )
        == report
    )


def test_cif_overlap_that_ase_would_merge_is_rejected(bundle_factory, tmp_path):
    root = bundle_factory()
    movie(
        root / "work/Movies/System_0/result_9.data",
        rows=[
            (1, 1, 1, 0.2, [2, 3, 4], "framework.cif"),
            (2, 2, 2, 0.0, [2, 3, 4], "TIP4P"),
            (3, 2, 3, 0.52, [2.75695, 3.58588, 4], "TIP4P"),
            (4, 2, 3, 0.52, [1.24305, 3.58588, 4], "TIP4P"),
        ],
    )
    report = export_adsorption_snapshots(root, output_dir=tmp_path / "overlap")
    assert not report.accepted
    assert "merge overlapping" in report.items[0].error


@pytest.mark.parametrize("frames", [[], [True], [-1], [1, 1], [1.2], "last", 9])
def test_bad_selection_rejected(bundle_factory, tmp_path, frames):
    with pytest.raises(ValueError):
        export_adsorption_snapshots(
            bundle_factory(), output_dir=tmp_path / "bad", frames=frames
        )
    assert not (tmp_path / "bad").exists()


def test_no_movies_and_collision(bundle_factory, tmp_path):
    with pytest.raises(ValueError, match="No production snapshots"):
        export_adsorption_snapshots(
            bundle_factory(indices=()),
            output_dir=tmp_path / "none",
            frames="all",
        )
    root = bundle_factory()
    output = tmp_path / "export"
    output.mkdir()
    with pytest.raises(FileExistsError):
        export_adsorption_snapshots(root, output_dir=output)
    with pytest.raises(ValueError, match="outside"):
        export_adsorption_snapshots(root, output_dir=root / "export")


@pytest.mark.parametrize(
    "mode",
    ["prepared", "running", "interrupted", "exit", "log", "input", "lock"],
)
def test_unavailable_bundle_rejected(bundle_factory, tmp_path, mode):
    root = bundle_factory()
    record_path = root / "run.json"
    record = json.loads(record_path.read_text())
    if mode in {"prepared", "running", "interrupted"}:
        record["state"] = mode
        if mode == "interrupted":
            record["failure"] = {
                "code": "timeout",
                "stage": "worker",
                "message": "interrupted",
            }
        record_path.write_text(json.dumps(record))
    elif mode == "exit":
        (root / "exit.json").write_text('{"returncode": 1}')
    elif mode == "log":
        (root / "work/raspa.log").write_text("RUNNING PRODUCTION PHASE")
    elif mode == "input":
        with (root / "work/pseudo_atoms.def").open("a") as stream:
            stream.write("# changed\n")
    else:
        (root / ".matkit.lock").touch()
    with pytest.raises(ValueError):
        export_adsorption_snapshots(root, output_dir=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_statistics_failure_can_export_geometry(bundle_factory, tmp_path):
    root = bundle_factory()
    record = json.loads((root / "run.json").read_text())
    record.update(
        state="failed",
        payload=None,
        numerical_validity="invalid",
        failure={
            "code": "ValueError",
            "stage": "calculation",
            "message": "Non-finite adsorption statistics",
        },
    )
    (root / "result.json").write_text(json.dumps(record))
    report = export_adsorption_snapshots(root, output_dir=tmp_path / "geometry")
    assert report.accepted
    assert report.provenance["matkit_state"] == "failed"
    assert report.provenance["matkit_failure"] == record["failure"]


def test_recorded_snapshot_hash_checked(bundle_factory, tmp_path):
    root = bundle_factory()
    record = json.loads((root / "run.json").read_text())
    relative = "work/Movies/System_0/result_9.data"
    record["artifacts"] = [
        {
            "path": relative,
            "sha256": "0" * 64,
            "size_bytes": 1,
            "media_type": "text/plain",
            "role": "output",
        }
    ]
    (root / "run.json").write_text(json.dumps(record))
    report = export_adsorption_snapshots(root, output_dir=tmp_path / "changed")
    assert not report.accepted
    assert "Snapshot changed" in report.items[0].error


def test_cli_api_agreement_and_exit_status(bundle_factory, tmp_path):
    root = bundle_factory()
    output = tmp_path / "cli"
    args = [
        "adsorption",
        "export-snapshots",
        str(root),
        "--outdir",
        str(output),
        "--check",
        "--frames",
        "0,9",
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    direct = export_adsorption_snapshots(
        root, output_dir=output, check_only=True, frames=[0, 9]
    )
    assert json.loads(result.output) == direct.model_dump(mode="json")
    assert not output.exists()
    args[-1] = "2,9"
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 1
    assert len(json.loads(result.output)["items"]) == 2
    args[-1] = "nonsense"
    assert CliRunner().invoke(main, args).exit_code == 2
