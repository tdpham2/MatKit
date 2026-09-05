"""Cross-adapter durability and outcomes, without model downloads."""

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys

import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.io import write

from matkit.api import (
    AdsorptionRequest,
    AlchemiAdapter,
    EvaluateRequest,
    ExecutionConfig,
    MLIPMethod,
    PoreRequest,
    RelaxRequest,
    StructureRef,
    analyze_adsorption,
    evaluate,
    execute,
    inspect_run,
    prepare,
    run,
    run_batch,
)
from matkit.api import bundles


@pytest.fixture
def input_file(tmp_path):
    path = tmp_path / "cu.extxyz"
    write(
        path,
        Atoms("Cu2", positions=[[0, 0, 0], [3, 0, 0]], cell=[10] * 3, pbc=True),
    )
    return str(path)


@pytest.fixture
def mock_calculator(monkeypatch):
    from matkit.mlip import runner

    calls = []

    @contextmanager
    def context(backend):
        calls.append(backend)
        yield EMT()

    monkeypatch.setattr(runner, "_ase_backend_context", context)
    return calls


def fake_execution(engine, *args, **kwargs):
    fixture = Path(__file__).parent / "fixtures" / "fake_engine.py"
    return ExecutionConfig(
        executables={engine: [sys.executable, str(fixture), engine, *args]},
        **kwargs,
    )


def energy_request(path, **kwargs):
    return EvaluateRequest(
        structure=StructureRef(path=path),
        method=MLIPMethod(checkpoint="fixture"),
        **kwargs,
    )


def test_energy_only_calculator_does_not_require_forces(
    input_file, tmp_path, monkeypatch
):
    from matkit.mlip import runner

    class EnergyOnly(Calculator):
        implemented_properties = ["energy"]

        def calculate(
            self, atoms=None, properties=None, system_changes=all_changes
        ):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": -2.5}

    @contextmanager
    def context(_):
        yield EnergyOnly()

    monkeypatch.setattr(runner, "_ase_backend_context", context)
    result = evaluate(energy_request(input_file), output_dir=tmp_path / "run")
    assert result.accepted
    assert result.payload.potential_energy == -2.5
    assert result.payload.forces is None
    failed = evaluate(
        energy_request(input_file, properties=["forces"]),
        output_dir=tmp_path / "forces",
    )
    assert not failed.accepted
    assert failed.failure is not None


def test_relaxation_retains_unconverged_result(
    input_file, tmp_path, mock_calculator
):
    request = RelaxRequest(
        structure=StructureRef(path=input_file),
        method=MLIPMethod(checkpoint="fixture"),
        steps=1,
        fmax=1e-12,
    )
    result = run(request, output_dir=tmp_path / "run")
    assert result.state == "completed"
    assert result.numerical_validity == "valid"
    assert not result.payload.converged
    assert not result.accepted
    assert inspect_run(tmp_path / "run") == result


def test_homogeneous_batch_reuses_calculator(
    input_file, tmp_path, mock_calculator
):
    request = energy_request(input_file)
    result = run_batch([request, request], output_dir=tmp_path / "batch")
    assert result.state == "completed"
    assert len(mock_calculator) == 1
    assert [i["index"] for i in result.items] == [0, 1]
    assert all(i["accepted"] for i in result.items)
    assert result.items[0]["result_file"] != result.items[1]["result_file"]


def test_batch_retains_success_when_input_is_missing(
    input_file, tmp_path, mock_calculator
):
    result = run_batch(
        [energy_request(input_file), energy_request("/missing/input.xyz")],
        output_dir=tmp_path / "batch",
    )
    assert result.state == "partial"
    assert result.items[0]["accepted"]
    assert result.items[1]["state"] == "failed"


def test_mixed_batch_rejected_before_side_effects(input_file, tmp_path):
    with pytest.raises(ValueError, match="homogeneous"):
        run_batch(
            [
                energy_request(input_file),
                energy_request(input_file, properties=["forces"]),
            ],
            output_dir=tmp_path / "run",
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("recovery_write_fails", [False, True])
def test_completed_result_survives_manifest_write_failure(
    input_file, tmp_path, mock_calculator, monkeypatch, recovery_write_fails
):
    root = tmp_path / "run"
    original = bundles.atomic_json

    def write_json(path, data):
        if path.name == "run.json" and (root / "result.json").exists():
            raise OSError("manifest unavailable")
        return original(path, data)

    monkeypatch.setattr(bundles, "atomic_json", write_json)
    if recovery_write_fails:
        from matkit.api import runtime

        monkeypatch.setattr(runtime, "atomic_json", write_json)
    with pytest.raises(OSError, match="manifest unavailable"):
        evaluate(energy_request(input_file), output_dir=root)
    from matkit.api import RunResult

    committed = RunResult.model_validate_json(
        (root / "result.json").read_text()
    )
    assert committed.accepted
    inspected = inspect_run(root)
    assert inspected.payload == committed.payload
    assert inspected.numerical_validity == "valid"
    assert inspected.state == (
        "completed" if recovery_write_fails else "interrupted"
    )


def test_interrupted_batch_retains_first_result(
    input_file, tmp_path, mock_calculator, monkeypatch
):
    from matkit.mlip import runner

    original = runner._run_ase_item
    count = 0

    def item(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise KeyboardInterrupt("stopped")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "_run_ase_item", item)
    root = tmp_path / "batch"
    with pytest.raises(KeyboardInterrupt):
        run_batch([energy_request(input_file)] * 2, output_dir=root)
    batch = json.loads((root / "batch_manifest.json").read_text())
    assert batch["state"] == "interrupted"
    assert inspect_run(root / "00000").accepted
    assert inspect_run(root / "00001").state == "interrupted"


def test_native_batch_grouping_without_cuda(input_file, tmp_path, monkeypatch):
    from matkit.mlip import runner

    chunks = []
    monkeypatch.setattr(runner, "_load_nvalchemi_model", lambda _: object())

    def chunk(model, entries, backend, calculation):
        chunks.append(len(entries))
        return [
            (
                index,
                runner._success_result(
                    path,
                    backend,
                    calculation,
                    atoms,
                    -1,
                    [[0, 0, 0]] * len(atoms),
                    None,
                    True,
                    0,
                    0.1,
                ),
            )
            for index, path, atoms in entries
        ]

    monkeypatch.setattr(runner, "_run_nvalchemi_chunk", chunk)
    request = energy_request(input_file, adapter=AlchemiAdapter(batch_size=2))
    result = run_batch([request] * 3, output_dir=tmp_path / "batch")
    assert result.state == "completed"
    assert chunks == [2, 1]


def test_relocated_bundle_and_all_requested_pore_outputs(sample_cif, tmp_path):
    root = tmp_path / "original"
    prepare(
        PoreRequest(
            structure=StructureRef(path=sample_cif), analyses=["res", "sa"]
        ),
        output_dir=root,
    )
    relocated = tmp_path / "relocated"
    shutil.move(root, relocated)
    result = execute(relocated, execution=fake_execution("zeopp"))
    assert result.accepted
    assert set(result.payload.results) == {"res", "sa"}
    assert (relocated / "work" / "engine.stdout.log").exists()


@pytest.mark.parametrize("args", [("--fail",), ("--partial",)])
def test_external_engine_failure_and_missing_analysis(
    sample_cif, tmp_path, args
):
    result = run(
        PoreRequest(
            structure=StructureRef(path=sample_cif), analyses=["res", "sa"]
        ),
        output_dir=tmp_path / "run",
        execution=fake_execution("zeopp", *args),
    )
    assert result.state == "failed"
    assert result.failure is not None


def test_tampered_bundle_is_refused(sample_cif, tmp_path):
    root = tmp_path / "run"
    prepare(
        PoreRequest(structure=StructureRef(path=sample_cif)), output_dir=root
    )
    (root / "inputs" / "radii.rad").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        execute(root, execution=fake_execution("zeopp"))


def test_subprocess_execution_and_timeout(sample_cif, tmp_path):
    request = PoreRequest(structure=StructureRef(path=sample_cif))
    result = run(
        request,
        output_dir=tmp_path / "good",
        execution=fake_execution("zeopp", mode="subprocess"),
    )
    assert result.accepted
    interrupted = run(
        request,
        output_dir=tmp_path / "timeout",
        execution=fake_execution(
            "zeopp", "--sleep", mode="subprocess", timeout_s=2
        ),
    )
    assert interrupted.state == "interrupted"
    from matkit.api.structures import sha256

    assert all(
        sha256(tmp_path / "timeout" / a.path) == a.sha256
        for a in interrupted.artifacts
    )
    assert not (tmp_path / "timeout" / ".matkit.lock").exists()


@pytest.mark.parametrize(
    "unit,fugacity,expected",
    [
        ("mol/kg", "PR-EOS", 12),
        ("mg/g", "PR-EOS", 6),
        ("g/L", "PR-EOS", 14),
        ("mol/kg", 1.0, 7),
    ],
)
def test_single_component_adsorption_requires_charges_and_collects(
    sample_cif, tmp_path, unit, fugacity, expected
):
    request = AdsorptionRequest(
        structure=StructureRef(path=sample_cif),
        adsorbate="CO2",
        temperature_K=298,
        pressure_Pa=1e5,
        unit=unit,
        fugacity_coefficient=fugacity,
    )
    with pytest.raises(ValueError, match="_atom_site_charge"):
        prepare(request, output_dir=tmp_path / "uncharged")
    charged = tmp_path / "charged.cif"
    text = (
        Path(sample_cif)
        .read_text()
        .replace(
            "  _atom_site_occupancy",
            "  _atom_site_charge\n  _atom_site_occupancy",
        )
    )
    text = (
        text.replace("0.00000  1.0000", "0.00000  0.0  1.0000")
        .replace("0.25000  1.0000", "0.25000  0.0  1.0000")
        .replace("0.75000  1.0000", "0.75000  0.0  1.0000")
    )
    charged.write_text(text)
    request.structure.path = str(charged)
    result = run(
        request,
        output_dir=tmp_path / "charged_run",
        execution=fake_execution("graspa"),
    )
    assert result.accepted
    assert result.payload.uptake == expected
    assert result.payload.component == "CO2"
    assert result.checks[0].status == "unknown"
    assert (
        tmp_path / "charged_run" / "inputs" / "structure.cif"
    ).read_bytes() == charged.read_bytes()
    assert analyze_adsorption(tmp_path / "charged_run") == result
