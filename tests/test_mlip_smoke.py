"""Check the live smoke runner's orchestration without launching engines."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from ase import Atoms
from ase.io import write


@pytest.fixture
def smoke():
    path = Path(__file__).parents[1] / "alcf/polaris/mlip/smoke.py"
    spec = importlib.util.spec_from_file_location("mlip_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_smoke_case_does_not_skip_other_backends(
    smoke, tmp_path, monkeypatch
):
    input_file = tmp_path / "input.xyz"
    write(
        input_file, Atoms("Cu2", positions=[[0, 0, 0], [2, 0, 0]], cell=[8] * 3)
    )
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1 if len(commands) == 1 else 0)

    monkeypatch.setattr(smoke, "_probe", lambda command: {"mocked": True})
    monkeypatch.setattr(smoke, "subprocess", SimpleNamespace(run=fake_run))
    monkeypatch.setattr(smoke, "_validate_outputs", lambda *args: [])
    out = tmp_path / "evidence"
    assert (
        smoke.main(["--input", str(input_file), "--output-dir", str(out)]) == 1
    )
    report = json.loads((out / "smoke_report.json").read_text())
    assert len(commands) == len(report["cases"]) == 6
    assert report["cases"][0]["status"] == "failed"
    assert all(case["status"] == "passed" for case in report["cases"][1:])
    for command in commands[-2:]:
        assert "run-batch" in command
        assert command.count("--input") == 2
        assert command[command.index("--batch-size") + 1] == "2"
    assert len(report["input_sha256"]) == 64


def test_smoke_rejects_unconverged_results(smoke, tmp_path):
    atoms = Atoms("Cu", cell=[8] * 3)
    data = {
        "success": True,
        "error": "",
        "energy": 1.0,
        "energy_unit": "eV",
        "force_unit": "eV/angstrom",
        "forces": [[0, 0, 0]],
        "stress": None,
        "converged": False,
        "n_steps": 1,
        "final_structure": {
            "positions": atoms.positions.tolist(),
            "atomic_numbers": atoms.numbers.tolist(),
            "cell": atoms.cell.array.tolist(),
            "pbc": atoms.pbc.tolist(),
        },
    }
    (tmp_path / "result.json").write_text(json.dumps(data))
    with pytest.raises(AssertionError, match="did not converge"):
        smoke._validate_outputs(tmp_path, [atoms], "opt", 0.01, False)
    assert (
        smoke._validate_outputs(tmp_path, [atoms], "energy", 0.01, False)[0][
            "energy"
        ]
        == 1.0
    )
