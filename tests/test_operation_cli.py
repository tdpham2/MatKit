"""Real CLI execution uses the same contracts as Python."""

import json
from pathlib import Path
import sys

from click.testing import CliRunner

from matkit.api import ExecutionConfig, PoreRequest, StructureRef, run
from matkit.cli import main


def inputs(tmp_path, sample_cif, *engine_args):
    request = {
        "operation": "pores",
        "structure": {"path": sample_cif},
        "analyses": ["res", "sa"],
    }
    spec = tmp_path / "pores.json"
    spec.write_text(json.dumps(request))
    fixture = Path(__file__).parent / "fixtures" / "fake_engine.py"
    config = {
        "executables": {
            "zeopp": [sys.executable, str(fixture), "zeopp", *engine_args]
        }
    }
    profile = tmp_path / "execution.json"
    profile.write_text(json.dumps(config))
    return spec, profile, config


def test_cli_python_equivalent_results(sample_cif, tmp_path):
    spec, profile, config = inputs(tmp_path, sample_cif)
    result = CliRunner().invoke(
        main,
        [
            "pores",
            "--spec",
            str(spec),
            "--execution",
            str(profile),
            "--outdir",
            str(tmp_path / "cli"),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    python = run(
        PoreRequest(
            structure=StructureRef(path=sample_cif), analyses=["res", "sa"]
        ),
        output_dir=tmp_path / "python",
        execution=ExecutionConfig(**config),
    )
    assert data["payload"] == python.payload.model_dump(mode="json")
    assert data["checks"] == [c.model_dump(mode="json") for c in python.checks]


def test_cli_preparation_then_execution_and_inspection(sample_cif, tmp_path):
    spec, profile, _ = inputs(tmp_path, sample_cif)
    runner = CliRunner()
    root = tmp_path / "run"
    prepared = runner.invoke(
        main, ["prepare", "--spec", str(spec), "--outdir", str(root)]
    )
    assert prepared.exit_code == 0, prepared.output
    assert json.loads(prepared.stdout)["state"] == "prepared"
    executed = runner.invoke(
        main, ["execute", str(root), "--execution", str(profile)]
    )
    assert executed.exit_code == 0, executed.output
    inspected = runner.invoke(main, ["inspect", str(root)])
    assert inspected.exit_code == 0
    assert json.loads(executed.stdout) == json.loads(inspected.stdout)
    rerun = runner.invoke(
        main, ["execute", str(root), "--execution", str(profile)]
    )
    assert rerun.exit_code == 2


def test_cli_failure_is_structured_and_nonzero(sample_cif, tmp_path):
    spec, profile, _ = inputs(tmp_path, sample_cif, "--fail")
    result = CliRunner().invoke(
        main,
        [
            "pores",
            "--spec",
            str(spec),
            "--execution",
            str(profile),
            "--outdir",
            str(tmp_path / "run"),
        ],
    )
    assert result.exit_code == 1, result.output
    data = json.loads(result.stdout)
    assert data["state"] == "failed"
    assert "code 7" in data["failure"]["message"]


def test_cli_invalid_scientific_request_exits_two(sample_cif, tmp_path):
    spec, _, _ = inputs(tmp_path, sample_cif)
    data = json.loads(spec.read_text())
    data["probe_radius"] = -1
    spec.write_text(json.dumps(data))
    result = CliRunner().invoke(
        main, ["pores", "--spec", str(spec), "--outdir", str(tmp_path / "run")]
    )
    assert result.exit_code == 2
    assert not (tmp_path / "run").exists()


def test_cli_batch_executes_in_worker(sample_cif, tmp_path):
    spec, profile, _ = inputs(tmp_path, sample_cif)
    request = json.loads(spec.read_text())
    spec.write_text(json.dumps([request, request]))
    result = CliRunner().invoke(
        main,
        [
            "batch",
            "--spec",
            str(spec),
            "--execution",
            str(profile),
            "--outdir",
            str(tmp_path / "batch"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.stdout)["items"]) == 2
