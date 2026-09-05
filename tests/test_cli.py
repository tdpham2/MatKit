"""Tests for matkit CLI."""

import json
import importlib.util
from pathlib import Path
import sys

import pytest

from click.testing import CliRunner
from matkit.cli import main


class TestCLI:
    """Tests for the Click CLI interface."""

    def test_main_help(self):
        """CLI should show help text."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "MatKit CLI" in result.output

    def test_graspa_group(self):
        """graspa command group should exist."""
        runner = CliRunner()
        result = runner.invoke(main, ["graspa", "--help"])
        assert result.exit_code == 0
        assert "setup" in result.output
        assert "analyze" in result.output

    def test_graspa_sycl_group(self):
        """graspa_sycl command group should exist."""
        runner = CliRunner()
        result = runner.invoke(main, ["graspa_sycl", "--help"])
        assert result.exit_code == 0
        assert "setup" in result.output
        assert "analyze" in result.output

    def test_raspa2_group(self):
        """raspa2 command group should exist."""
        runner = CliRunner()
        result = runner.invoke(main, ["raspa2", "--help"])
        assert result.exit_code == 0
        assert "setup" in result.output
        assert "analyze" in result.output

    def test_tobacco_group(self):
        """tobacco command group should exist."""
        runner = CliRunner()
        result = runner.invoke(main, ["tobacco", "--help"])
        assert result.exit_code == 0
        assert "create" in result.output

    def test_graspa_setup_help(self):
        """graspa setup should show options."""
        runner = CliRunner()
        result = runner.invoke(main, ["graspa", "setup", "--help"])
        assert result.exit_code == 0
        assert "--cif" in result.output
        assert "--outdir" in result.output
        assert "--adsorbate" in result.output

    def test_unknown_command(self):
        """Unknown command should fail gracefully."""
        runner = CliRunner()
        result = runner.invoke(main, ["nonexistent"])
        assert result.exit_code != 0

    def test_mlip_group_exposes_runtime_neutral_commands(self):
        runner = CliRunner()
        result = runner.invoke(main, ["mlip", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "run-batch" in result.output

    def test_mlip_run_delegates_to_python_api(
        self, sample_cif, tmp_path, monkeypatch
    ):
        import matkit.mlip

        received = {}

        def fake_run(input_file, backend, calculation, output_file):
            received.update(
                {
                    "input": input_file,
                    "backend": backend,
                    "calculation": calculation,
                    "output": output_file,
                }
            )
            return {
                "success": True,
                "energy": -1.25,
                "energy_unit": "eV",
                "converged": True,
            }

        monkeypatch.setattr(matkit.mlip, "run_mlip", fake_run)
        output = tmp_path / "result.json"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "mlip",
                "run",
                "--input",
                sample_cif,
                "--output",
                str(output),
                "--backend",
                "rootstock",
                "--checkpoint",
                "mace-mp-0-medium",
                "--cluster",
                "polaris",
                "--device",
                "cuda",
                "--setup-kwarg",
                'default_dtype="float32"',
            ],
        )

        assert result.exit_code == 0
        assert json.loads(result.output)["energy"] == -1.25
        assert received["backend"].cluster == "polaris"
        assert received["backend"].setup_kwargs == {"default_dtype": "float32"}
        assert received["output"] == str(output)

    def test_mlip_batch_requires_one_input_source(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "mlip",
                "run-batch",
                "--backend",
                "ase-mace",
                "--checkpoint",
                "medium",
            ],
        )
        assert result.exit_code != 0
        assert "exactly one" in result.output


@pytest.mark.parametrize(
    "backend,flags,message",
    [
        ("rootstock", ["--dtype", "float32"], "--setup-kwarg"),
        ("ase-mace", ["--cluster", "polaris"], "requires --backend rootstock"),
        ("ase-mace", ["--enable-cueq"], "requires --backend nvalchemi-mace"),
        ("nvalchemi-mace", ["--dispersion"], "requires --backend ase-mace"),
        (
            "ase-mace",
            ["--calculator-type", "mace_anicc", "--dtype", "float64"],
            "mace_anicc factory",
        ),
        (
            "ase-mace",
            ["--calculator-type", "mace_off", "--dispersion"],
            "requires --calculator-type mace_mp",
        ),
        ("ase-mace", ["--steps", "10"], "requires --driver opt"),
        ("ase-mace", ["--fmax", "0.01"], "requires --driver opt"),
        ("nvalchemi-mace", ["--dt", "0.1"], "requires --driver opt"),
        (
            "nvalchemi-mace",
            ["--driver", "opt", "--optimizer", "bfgs"],
            "only the FIRE optimizer",
        ),
        ("ase-mace", ["--driver", "opt", "--fmax", "nan"], "finite"),
    ],
)
def test_mlip_rejects_unsupported_explicit_options(
    sample_cif,
    monkeypatch,
    backend,
    flags,
    message,
):
    import matkit.mlip

    monkeypatch.setattr(
        matkit.mlip,
        "run_mlip",
        lambda *a, **kw: pytest.fail("invalid request ran"),
    )
    result = CliRunner().invoke(
        main,
        [
            "mlip",
            "run",
            "--input",
            sample_cif,
            "--backend",
            backend,
            "--checkpoint",
            "medium",
            *flags,
        ],
    )
    assert result.exit_code == 2
    assert message in result.output


@pytest.mark.parametrize(
    "success,converged,code",
    [(True, True, 0), (True, False, 1), (False, False, 1)],
)
def test_mlip_single_exit_and_summary(
    sample_cif, monkeypatch, success, converged, code
):
    import matkit.mlip

    monkeypatch.setattr(
        matkit.mlip,
        "run_mlip",
        lambda *a, **kw: {
            "success": success,
            "converged": converged,
            "energy": 1.0 if success else None,
            "energy_unit": "eV",
            "error": "" if success else "calculation failed",
        },
    )
    result = CliRunner().invoke(
        main,
        [
            "mlip",
            "run",
            "--input",
            sample_cif,
            "--backend",
            "ase-mace",
            "--checkpoint",
            "medium",
            "--driver",
            "opt",
        ],
    )
    assert result.exit_code == code
    summary = json.loads(result.stdout)
    assert summary["converged"] is converged
    assert summary["status"] == (
        "failure" if not success else "success" if converged else "unconverged"
    )
    assert bool(result.stderr) is bool(code)


@pytest.mark.parametrize("failed,unconverged", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_mlip_batch_exit_and_summary(
    sample_cif, monkeypatch, failed, unconverged
):
    import matkit.mlip

    def fake_batch(*args, **kwargs):
        return {
            "status": "partial" if failed else "completed",
            "total": 2,
            "succeeded": 2 - failed,
            "failed": failed,
            "manifest_file": "batch.json",
            "results": [
                {"success": True, "converged": not unconverged},
                {"success": not failed, "converged": not failed},
            ],
        }

    monkeypatch.setattr(matkit.mlip, "run_mlip_batch", fake_batch)
    result = CliRunner().invoke(
        main,
        [
            "mlip",
            "run-batch",
            "--input",
            sample_cif,
            "--backend",
            "ase-mace",
            "--checkpoint",
            "medium",
            "--driver",
            "opt",
        ],
    )
    assert result.exit_code == int(bool(failed or unconverged))
    assert json.loads(result.stdout)["unconverged"] == unconverged
    assert bool(result.stderr) is bool(failed or unconverged)


def test_mlip_runtime_error_summary(sample_cif, monkeypatch):
    import matkit.mlip

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(matkit.mlip, "run_mlip", fail)
    result = CliRunner().invoke(
        main,
        [
            "mlip",
            "run",
            "--input",
            sample_cif,
            "--backend",
            "ase-mace",
            "--checkpoint",
            "medium",
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "status": "failure",
        "error": "disk full",
    }
    assert "disk full" in result.stderr


@pytest.fixture
def gpu_example():
    path = Path(__file__).parents[1] / "examples/mlip_gpu.py"
    spec = importlib.util.spec_from_file_location("mlip_gpu_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_example_rejects_unused_options(gpu_example, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlip_gpu.py",
            "--backend",
            "ase-mace",
            "--cluster",
            "polaris",
            "in.cif",
        ],
    )
    with pytest.raises(SystemExit) as error:
        gpu_example.main()
    assert error.value.code == 2


def test_gpu_example_unconverged_exit(gpu_example, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlip_gpu.py",
            "--backend",
            "ase-mace",
            "--driver",
            "opt",
            "in.cif",
        ],
    )
    monkeypatch.setattr(
        gpu_example,
        "run_mlip_batch",
        lambda *a, **kw: {
            "status": "completed",
            "total": 1,
            "succeeded": 1,
            "failed": 0,
            "manifest_file": "batch.json",
            "results": [{"success": True, "converged": False}],
        },
    )
    assert gpu_example.main() == 1
    output = capsys.readouterr()
    assert json.loads(output.out)["unconverged"] == 1
    assert "unconverged" in output.err
