"""Tests for matkit CLI."""

import json

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
