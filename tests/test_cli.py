"""Tests for matkit CLI."""

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
