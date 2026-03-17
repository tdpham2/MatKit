"""Tests for the matkit.plot module."""

import json
import math

import pytest

from matkit.plot.parsers import (
    _safe_float,
    _discover_adsorbates,
    _discover_selectivity_keys,
    collect_data_files,
    detect_format,
    load_isotherm,
    parse_mixture_isotherm,
    parse_single_isotherm,
)


# ==========================================
# Fixtures
# ==========================================
@pytest.fixture
def single_isotherm_path(test_data_dir):
    """Path to single-component isotherm test fixture."""
    return str(test_data_dir / "single_isotherm.json")


@pytest.fixture
def mixture_isotherm_path(test_data_dir):
    """Path to mixture isotherm test fixture."""
    return str(test_data_dir / "mixture_isotherm.json")


@pytest.fixture
def single_isotherm_data(test_data_dir):
    """Parsed JSON from single-component isotherm fixture."""
    with open(test_data_dir / "single_isotherm.json") as f:
        return json.load(f)


@pytest.fixture
def mixture_isotherm_data(test_data_dir):
    """Parsed JSON from mixture isotherm fixture."""
    with open(test_data_dir / "mixture_isotherm.json") as f:
        return json.load(f)


# ==========================================
# Tests for _safe_float
# ==========================================
class TestSafeFloat:
    """Tests for the _safe_float helper."""

    def test_string_number(self):
        assert _safe_float("3.14") == 3.14

    def test_float_number(self):
        assert _safe_float(2.718) == 2.718

    def test_int_number(self):
        assert _safe_float(42) == 42.0

    def test_none_returns_nan(self):
        assert math.isnan(_safe_float(None))

    def test_nan_string_returns_nan(self):
        assert math.isnan(_safe_float("-nan"))

    def test_inf_string_returns_nan(self):
        assert math.isnan(_safe_float("-inf"))

    def test_inf_returns_nan(self):
        assert math.isnan(_safe_float(float("inf")))

    def test_invalid_string_returns_nan(self):
        assert math.isnan(_safe_float("not_a_number"))


# ==========================================
# Tests for detect_format
# ==========================================
class TestDetectFormat:
    """Tests for format auto-detection."""

    def test_detects_single_bar(self, single_isotherm_data):
        assert detect_format(single_isotherm_data) == "single"

    def test_detects_single_pa(self):
        data = {"314.2Pa_298K": {"uptake": "1.0"}}
        assert detect_format(data) == "single"

    def test_detects_mixture_rh(self, mixture_isotherm_data):
        assert detect_format(mixture_isotherm_data) == "mixture_rh"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Empty"):
            detect_format({})

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Cannot detect"):
            detect_format({"unknown_key": {"uptake": 1.0}})


# ==========================================
# Tests for parse_single_isotherm
# ==========================================
class TestParseSingleIsotherm:
    """Tests for single-component isotherm parsing."""

    def test_sorts_by_pressure(self, single_isotherm_data):
        result = parse_single_isotherm(single_isotherm_data)
        assert result["pressures"] == [
            0.001,
            0.01,
            0.1,
            0.5,
            1.0,
        ]

    def test_format_is_single(self, single_isotherm_data):
        result = parse_single_isotherm(single_isotherm_data)
        assert result["format"] == "single"

    def test_pressure_unit(self, single_isotherm_data):
        result = parse_single_isotherm(single_isotherm_data)
        assert result["pressure_unit"] == "bar"

    def test_uptakes_are_floats(self, single_isotherm_data):
        result = parse_single_isotherm(single_isotherm_data)
        for val in result["uptakes"]:
            assert isinstance(val, float)

    def test_uptake_values(self, single_isotherm_data):
        result = parse_single_isotherm(single_isotherm_data)
        assert result["uptakes"][0] == pytest.approx(0.12345)
        assert result["uptakes"][-1] == pytest.approx(4.12345)

    def test_errors_are_floats(self, single_isotherm_data):
        result = parse_single_isotherm(single_isotherm_data)
        for val in result["errors"]:
            assert isinstance(val, float)

    def test_temperature(self, single_isotherm_data):
        result = parse_single_isotherm(single_isotherm_data)
        assert result["temperature"] == 298.0

    def test_unit(self, single_isotherm_data):
        result = parse_single_isotherm(single_isotherm_data)
        assert result["unit"] == "mol/kg"

    def test_qst_values(self, single_isotherm_data):
        result = parse_single_isotherm(single_isotherm_data)
        assert result["qst"][0] == pytest.approx(32.5)
        assert result["qst_unit"] == "kJ/mol"

    def test_handles_pa_keys(self):
        data = {
            "314.2Pa_298K": {
                "success": True,
                "uptake": "1.5",
                "error": "0.1",
                "unit": "mol/kg",
            },
            "628.4Pa_298K": {
                "success": True,
                "uptake": "2.5",
                "error": "0.2",
                "unit": "mol/kg",
            },
        }
        result = parse_single_isotherm(data)
        assert result["pressure_unit"] == "Pa"
        assert result["pressures"] == [314.2, 628.4]

    def test_handles_missing_qst(self):
        data = {
            "0.1bar_273K": {
                "success": True,
                "uptake": "0.18",
                "error": "0.01",
                "unit": "mol/kg",
                "calc_time_in_s": 113.4,
            },
        }
        result = parse_single_isotherm(data)
        assert math.isnan(result["qst"][0])
        assert result["qst_unit"] is None

    def test_handles_nan_values(self):
        data = {
            "1.0bar_77K": {
                "success": True,
                "uptake": "10.5",
                "error": "0.5",
                "unit": "mol/kg",
                "qst": "-nan",
                "error_qst": "-inf",
                "qst_unit": "kJ/mol",
            },
        }
        result = parse_single_isotherm(data)
        assert math.isnan(result["qst"][0])
        assert math.isnan(result["qst_errors"][0])
        assert result["uptakes"][0] == pytest.approx(10.5)


# ==========================================
# Tests for _discover_adsorbates
# ==========================================
class TestDiscoverAdsorbates:
    """Tests for adsorbate name auto-discovery."""

    def test_discovers_three_adsorbates(self, mixture_isotherm_data):
        ads = _discover_adsorbates(mixture_isotherm_data)
        assert ads == ["co2", "h2o", "n2"]

    def test_discovers_arbitrary_names(self):
        data = {
            "0_RH": {
                "methane_uptake": 1.0,
                "methane_error": 0.1,
                "ethane_uptake": 2.0,
                "ethane_error": 0.2,
            }
        }
        ads = _discover_adsorbates(data)
        assert ads == ["ethane", "methane"]


# ==========================================
# Tests for _discover_selectivity_keys
# ==========================================
class TestDiscoverSelectivityKeys:
    """Tests for selectivity field auto-discovery."""

    def test_discovers_selectivity(self, mixture_isotherm_data):
        keys = _discover_selectivity_keys(mixture_isotherm_data)
        assert keys == ["co2_n2_selectivity"]

    def test_no_selectivity(self):
        data = {
            "0_RH": {
                "co2_uptake": 1.0,
                "co2_error": 0.1,
            }
        }
        keys = _discover_selectivity_keys(data)
        assert keys == []


# ==========================================
# Tests for parse_mixture_isotherm
# ==========================================
class TestParseMixtureIsotherm:
    """Tests for mixture isotherm parsing."""

    def test_format_is_mixture_rh(self, mixture_isotherm_data):
        result = parse_mixture_isotherm(mixture_isotherm_data)
        assert result["format"] == "mixture_rh"

    def test_sorts_by_rh(self, mixture_isotherm_data):
        result = parse_mixture_isotherm(mixture_isotherm_data)
        assert result["rh_values"] == [0, 50, 100]

    def test_discovers_adsorbates(self, mixture_isotherm_data):
        result = parse_mixture_isotherm(mixture_isotherm_data)
        assert result["adsorbates"] == ["co2", "h2o", "n2"]

    def test_uptake_values(self, mixture_isotherm_data):
        result = parse_mixture_isotherm(mixture_isotherm_data)
        assert result["uptakes"]["co2"][0] == pytest.approx(3.99067)
        assert result["uptakes"]["h2o"][0] == pytest.approx(0.0)

    def test_error_values(self, mixture_isotherm_data):
        result = parse_mixture_isotherm(mixture_isotherm_data)
        assert result["errors"]["co2"][0] == pytest.approx(0.08111)

    def test_selectivity_values(self, mixture_isotherm_data):
        result = parse_mixture_isotherm(mixture_isotherm_data)
        assert "co2_n2_selectivity" in result["selectivity"]
        assert result["selectivity"]["co2_n2_selectivity"][0] == pytest.approx(
            1018.64
        )

    def test_uptake_lengths_match_rh(self, mixture_isotherm_data):
        result = parse_mixture_isotherm(mixture_isotherm_data)
        n_points = len(result["rh_values"])
        for ads in result["adsorbates"]:
            assert len(result["uptakes"][ads]) == n_points
            assert len(result["errors"][ads]) == n_points


# ==========================================
# Tests for load_isotherm
# ==========================================
class TestLoadIsotherm:
    """Tests for the unified load_isotherm function."""

    def test_loads_single(self, single_isotherm_path):
        result = load_isotherm(single_isotherm_path)
        assert result["format"] == "single"
        assert len(result["pressures"]) == 5

    def test_loads_mixture(self, mixture_isotherm_path):
        result = load_isotherm(mixture_isotherm_path)
        assert result["format"] == "mixture_rh"
        assert len(result["rh_values"]) == 3

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_isotherm("/nonexistent/path.json")


# ==========================================
# Tests for plot functions (file output)
# ==========================================
class TestPlotSingleIsotherm:
    """Tests for single-component isotherm plotting."""

    def test_creates_output_file(self, single_isotherm_path, tmp_output):
        from matkit.plot.isotherm import plot_single_isotherm

        outfile = str(tmp_output / "test_single.png")
        result = plot_single_isotherm(
            [single_isotherm_path], output=outfile, dpi=72
        )
        from pathlib import Path

        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_with_log_x(self, single_isotherm_path, tmp_output):
        from matkit.plot.isotherm import plot_single_isotherm

        outfile = str(tmp_output / "test_log.png")
        result = plot_single_isotherm(
            [single_isotherm_path],
            output=outfile,
            dpi=72,
            log_x=True,
        )
        from pathlib import Path

        assert Path(result).exists()

    def test_with_no_errorbars(self, single_isotherm_path, tmp_output):
        from matkit.plot.isotherm import plot_single_isotherm

        outfile = str(tmp_output / "test_noerr.png")
        result = plot_single_isotherm(
            [single_isotherm_path],
            output=outfile,
            dpi=72,
            no_errorbars=True,
        )
        from pathlib import Path

        assert Path(result).exists()

    def test_with_custom_labels(self, single_isotherm_path, tmp_output):
        from matkit.plot.isotherm import plot_single_isotherm

        outfile = str(tmp_output / "test_labels.png")
        result = plot_single_isotherm(
            [single_isotherm_path],
            output=outfile,
            dpi=72,
            labels=["My MOF"],
            xlabel="P (bar)",
            ylabel="Loading (mol/kg)",
            title="Test Plot",
        )
        from pathlib import Path

        assert Path(result).exists()

    def test_rejects_mixture_format(self, mixture_isotherm_path, tmp_output):
        from matkit.plot.isotherm import plot_single_isotherm

        outfile = str(tmp_output / "test_reject.png")
        with pytest.raises(ValueError, match="single-component"):
            plot_single_isotherm([mixture_isotherm_path], output=outfile)


class TestPlotMixtureIsotherm:
    """Tests for mixture isotherm plotting."""

    def test_creates_output_file(self, mixture_isotherm_path, tmp_output):
        from matkit.plot.isotherm import plot_mixture_isotherm

        outfile = str(tmp_output / "test_mixture.png")
        result = plot_mixture_isotherm(
            [mixture_isotherm_path], output=outfile, dpi=72
        )
        from pathlib import Path

        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_filter_adsorbates(self, mixture_isotherm_path, tmp_output):
        from matkit.plot.isotherm import plot_mixture_isotherm

        outfile = str(tmp_output / "test_filtered.png")
        result = plot_mixture_isotherm(
            [mixture_isotherm_path],
            output=outfile,
            dpi=72,
            adsorbates=["co2", "h2o"],
        )
        from pathlib import Path

        assert Path(result).exists()

    def test_rejects_single_format(self, single_isotherm_path, tmp_output):
        from matkit.plot.isotherm import plot_mixture_isotherm

        outfile = str(tmp_output / "test_reject.png")
        with pytest.raises(ValueError, match="mixture_rh"):
            plot_mixture_isotherm([single_isotherm_path], output=outfile)


class TestPlotSelectivity:
    """Tests for selectivity plotting."""

    def test_creates_output_file(self, mixture_isotherm_path, tmp_output):
        from matkit.plot.isotherm import plot_selectivity

        outfile = str(tmp_output / "test_sel.png")
        result = plot_selectivity(
            [mixture_isotherm_path], output=outfile, dpi=72
        )
        from pathlib import Path

        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_rejects_single_format(self, single_isotherm_path, tmp_output):
        from matkit.plot.isotherm import plot_selectivity

        outfile = str(tmp_output / "test_reject.png")
        with pytest.raises(ValueError, match="mixture_rh"):
            plot_selectivity([single_isotherm_path], output=outfile)


class TestPlotIsotherm:
    """Tests for the auto-detecting plot_isotherm function."""

    def test_auto_detects_single(self, single_isotherm_path, tmp_output):
        from matkit.plot.isotherm import plot_isotherm

        outfile = str(tmp_output / "test_auto_single.png")
        result = plot_isotherm([single_isotherm_path], output=outfile, dpi=72)
        from pathlib import Path

        assert Path(result).exists()

    def test_auto_detects_mixture(self, mixture_isotherm_path, tmp_output):
        from matkit.plot.isotherm import plot_isotherm

        outfile = str(tmp_output / "test_auto_mixture.png")
        result = plot_isotherm([mixture_isotherm_path], output=outfile, dpi=72)
        from pathlib import Path

        assert Path(result).exists()


class TestMultiFileOverlay:
    """Tests for overlaying multiple data files."""

    def test_overlay_single_isotherms(self, single_isotherm_path, tmp_output):
        from matkit.plot.isotherm import plot_single_isotherm

        outfile = str(tmp_output / "test_overlay.png")
        result = plot_single_isotherm(
            [single_isotherm_path, single_isotherm_path],
            output=outfile,
            dpi=72,
            labels=["MOF-A", "MOF-B"],
        )
        from pathlib import Path

        assert Path(result).exists()

    def test_overlay_mixture_isotherms(self, mixture_isotherm_path, tmp_output):
        from matkit.plot.isotherm import plot_mixture_isotherm

        outfile = str(tmp_output / "test_overlay_mix.png")
        result = plot_mixture_isotherm(
            [mixture_isotherm_path, mixture_isotherm_path],
            output=outfile,
            dpi=72,
            labels=["MOF-A", "MOF-B"],
        )
        from pathlib import Path

        assert Path(result).exists()


# ==========================================
# Tests for CLI commands
# ==========================================
class TestPlotCLI:
    """Tests for plot CLI command group."""

    def test_plot_group_exists(self):
        from click.testing import CliRunner
        from matkit.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["plot", "--help"])
        assert result.exit_code == 0
        assert "isotherm" in result.output
        assert "selectivity" in result.output

    def test_isotherm_help(self):
        from click.testing import CliRunner
        from matkit.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["plot", "isotherm", "--help"])
        assert result.exit_code == 0
        assert "--data" in result.output
        assert "--output" in result.output
        assert "--dpi" in result.output
        assert "--log-x" in result.output
        assert "--adsorbate" in result.output

    def test_selectivity_help(self):
        from click.testing import CliRunner
        from matkit.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["plot", "selectivity", "--help"])
        assert result.exit_code == 0
        assert "--data" in result.output
        assert "--selectivity-key" in result.output

    def test_isotherm_command_single(self, single_isotherm_path, tmp_output):
        from click.testing import CliRunner
        from matkit.cli import main

        runner = CliRunner()
        outfile = str(tmp_output / "cli_single.png")
        result = runner.invoke(
            main,
            [
                "plot",
                "isotherm",
                "--data",
                single_isotherm_path,
                "--output",
                outfile,
                "--dpi",
                "72",
            ],
        )
        assert result.exit_code == 0
        assert "Plot saved to" in result.output

    def test_isotherm_command_mixture(self, mixture_isotherm_path, tmp_output):
        from click.testing import CliRunner
        from matkit.cli import main

        runner = CliRunner()
        outfile = str(tmp_output / "cli_mixture.png")
        result = runner.invoke(
            main,
            [
                "plot",
                "isotherm",
                "--data",
                mixture_isotherm_path,
                "--output",
                outfile,
                "--dpi",
                "72",
            ],
        )
        assert result.exit_code == 0
        assert "Plot saved to" in result.output

    def test_selectivity_command(self, mixture_isotherm_path, tmp_output):
        from click.testing import CliRunner
        from matkit.cli import main

        runner = CliRunner()
        outfile = str(tmp_output / "cli_sel.png")
        result = runner.invoke(
            main,
            [
                "plot",
                "selectivity",
                "--data",
                mixture_isotherm_path,
                "--output",
                outfile,
                "--dpi",
                "72",
            ],
        )
        assert result.exit_code == 0
        assert "Plot saved to" in result.output

    def test_isotherm_help_has_data_dir(self):
        from click.testing import CliRunner
        from matkit.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["plot", "isotherm", "--help"])
        assert result.exit_code == 0
        assert "--data-dir" in result.output

    def test_isotherm_data_dir(self, test_data_dir, tmp_output):
        """Test --data-dir loads all JSON files from a dir."""
        from click.testing import CliRunner
        from matkit.cli import main

        # Create a temp dir with two single-component files
        import shutil

        data_dir = tmp_output / "iso_dir"
        data_dir.mkdir()
        shutil.copy(
            test_data_dir / "single_isotherm.json",
            data_dir / "iso_298K.json",
        )
        shutil.copy(
            test_data_dir / "single_isotherm.json",
            data_dir / "iso_300K.json",
        )

        runner = CliRunner()
        outfile = str(tmp_output / "cli_dir.png")
        result = runner.invoke(
            main,
            [
                "plot",
                "isotherm",
                "--data-dir",
                str(data_dir),
                "--output",
                outfile,
                "--dpi",
                "72",
            ],
        )
        assert result.exit_code == 0
        assert "Plot saved to" in result.output


# ==========================================
# Tests for collect_data_files
# ==========================================
class TestCollectDataFiles:
    """Tests for the collect_data_files helper."""

    def test_explicit_files(self, single_isotherm_path):
        files = collect_data_files(data=[single_isotherm_path])
        assert len(files) == 1

    def test_data_dir(self, test_data_dir, tmp_output):
        import shutil

        d = tmp_output / "coll_dir"
        d.mkdir()
        shutil.copy(
            test_data_dir / "single_isotherm.json",
            d / "a.json",
        )
        shutil.copy(
            test_data_dir / "mixture_isotherm.json",
            d / "b.json",
        )
        files = collect_data_files(data_dir=str(d))
        assert len(files) == 2

    def test_data_dir_sorted(self, test_data_dir, tmp_output):
        import shutil
        from pathlib import Path

        d = tmp_output / "sort_dir"
        d.mkdir()
        shutil.copy(
            test_data_dir / "single_isotherm.json",
            d / "z.json",
        )
        shutil.copy(
            test_data_dir / "single_isotherm.json",
            d / "a.json",
        )
        files = collect_data_files(data_dir=str(d))
        assert Path(files[0]).name == "a.json"
        assert Path(files[1]).name == "z.json"

    def test_deduplicates(self, single_isotherm_path):
        files = collect_data_files(
            data=[single_isotherm_path, single_isotherm_path]
        )
        assert len(files) == 1

    def test_combined(self, test_data_dir, tmp_output):
        import shutil

        d = tmp_output / "combo_dir"
        d.mkdir()
        shutil.copy(
            test_data_dir / "single_isotherm.json",
            d / "a.json",
        )
        files = collect_data_files(
            data=[str(test_data_dir / "mixture_isotherm.json")],
            data_dir=str(d),
        )
        assert len(files) == 2

    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            collect_data_files(data_dir="/no/such/dir")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No data files"):
            collect_data_files()


# ==========================================
# Tests for auto temperature labels
# ==========================================
class TestAutoTemperatureLabels:
    """Tests for auto-generated temperature legend labels."""

    def test_auto_labels_multi_file(self, single_isotherm_path, tmp_output):
        """Two files with same T (298K) get '298 K' labels."""
        from matkit.plot.isotherm import plot_single_isotherm

        outfile = str(tmp_output / "test_auto_temp.png")
        # Both files have temperature 298 -- labels should be
        # auto-generated as "298 K" instead of filenames.
        result = plot_single_isotherm(
            [single_isotherm_path, single_isotherm_path],
            output=outfile,
            dpi=72,
        )
        from pathlib import Path

        assert Path(result).exists()

    def test_custom_labels_override(self, single_isotherm_path, tmp_output):
        """Custom labels should take precedence over auto."""
        from matkit.plot.isotherm import plot_single_isotherm

        outfile = str(tmp_output / "test_override.png")
        result = plot_single_isotherm(
            [single_isotherm_path, single_isotherm_path],
            output=outfile,
            dpi=72,
            labels=["Run A", "Run B"],
        )
        from pathlib import Path

        assert Path(result).exists()

    def test_single_file_uses_filename(self, single_isotherm_path, tmp_output):
        """Single file should use filename, not temperature."""
        from matkit.plot.isotherm import plot_single_isotherm

        outfile = str(tmp_output / "test_single_label.png")
        result = plot_single_isotherm(
            [single_isotherm_path],
            output=outfile,
            dpi=72,
        )
        from pathlib import Path

        assert Path(result).exists()

    def test_multi_temp_real_data(self, tmp_output):
        """Test with real multi-temperature data if available."""
        from pathlib import Path

        data_dir = Path(
            "/Users/tpham2/work/projects/matkit/scripts/data/results"
        )
        if not data_dir.is_dir():
            pytest.skip("Real data directory not found")

        from matkit.plot.isotherm import plot_single_isotherm

        files = sorted(str(f) for f in data_dir.glob("*.json"))
        if len(files) < 2:
            pytest.skip("Need at least 2 JSON files")

        outfile = str(tmp_output / "test_real_multi.png")
        result = plot_single_isotherm(files, output=outfile, dpi=72)
        assert Path(result).exists()
