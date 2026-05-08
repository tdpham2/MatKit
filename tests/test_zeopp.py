"""Tests for matkit.zeopp module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import shutil

from matkit.zeopp.zeopp import (
    _find_network_binary,
    _parse_res,
    _parse_sa,
    _parse_vol,
    _parse_psd,
    _parse_chan,
    get_output_data,
    run_zeopp,
)


@pytest.fixture
def zeopp_data_dir(test_data_dir):
    """Path to the zeopp test data directory."""
    return test_data_dir / "zeopp"


class TestParseRes:
    """Tests for .res file parsing."""

    def test_parse_valid_res(self, zeopp_data_dir):
        """Should extract Di, Df, Dif from .res file."""
        result = _parse_res(zeopp_data_dir / "test_structure.res")
        assert result["Di"] == pytest.approx(18.569)
        assert result["Df"] == pytest.approx(8.023)
        assert result["Dif"] == pytest.approx(10.928)
        assert result["unit"] == "Angstrom"

    def test_parse_res_missing_file(self, tmp_path):
        """Should raise ValueError for missing file."""
        with pytest.raises(ValueError, match="Failed to parse .res"):
            _parse_res(tmp_path / "nonexistent.res")


class TestParseSa:
    """Tests for .sa file parsing."""

    def test_parse_valid_sa(self, zeopp_data_dir):
        """Should extract ASA fields from .sa file."""
        result = _parse_sa(zeopp_data_dir / "test_structure.sa")
        assert result["unitcell_volume"] == pytest.approx(17237.8)
        assert result["density"] == pytest.approx(0.593116)
        assert result["ASA"] == pytest.approx(4004.75)
        assert result["ASA_m2_cm3"] == pytest.approx(2323.62)
        assert result["ASA_m2_g"] == pytest.approx(3918.33)
        assert result["NASA"] == pytest.approx(0.0)
        assert result["NASA_m2_cm3"] == pytest.approx(0.0)
        assert result["NASA_m2_g"] == pytest.approx(0.0)


class TestParseVol:
    """Tests for .vol file parsing."""

    def test_parse_valid_vol(self, zeopp_data_dir):
        """Should extract AV fields from .vol file."""
        result = _parse_vol(zeopp_data_dir / "test_structure.vol")
        assert result["unitcell_volume"] == pytest.approx(17237.8)
        assert result["density"] == pytest.approx(0.593116)
        assert result["AV"] == pytest.approx(12456.3)
        assert result["AV_volume_fraction"] == pytest.approx(0.72255)
        assert result["AV_cm3_g"] == pytest.approx(1.219)
        assert result["NAV"] == pytest.approx(4781.5)
        assert result["NAV_volume_fraction"] == pytest.approx(0.27745)
        assert result["NAV_cm3_g"] == pytest.approx(0.468)


class TestParseChan:
    """Tests for .chan file parsing."""

    def test_parse_valid_chan(self, zeopp_data_dir):
        """Should extract channel info from .chan file."""
        result = _parse_chan(zeopp_data_dir / "test_structure.chan")
        assert result["num_channels"] == 2
        assert result["dimensionalities"] == [3, 3]


class TestParsePsd:
    """Tests for .psd file parsing."""

    def test_parse_valid_psd(self, zeopp_data_dir):
        """Should extract histogram data from .psd file."""
        result = _parse_psd(zeopp_data_dir / "test_structure.psd")
        assert len(result["bin_lower"]) == 8
        assert len(result["counts"]) == 8
        assert result["bin_lower"][0] == pytest.approx(0.1)
        assert result["counts"][3] == pytest.approx(25.0)
        assert result["bin_size"] == pytest.approx(0.1)


class TestFindNetworkBinary:
    """Tests for network binary discovery."""

    def test_explicit_path_exists(self, tmp_path):
        """Should return path when binary exists at given location."""
        fake_binary = tmp_path / "network"
        fake_binary.touch()
        result = _find_network_binary(str(fake_binary))
        assert result == str(fake_binary)

    def test_explicit_path_missing(self):
        """Should raise FileNotFoundError for missing binary."""
        with pytest.raises(FileNotFoundError, match="not found at"):
            _find_network_binary("/nonexistent/network")

    @patch("matkit.zeopp.zeopp.shutil.which", return_value=None)
    def test_not_on_path(self, mock_which):
        """Should raise FileNotFoundError when not on PATH."""
        with pytest.raises(FileNotFoundError, match="not found on PATH"):
            _find_network_binary(None)

    @patch("matkit.zeopp.zeopp.shutil.which", return_value="/usr/bin/network")
    def test_found_on_path(self, mock_which):
        """Should return path when found on PATH."""
        result = _find_network_binary(None)
        assert result == "/usr/bin/network"


class TestGetOutputData:
    """Tests for parsing pre-existing Zeo++ output files."""

    def test_auto_detect_analyses(self, zeopp_data_dir):
        """Should auto-detect available output files."""
        result = get_output_data(str(zeopp_data_dir))
        assert result["success"] is True
        assert "res" in result
        assert "sa" in result
        assert "vol" in result
        assert "chan" in result

    def test_parse_specific_analysis(self, zeopp_data_dir):
        """Should parse only requested analysis types."""
        result = get_output_data(str(zeopp_data_dir), analyses=["res"])
        assert result["success"] is True
        assert "res" in result
        assert result["res"]["Di"] == pytest.approx(18.569)

    def test_single_file_mode(self, zeopp_data_dir):
        """Should parse a single output file directly."""
        result = get_output_data(
            str(zeopp_data_dir / "test_structure.res")
        )
        assert result["success"] is True
        assert "res" in result

    def test_nonexistent_path_raises(self):
        """Should raise FileNotFoundError for missing path."""
        with pytest.raises(FileNotFoundError):
            get_output_data("/nonexistent/path")

    def test_invalid_analysis_raises(self, zeopp_data_dir):
        """Should raise ValueError for invalid analysis type."""
        with pytest.raises(ValueError, match="Invalid analysis"):
            get_output_data(str(zeopp_data_dir), analyses=["invalid"])


class TestRunZeopp:
    """Tests for running the Zeo++ network binary."""

    def test_missing_cif_raises(self):
        """Should raise FileNotFoundError for missing CIF."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            run_zeopp("/nonexistent/file.cif")

    def test_invalid_analysis_raises(self, sample_cif):
        """Should raise ValueError for invalid analysis type."""
        with pytest.raises(ValueError, match="Invalid analysis"):
            run_zeopp(sample_cif, analyses=["invalid"])

    @patch("matkit.zeopp.zeopp._find_network_binary")
    @patch("matkit.zeopp.zeopp.subprocess.run")
    def test_run_res_analysis(self, mock_run, mock_find, sample_cif,
                              zeopp_data_dir, tmp_path):
        """Should run network binary and parse .res output."""
        mock_find.return_value = "/usr/bin/network"

        # Mock subprocess to copy sample output to working dir
        def side_effect(cmd, **kwargs):
            # The CIF path in the command is the last argument
            cif_path = Path(cmd[-1])
            stem = cif_path.stem
            workdir = cif_path.parent
            shutil.copy(
                zeopp_data_dir / "test_structure.res",
                workdir / f"{stem}.res",
            )
            return MagicMock(returncode=0, stderr="")

        mock_run.side_effect = side_effect

        result = run_zeopp(
            sample_cif,
            analyses=["res"],
            output_dir=str(tmp_path / "out"),
        )
        assert result["success"] is True
        assert "res" in result["results"]
        assert result["results"]["res"]["Di"] == pytest.approx(18.569)

    @patch("matkit.zeopp.zeopp._find_network_binary")
    @patch("matkit.zeopp.zeopp.subprocess.run")
    def test_run_includes_ha_flag(self, mock_run, mock_find,
                                  sample_cif, tmp_path):
        """Should include -ha flag by default."""
        mock_find.return_value = "/usr/bin/network"
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        run_zeopp(sample_cif, analyses=["res"],
                  output_dir=str(tmp_path / "out"))
        cmd = mock_run.call_args[0][0]
        assert "-ha" in cmd

    @patch("matkit.zeopp.zeopp._find_network_binary")
    @patch("matkit.zeopp.zeopp.subprocess.run")
    def test_run_uses_bundled_radii_by_default(
        self, mock_run, mock_find, sample_cif, tmp_path,
    ):
        """Should use bundled UFF.rad when no radii file given."""
        mock_find.return_value = "/usr/bin/network"
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        outdir = tmp_path / "out"
        run_zeopp(sample_cif, analyses=["res"],
                  output_dir=str(outdir))
        cmd = mock_run.call_args[0][0]
        assert "-r" in cmd
        assert (outdir / "UFF.rad").exists()

    @patch("matkit.zeopp.zeopp._find_network_binary")
    @patch("matkit.zeopp.zeopp.subprocess.run")
    def test_run_no_ha_flag(self, mock_run, mock_find, sample_cif, tmp_path):
        """Should omit -ha flag when ha=False."""
        mock_find.return_value = "/usr/bin/network"
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        run_zeopp(sample_cif, analyses=["res"], ha=False,
                  output_dir=str(tmp_path / "out"))
        cmd = mock_run.call_args[0][0]
        assert "-ha" not in cmd

    @patch("matkit.zeopp.zeopp._find_network_binary")
    @patch("matkit.zeopp.zeopp.subprocess.run")
    def test_run_with_radii_file(self, mock_run, mock_find, sample_cif,
                                 tmp_path):
        """Should include -r flag and copy radii file to workdir."""
        mock_find.return_value = "/usr/bin/network"
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        # Create a fake radii file
        rad_file = tmp_path / "UFF.rad"
        rad_file.write_text("H 1.0\nC 1.7\n")

        outdir = tmp_path / "out"
        run_zeopp(sample_cif, analyses=["res"],
                  radii_file=str(rad_file),
                  output_dir=str(outdir))

        cmd = mock_run.call_args[0][0]
        assert "-r" in cmd
        # Radii file should be copied to workdir
        assert (outdir / "UFF.rad").exists()

    def test_run_missing_radii_file_raises(self, sample_cif):
        """Should raise FileNotFoundError for missing radii file."""
        with pytest.raises(FileNotFoundError, match="Radii file"):
            run_zeopp(sample_cif, radii_file="/nonexistent/UFF.rad")

    @patch("matkit.zeopp.zeopp._find_network_binary")
    @patch("matkit.zeopp.zeopp.subprocess.run")
    def test_run_failure_raises(self, mock_run, mock_find, sample_cif):
        """Should raise ValueError when network binary fails."""
        mock_find.return_value = "/usr/bin/network"
        mock_run.return_value = MagicMock(
            returncode=1, stderr="Error: bad input"
        )
        with pytest.raises(ValueError, match="network failed"):
            run_zeopp(sample_cif, output_dir="/tmp/zeopp_test_fail")
