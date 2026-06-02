"""Tests for matkit.graspa module."""

import json
import shutil

import pytest
from pathlib import Path

from matkit.graspa.graspa import (
    generate_component_blocks,
    setup_batch,
    setup_simulation,
)
from matkit.utils.cif import sanitize_cif_stem


class TestGenerateComponentBlocks:
    """Tests for gRASPA component block generation."""

    def test_single_adsorbate(self):
        """Should generate block for a single adsorbate."""
        adsorbates = [{"MoleculeName": "CO2"}]
        result = generate_component_blocks(adsorbates)
        assert "Component 0 MoleculeName" in result
        assert "CO2" in result

    def test_multiple_adsorbates(self):
        """Should generate blocks for multiple adsorbates."""
        adsorbates = [
            {"MoleculeName": "CO2"},
            {"MoleculeName": "N2"},
        ]
        result = generate_component_blocks(adsorbates)
        assert "Component 0 MoleculeName" in result
        assert "Component 1 MoleculeName" in result
        assert "CO2" in result
        assert "N2" in result

    def test_default_params_included(self):
        """Default parameters should be included in output."""
        adsorbates = [{"MoleculeName": "CO2"}]
        result = generate_component_blocks(adsorbates)
        assert "TranslationProbability" in result
        assert "RotationProbability" in result
        assert "SwapProbability" in result

    def test_custom_params_override_defaults(self):
        """Custom parameters should override defaults."""
        adsorbates = [
            {
                "MoleculeName": "CO2",
                "SwapProbability": 5.0,
            }
        ]
        result = generate_component_blocks(adsorbates)
        assert "5.0" in result

    def test_missing_molecule_name_raises(self):
        """Should raise ValueError if MoleculeName is missing."""
        adsorbates = [{"SwapProbability": 2.0}]
        with pytest.raises(ValueError, match="MoleculeName"):
            generate_component_blocks(adsorbates)

    def test_none_param_excluded(self):
        """Setting a param to None should exclude it."""
        adsorbates = [
            {
                "MoleculeName": "CO2",
                "RotationProbability": None,
            }
        ]
        result = generate_component_blocks(adsorbates)
        assert "RotationProbability" not in result


class TestSetupSimulation:
    """Tests for gRASPA simulation setup."""

    def test_setup_creates_output(self, sample_cif, tmp_path):
        """Should create output directory with simulation files."""
        outdir = tmp_path / "sim_output"
        adsorbates = [{"MoleculeName": "CO2"}]
        result = setup_simulation(
            cif=sample_cif,
            outpath=str(outdir),
            adsorbates=adsorbates,
        )
        assert result is True
        assert outdir.exists()
        assert (outdir / "simulation.input").exists()

    def test_setup_copies_cif(self, sample_cif, tmp_path):
        """Should copy CIF file to output directory."""
        outdir = tmp_path / "sim_output"
        adsorbates = [{"MoleculeName": "CO2"}]
        setup_simulation(
            cif=sample_cif,
            outpath=str(outdir),
            adsorbates=adsorbates,
        )
        cif_name = Path(sample_cif).stem + ".cif"
        assert (outdir / cif_name).exists()

    def test_setup_replaces_placeholders(self, sample_cif, tmp_path):
        """Should replace placeholders in simulation.input."""
        outdir = tmp_path / "sim_output"
        adsorbates = [{"MoleculeName": "CO2"}]
        setup_simulation(
            cif=sample_cif,
            outpath=str(outdir),
            adsorbates=adsorbates,
            temperature=300.0,
            pressure=2e5,
            n_cycle=500,
        )
        content = (outdir / "simulation.input").read_text()
        assert "300.0" in content
        assert "200000.0" in content
        assert "500" in content
        # Placeholders should be gone
        assert "NCYCLE" not in content
        assert "TEMPERATURE" not in content

    def test_setup_nonexistent_cif_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing CIF."""
        with pytest.raises(FileNotFoundError):
            setup_simulation(
                cif="/nonexistent/file.cif",
                outpath=str(tmp_path / "out"),
                adsorbates=[{"MoleculeName": "CO2"}],
            )

    def test_setup_sanitizes_multi_period_cif(self, sample_cif, tmp_path):
        """CIFs with extra periods should be copied under a safe name."""
        weird_cif = tmp_path / "foo.bar.cif"
        shutil.copy(sample_cif, weird_cif)
        outdir = tmp_path / "sim_output"

        setup_simulation(
            cif=str(weird_cif),
            outpath=str(outdir),
            adsorbates=[{"MoleculeName": "CO2"}],
        )

        assert (outdir / "foo_bar.cif").exists()
        assert not (outdir / "foo.bar.cif").exists()
        content = (outdir / "simulation.input").read_text()
        assert "FrameworkName foo_bar" in content


class TestSanitizeCifStem:
    """Tests for the sanitize_cif_stem helper."""

    def test_no_period_unchanged(self):
        assert sanitize_cif_stem("MOF5") == "MOF5"

    def test_single_internal_period_replaced(self):
        assert sanitize_cif_stem("foo.bar") == "foo_bar"

    def test_multiple_periods_replaced(self):
        assert (
            sanitize_cif_stem("str_m5_o11_o18_sra_sym.22")
            == "str_m5_o11_o18_sra_sym_22"
        )


class TestSetupBatch:
    """Tests for gRASPA batch setup, focusing on CIF rename mapping."""

    def test_batch_writes_mapping_only_for_renamed(
        self, sample_cif, tmp_path
    ):
        """cif_mapping.json should list only CIFs that needed renaming."""
        cif_dir = tmp_path / "cifs"
        cif_dir.mkdir()
        shutil.copy(sample_cif, cif_dir / "clean.cif")
        shutil.copy(sample_cif, cif_dir / "weird.v2.cif")

        out_dir = tmp_path / "batch_out"
        manifest = setup_batch(
            cif_dir=str(cif_dir),
            outpath=str(out_dir),
            adsorbates=[{"MoleculeName": "CO2"}],
            temperatures=[298.0],
            pressures=[1e5],
            n_cycle=10,
        )

        assert len(manifest) == 2
        mapping_path = out_dir / "cif_mapping.json"
        assert mapping_path.exists()
        mapping = json.loads(mapping_path.read_text())
        assert mapping == {"weird.v2": "weird_v2"}

        assert (out_dir / "weird_v2" / "T298.0_P100000" / "weird_v2.cif").exists()
        assert (out_dir / "clean" / "T298.0_P100000" / "clean.cif").exists()

    def test_batch_no_mapping_when_all_clean(self, sample_cif, tmp_path):
        """cif_mapping.json should not be written if no rename occurred."""
        cif_dir = tmp_path / "cifs"
        cif_dir.mkdir()
        shutil.copy(sample_cif, cif_dir / "clean.cif")

        out_dir = tmp_path / "batch_out"
        setup_batch(
            cif_dir=str(cif_dir),
            outpath=str(out_dir),
            adsorbates=[{"MoleculeName": "CO2"}],
            temperatures=[298.0],
            pressures=[1e5],
            n_cycle=10,
        )

        assert not (out_dir / "cif_mapping.json").exists()
