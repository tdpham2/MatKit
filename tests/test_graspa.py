"""Tests for matkit.graspa module."""

import pytest
from pathlib import Path

from matkit.graspa.graspa import (
    generate_component_blocks,
    setup_simulation,
)


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
