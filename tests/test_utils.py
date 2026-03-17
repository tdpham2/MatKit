"""Tests for matkit.utils module."""

import pytest
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.io import write as ase_write

from matkit.utils.unitcell_calculator import calculate_cell_size
from matkit.utils.remove_solvent import remove_solvent
from matkit.utils.cifsampler import sample_cifs


class TestCalculateCellSize:
    """Tests for the unit cell calculator."""

    def test_cubic_cell(self):
        """A 10A cubic cell with 12.8A cutoff should need 3x3x3."""
        atoms = Atoms("Si", positions=[[0, 0, 0]])
        atoms.set_cell([10.0, 10.0, 10.0])
        atoms.set_pbc(True)
        result = calculate_cell_size(atoms, cutoff=12.8)
        assert len(result) == 3
        assert all(isinstance(x, int) for x in result)
        # ceil(12.8 / (0.5 * 10)) = ceil(2.56) = 3
        assert result == [3, 3, 3]

    def test_large_cell_needs_fewer_replications(self):
        """A 30A cell with 12.8A cutoff should need only 1x1x1."""
        atoms = Atoms("Si", positions=[[0, 0, 0]])
        atoms.set_cell([30.0, 30.0, 30.0])
        atoms.set_pbc(True)
        result = calculate_cell_size(atoms, cutoff=12.8)
        assert result == [1, 1, 1]

    def test_rectangular_cell(self):
        """Non-cubic cells should give different replications per axis."""
        atoms = Atoms("Si", positions=[[0, 0, 0]])
        atoms.set_cell([30.0, 10.0, 5.0])
        atoms.set_pbc(True)
        result = calculate_cell_size(atoms, cutoff=12.8)
        assert result[0] < result[1] < result[2]

    def test_custom_cutoff(self):
        """Smaller cutoff should require fewer replications."""
        atoms = Atoms("Si", positions=[[0, 0, 0]])
        atoms.set_cell([10.0, 10.0, 10.0])
        atoms.set_pbc(True)
        small_cutoff = calculate_cell_size(atoms, cutoff=4.0)
        large_cutoff = calculate_cell_size(atoms, cutoff=12.8)
        assert all(s <= l for s, l in zip(small_cutoff, large_cutoff))

    def test_returns_list_of_ints(self):
        """Return type should be a list of integers."""
        atoms = Atoms("Si", positions=[[0, 0, 0]])
        atoms.set_cell([10.0, 10.0, 10.0])
        atoms.set_pbc(True)
        result = calculate_cell_size(atoms)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(x, (int, np.integer)) for x in result)


class TestSampleCifs:
    """Tests for the CIF sampler."""

    def test_sample_cifs_basic(self, tmp_path):
        """Should return requested number of CIF filenames."""
        # Create dummy CIF files
        for i in range(5):
            (tmp_path / f"test_{i}.cif").write_text(f"data_{i}")

        result = sample_cifs(3, str(tmp_path))
        assert len(result) == 3
        assert all(f.endswith(".cif") for f in result)

    def test_sample_cifs_reproducible(self, tmp_path):
        """Same seed should give same results."""
        for i in range(10):
            (tmp_path / f"test_{i}.cif").write_text(f"data_{i}")

        result1 = sample_cifs(5, str(tmp_path), seed=42)
        result2 = sample_cifs(5, str(tmp_path), seed=42)
        assert result1 == result2

    def test_sample_cifs_different_seeds(self, tmp_path):
        """Different seeds should (likely) give different results."""
        for i in range(20):
            (tmp_path / f"test_{i}.cif").write_text(f"data_{i}")

        result1 = sample_cifs(10, str(tmp_path), seed=1)
        result2 = sample_cifs(10, str(tmp_path), seed=2)
        # With 20 files and 10 samples, different seeds should differ
        assert result1 != result2

    def test_sample_cifs_copy(self, tmp_path):
        """When copy=True, files should be copied to output path."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        for i in range(5):
            (src / f"test_{i}.cif").write_text(f"data_{i}")

        result = sample_cifs(3, str(src), copy=True, outpath=str(dst))
        assert len(result) == 3
        assert dst.exists()
        copied_files = list(dst.glob("*.cif"))
        assert len(copied_files) == 3

    def test_sample_cifs_too_many_requested(self, tmp_path):
        """Should raise ValueError if requesting more than available."""
        for i in range(3):
            (tmp_path / f"test_{i}.cif").write_text(f"data_{i}")

        with pytest.raises(ValueError, match="Requested 10"):
            sample_cifs(10, str(tmp_path))

    def test_sample_cifs_nonexistent_dir(self):
        """Should raise FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            sample_cifs(1, "/nonexistent/path")

    def test_sample_cifs_copy_no_outpath(self, tmp_path):
        """Should raise ValueError if copy=True without outpath."""
        (tmp_path / "test.cif").write_text("data")
        with pytest.raises(ValueError, match="output path"):
            sample_cifs(1, str(tmp_path), copy=True)


class TestRemoveSolvent:
    """Tests for solvent removal."""

    def test_remove_solvent_creates_output(self, sample_cif, tmp_path):
        """Should create an output CIF file."""
        output = str(tmp_path / "output.cif")
        remove_solvent(sample_cif, output)
        assert Path(output).exists()
