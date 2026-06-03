"""Tests for matkit.pygraspa module."""

import json
import shutil
from pathlib import Path

import pytest

from matkit.pygraspa.pygraspa import (
    _render_run_py,
    get_output_data,
    setup_batch,
    setup_simulation,
)


class TestRenderRunPy:
    """Verify the per-simulation run.py launcher is rendered correctly."""

    def test_run_mode(self):
        out = _render_run_py(
            mode="run",
            model_path="/m/model.pt",
            model_type="FAIRChem-esen",
            task=None,
            E_comps=[-22.97915],
            save_poscar=False,
        )
        assert "from pygRASPA.main import run" in out
        assert "run(" in out
        assert "/m/model.pt" in out
        assert "FAIRChem-esen" in out
        assert "-22.97915" in out
        assert "task=None" in out
        assert "save_trial_poscar=False" in out
        assert 'log_file_path="output.log"' in out
        assert "run_auto" not in out

    def test_run_auto_mode(self):
        out = _render_run_py(
            mode="run-auto",
            model_path="/m/model.pt",
            model_type="FAIRChem-uma",
            task="ads_eng",
            E_comps=[-22.0, -17.5],
            save_poscar=True,
        )
        assert "from pygRASPA.auto.main import run_auto" in out
        assert "run_auto(" in out
        assert "'ads_eng'" in out
        assert "[-22.0, -17.5]" in out
        assert "save_trial_poscar=True" in out
        assert 'checkpoint_file="checkpoint.json"' in out
        assert 'stop_flag="STOP_FLAG"' in out

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            _render_run_py(
                mode="invalid",
                model_path="x",
                model_type="FAIRChem-esen",
                task=None,
                E_comps=[1.0],
                save_poscar=False,
            )


class TestSetupSimulation:
    """End-to-end setup writes simulation.input, run.py, and CIF."""

    def _run(self, cif, outdir, **overrides):
        defaults = dict(
            cif=cif,
            outpath=str(outdir),
            adsorbates=[{"MoleculeName": "CO2"}],
            model_path="/path/to/model.pt",
            model_type="FAIRChem-esen",
            E_comps=[-22.97915],
        )
        defaults.update(overrides)
        return setup_simulation(**defaults)

    def test_creates_simulation_input(self, sample_cif, tmp_path):
        out = tmp_path / "sim"
        self._run(sample_cif, out, temperature=300.0, pressure=2e5, n_cycle=500)
        assert (out / "simulation.input").exists()
        content = (out / "simulation.input").read_text()
        assert "300.0" in content
        assert "200000.0" in content
        assert "500" in content
        assert "NCYCLE" not in content
        assert "TEMPERATURE" not in content
        assert "__COMPONENTS__" not in content

    def test_writes_run_py_with_baked_args(self, sample_cif, tmp_path):
        out = tmp_path / "sim"
        self._run(sample_cif, out, mode="run-auto")
        run_py = (out / "run.py").read_text()
        assert "/path/to/model.pt" in run_py
        assert "FAIRChem-esen" in run_py
        assert "-22.97915" in run_py
        assert "from pygRASPA.auto.main import run_auto" in run_py

    def test_run_py_is_valid_python(self, sample_cif, tmp_path):
        out = tmp_path / "sim"
        self._run(sample_cif, out)
        compile((out / "run.py").read_text(), str(out / "run.py"), "exec")

    def test_copies_cif(self, sample_cif, tmp_path):
        out = tmp_path / "sim"
        self._run(sample_cif, out)
        assert (out / (Path(sample_cif).stem + ".cif")).exists()

    def test_mismatched_ecomps_raises(self, sample_cif, tmp_path):
        with pytest.raises(ValueError, match="same length"):
            self._run(
                sample_cif,
                tmp_path / "sim",
                adsorbates=[{"MoleculeName": "CO2"}, {"MoleculeName": "N2"}],
                E_comps=[-22.0],
            )

    def test_uma_requires_task(self, sample_cif, tmp_path):
        with pytest.raises(ValueError, match="task is required"):
            self._run(
                sample_cif,
                tmp_path / "sim",
                model_type="FAIRChem-uma",
                task=None,
            )

    def test_missing_cif_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            self._run("/nonexistent.cif", tmp_path / "sim")

    def test_sanitizes_multi_period_cif(self, sample_cif, tmp_path):
        weird = tmp_path / "foo.bar.cif"
        shutil.copy(sample_cif, weird)
        out = tmp_path / "sim"
        self._run(str(weird), out)
        assert (out / "foo_bar.cif").exists()
        content = (out / "simulation.input").read_text()
        assert "FrameworkName foo_bar" in content


class TestSetupBatch:
    """Batch setup mirrors graspa.setup_batch structure."""

    def test_batch_manifest_and_rename(self, sample_cif, tmp_path):
        cif_dir = tmp_path / "cifs"
        cif_dir.mkdir()
        shutil.copy(sample_cif, cif_dir / "clean.cif")
        shutil.copy(sample_cif, cif_dir / "weird.v2.cif")

        out_dir = tmp_path / "batch"
        manifest = setup_batch(
            cif_dir=str(cif_dir),
            outpath=str(out_dir),
            adsorbates=[{"MoleculeName": "CO2"}],
            model_path="/m.pt",
            model_type="FAIRChem-esen",
            E_comps=[-22.97915],
            temperatures=[298.0],
            pressures=[1e5],
            n_cycle=10,
        )

        assert len(manifest) == 2
        assert (out_dir / "simulations.jsonl").exists()
        assert (out_dir / "cif_mapping.json").exists()

        # Each sim dir should have its own run.py
        for entry in manifest:
            assert (Path(entry["sim_dir"]) / "run.py").exists()
            assert (Path(entry["sim_dir"]) / "simulation.input").exists()
            assert entry["model_path"] == "/m.pt"
            assert entry["model_type"] == "FAIRChem-esen"
            assert entry["mode"] == "run-auto"

        mapping = json.loads((out_dir / "cif_mapping.json").read_text())
        assert mapping == {"weird.v2": "weird_v2"}


class TestGetOutputData:
    """JSON-lines log parser averages production-cycle counts."""

    def _write_log(self, dir_, n_init, n_total, n_comp):
        """Write a fake JSON-lines log: N=10*cycle for each cycle."""
        lines = []
        for c in range(1, n_total + 1):
            counts = [10.0 * c + i for i in range(n_comp)]
            lines.append(json.dumps({"cycle": c, "molecules": counts}))
        (dir_ / "output.log").write_text("\n".join(lines) + "\n")

    def _write_sim_input(self, dir_, n_init, n_prod, ux=2, uy=2, uz=2):
        (dir_ / "simulation.input").write_text(
            f"NumberOfInitializationCycles {n_init}\n"
            "NumberOfEquilibrationCycles 0\n"
            f"NumberOfProductionCycles {n_prod}\n"
            f"UnitCells 0 {ux} {uy} {uz}\n"
            "Component 0 MoleculeName CO2\n"
        )

    def _write_cif(self, dir_, sample_cif):
        shutil.copy(sample_cif, dir_ / "frame.cif")

    def test_parses_single_component(self, sample_cif, tmp_path):
        d = tmp_path / "out"
        d.mkdir()
        self._write_log(d, n_init=2, n_total=5, n_comp=1)
        self._write_sim_input(d, n_init=2, n_prod=3)
        self._write_cif(d, sample_cif)

        result = get_output_data(str(d), unit="mol/kg")
        assert result["success"] is True
        assert result["n_components"] == 1
        assert result["n_production_cycles"] == 4  # cycles 2,3,4,5
        assert isinstance(result["uptake"], float)
        assert result["unit"] == "mol/kg"

    def test_parses_mixture(self, sample_cif, tmp_path):
        d = tmp_path / "out"
        d.mkdir()
        self._write_log(d, n_init=1, n_total=4, n_comp=2)
        self._write_sim_input(d, n_init=1, n_prod=3)
        # Need both components in input for mol/kg path (mg/g would need names)
        (d / "simulation.input").write_text(
            "NumberOfInitializationCycles 1\n"
            "NumberOfEquilibrationCycles 0\n"
            "NumberOfProductionCycles 3\n"
            "UnitCells 0 1 1 1\n"
            "Component 0 MoleculeName CO2\n"
            "Component 1 MoleculeName N2\n"
        )
        self._write_cif(d, sample_cif)

        result = get_output_data(str(d), unit="mol/kg")
        assert result["n_components"] == 2
        assert isinstance(result["uptake"], list)
        assert len(result["uptake"]) == 2

    def test_skip_cycles_override(self, sample_cif, tmp_path):
        d = tmp_path / "out"
        d.mkdir()
        self._write_log(d, n_init=0, n_total=5, n_comp=1)
        self._write_sim_input(d, n_init=0, n_prod=5)
        self._write_cif(d, sample_cif)

        full = get_output_data(str(d), n_skip_cycles=0)
        partial = get_output_data(str(d), n_skip_cycles=4)
        # With strictly increasing counts, dropping early cycles raises mean.
        assert partial["uptake"] > full["uptake"]

    def test_missing_log_raises(self, tmp_path):
        d = tmp_path / "out"
        d.mkdir()
        with pytest.raises(ValueError, match="Output log not found"):
            get_output_data(str(d))

    def test_unparseable_log_raises(self, tmp_path):
        d = tmp_path / "out"
        d.mkdir()
        (d / "output.log").write_text("not json\nalso not json\n")
        with pytest.raises(ValueError, match="No parseable cycle records"):
            get_output_data(str(d))
