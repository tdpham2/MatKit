"""Legacy MACE imports and optimization without MACE or GPU dependencies."""

import subprocess
import sys
from textwrap import dedent

import numpy as np
import pytest
from ase.build import bulk
from ase.io import read, write


_MACE_STUB = """
import sys
from types import ModuleType
from ase.calculators.emt import EMT

mace = ModuleType("mace")
mace.__path__ = []
calculators = ModuleType("mace.calculators")
calculators.mace_mp = lambda **kwargs: EMT()
mace.calculators = calculators
sys.modules["mace"] = mace
sys.modules["mace.calculators"] = calculators
"""


def _run_isolated(script, *args, stub_mace=True):
    # Fresh processes exercise package exports without polluting other tests.
    code = (dedent(_MACE_STUB) if stub_mace else "") + dedent(script)
    result = subprocess.run(
        [sys.executable, "-c", code, *map(str, args)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("layout", ["modern", "legacy"])
def test_public_mace_import_supports_both_ase_layouts(layout):
    _run_isolated(
        """
        import builtins
        from types import SimpleNamespace

        try:
            from ase.filters import ExpCellFilter
        except ImportError:
            from ase.constraints import ExpCellFilter

        layout = sys.argv[1]
        original_import = builtins.__import__

        def ase_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in {"ase.filters", "ase.constraints"} and (
                "ExpCellFilter" in (fromlist or ())
            ):
                available = (
                    "ase.filters" if layout == "modern" else "ase.constraints"
                )
                if name != available:
                    raise ImportError(f"ExpCellFilter is unavailable in {name}")
                return SimpleNamespace(ExpCellFilter=ExpCellFilter)
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = ase_import
        from matkit.mlip import run_opt_mace
        from matkit.mlip import mace_opt
        import matkit.mlip

        assert run_opt_mace is mace_opt.run_opt_mace
        assert "run_opt_mace" in matkit.mlip.__all__
        assert mace_opt.ExpCellFilter is ExpCellFilter
        """,
        layout,
    )


def test_core_mlip_import_remains_available_without_mace():
    _run_isolated(
        """
        import sys

        sys.modules["mace"] = None
        sys.modules["mace.calculators"] = None
        import matkit.mlip

        assert callable(matkit.mlip.run_mlip)
        assert callable(matkit.mlip.run_mlip_batch)
        assert not hasattr(matkit.mlip, "run_opt_mace")
        assert "run_opt_mace" not in matkit.mlip.__all__
        """,
        stub_mace=False,
    )


@pytest.fixture
def copper(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.8, cubic=True)
    atoms.positions[0, 0] += 0.05
    source = tmp_path / "copper.extxyz"
    write(source, atoms)
    return source, atoms


@pytest.mark.parametrize(
    "run_type", ["geo_opt", "cell_opt", "geo_opt_cell_opt"]
)
def test_legacy_mace_optimization_writes_valid_geometry(
    copper, tmp_path, run_type
):
    source, original = copper
    output = tmp_path / "optimized.extxyz"
    _run_isolated(
        """
        from matkit.mlip import run_opt_mace

        source, output, run_type = sys.argv[1:]
        run_opt_mace(
            source, run_type=run_type, steps=2, fmax=0.01,
            dispersion=False, output_fname=output,
        )
        """,
        source,
        output,
        run_type,
    )
    final = read(output)
    np.testing.assert_array_equal(final.numbers, original.numbers)
    assert np.isfinite(final.positions).all()
    assert np.isfinite(final.cell.array).all()
    assert np.isfinite(final.get_potential_energy())
    if run_type == "geo_opt":
        np.testing.assert_allclose(final.cell.array, original.cell.array)
    else:
        assert not np.allclose(final.cell.array, original.cell.array)


def test_legacy_mace_cli_creates_output(copper, tmp_path):
    source, _ = copper
    output = tmp_path / "cli.extxyz"
    _run_isolated(
        """
        from click.testing import CliRunner
        from matkit.cli import main

        result = CliRunner().invoke(main, [
            "mlip", "mace-opt", "--fname", sys.argv[1],
            "--output", sys.argv[2], "--run-type", "cell_opt",
            "--steps", "2", "--no-dispersion",
        ])
        assert result.exit_code == 0, result.output
        assert "MACE-MP optimization complete." in result.output, result.output
        """,
        source,
        output,
    )
    assert np.isfinite(read(output).positions).all()
