"""Scientific regressions shared by the GCMC setup implementations."""

import importlib
import re

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from matkit.raspa2 import get_output_data
from matkit.utils import calculate_cell_size


@pytest.mark.parametrize(
    "engine", ["graspa", "graspa_sycl", "raspa2", "pygraspa"]
)
@pytest.mark.parametrize("cutoff", [4.0, 18.0])
def test_setup_uses_requested_cutoff(engine, cutoff, tmp_path):
    atoms = Atoms("Si", cell=[[10, 0, 0], [3, 9, 0], [1, 2, 8]], pbc=True)
    cif = tmp_path / "input.cif"
    write(cif, atoms)
    module = importlib.import_module(f"matkit.{engine}")
    out = tmp_path / "sim"
    if engine == "raspa2":
        module.setup_input_simulation([str(cif)], str(out), cutoff=cutoff)
        out = out / "input"
    elif engine == "graspa_sycl":
        module.setup_simulation(str(cif), str(out), cutoff=cutoff)
    else:
        kwargs = {}
        if engine == "pygraspa":
            kwargs = dict(
                model_path="/model.pt",
                model_type="FAIRChem-esen",
                E_comps=[-1.0],
            )
        module.setup_simulation(
            str(cif),
            str(out),
            [{"MoleculeName": "CO2"}],
            cutoff=cutoff,
            **kwargs,
        )
    text = (out / "simulation.input").read_text()
    match = re.search(
        r"^UnitCells\s+(?:0\s+)?(\d+)\s+(\d+)\s+(\d+)$", text, re.M
    )
    sizes = [int(n) for n in match.groups()]
    assert sizes == calculate_cell_size(atoms, cutoff)
    cell = atoms.cell.array
    heights = [
        abs(np.linalg.det(cell))
        / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
        for i in range(3)
    ]
    assert all(n * h >= 2 * cutoff for n, h in zip(sizes, heights))


@pytest.mark.parametrize("engine", ["graspa", "pygraspa"])
def test_batch_cached_replication_uses_cutoff(engine, tmp_path):
    module = importlib.import_module(f"matkit.{engine}")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    write(inputs / "input.cif", Atoms("Si", cell=[10, 10, 10], pbc=True))
    kwargs = {}
    if engine == "pygraspa":
        kwargs = dict(
            model_path="/model.pt", model_type="FAIRChem-esen", E_comps=[-1.0]
        )
    module.setup_batch(
        str(inputs),
        str(tmp_path / "out"),
        [{"MoleculeName": "CO2"}],
        temperatures=[298],
        pressures=[1e4, 1e5],
        cutoff=18,
        **kwargs,
    )
    files = list((tmp_path / "out").rglob("simulation.input"))
    assert len(files) == 2
    assert all("UnitCells 0 4 4 4" in f.read_text() for f in files)


@pytest.mark.parametrize("unit,expected", [("mol/kg", 2), ("g/L", 44)])
def test_raspa2_valid_result_is_successful(tmp_path, unit, expected):
    log = tmp_path / "raspa.log"
    log.write_text(
        "Average loading absolute [mol/kg framework] 2 +/- 0.1\n"
        "Average loading absolute [milligram/gram framework] 88 +/- 4.4\n"
        "Framework Density 500\n"
    )
    result = get_output_data(str(log), unit=unit)
    assert result["success"] is True
    assert result["uptake"] == expected


def test_raspa2_incomplete_result_fails(tmp_path):
    log = tmp_path / "bad.log"
    log.write_text("incomplete\n")
    with pytest.raises(ValueError, match="expected lines"):
        get_output_data(str(log))


@pytest.mark.parametrize("cutoff", [True, 0, -1, float("inf"), float("nan")])
def test_invalid_cutoff_rejected(cutoff):
    with pytest.raises(ValueError, match="positive and finite"):
        calculate_cell_size(Atoms("Si", cell=[10] * 3), cutoff)


def test_singular_replication_cell_rejected():
    with pytest.raises(ValueError, match="independent"):
        calculate_cell_size(Atoms("Si", cell=[10, 10, 0]))
