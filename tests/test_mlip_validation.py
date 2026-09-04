"""Scientific validation and execution outcome regressions."""

from contextlib import contextmanager
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)
from ase.calculators.emt import EMT
from ase.io import write

from matkit.mlip import (
    ASEMACEConfig,
    MLIPCalculationConfig,
    NVAlchemiMACEConfig,
    RootstockConfig,
    run_mlip,
    run_mlip_batch,
)
from matkit.mlip import runner


@pytest.fixture
def copper(tmp_path):
    path = tmp_path / "copper.xyz"
    write(path, Atoms("Cu2", positions=[[0, 0, 0], [3, 0, 0]], cell=[8] * 3))
    return path


def use_calculator(monkeypatch, calculator):
    @contextmanager
    def context(_backend):
        yield calculator

    monkeypatch.setattr(runner, "_ase_backend_context", context)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), True]
)
@pytest.mark.parametrize(
    "field", ["fmax", "dt", "timeout", "dispersion_cutoff"]
)
def test_nonfinite_config_rejected(field, value):
    factory = {
        "fmax": MLIPCalculationConfig,
        "dt": lambda **kw: NVAlchemiMACEConfig("medium", **kw),
        "timeout": lambda **kw: RootstockConfig("medium", **kw),
        "dispersion_cutoff": ASEMACEConfig,
    }[field]
    with pytest.raises(ValueError, match=field):
        factory(**{field: value})


@pytest.mark.parametrize("value", [0, -1, 1.5, True, float("nan")])
def test_integer_limits(copper, tmp_path, value):
    with pytest.raises(ValueError, match="steps"):
        MLIPCalculationConfig(steps=value)
    for field in ("batch_size", "max_atoms"):
        with pytest.raises(ValueError, match=field):
            run_mlip_batch(
                [copper],
                ASEMACEConfig(),
                output_dir=tmp_path / field,
                **{field: value},
            )


def test_rootstock_kwargs_must_be_serializable():
    for value in (float("nan"), object()):
        with pytest.raises(ValueError, match="finite JSON"):
            RootstockConfig("medium", setup_kwargs={"nested": [value]})


@pytest.mark.parametrize("case", ["empty", "positions", "cell", "periodic"])
def test_invalid_structure_fails_before_model_load(copper, monkeypatch, case):
    atoms = Atoms("Cu", cell=[3] * 3, pbc=True)
    if case == "empty":
        atoms = Atoms()
    elif case == "positions":
        atoms.positions[0, 0] = float("nan")
    elif case == "cell":
        atoms.cell[0, 0] = float("inf")
    else:
        atoms.cell[2] = atoms.cell[1]
    monkeypatch.setattr(runner, "ase_read", lambda _: atoms)
    monkeypatch.setattr(
        runner,
        "_ase_backend_context",
        lambda _: pytest.fail("Invalid input must not load a model"),
    )
    result = run_mlip(copper, ASEMACEConfig())
    assert not result["success"]
    assert result["final_structure"] is None
    assert result["error"]


def test_partial_periodic_cell_and_molecule_without_cell():
    runner._validate_atoms(Atoms("Cu", cell=[3, 3, 0], pbc=[True, True, False]))
    runner._validate_atoms(Atoms("H"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("energy", float("nan")),
        ("energy", float("inf")),
        ("forces", np.full((2, 3), float("nan"))),
        ("forces", np.zeros((1, 3))),
        ("stress", np.full((3, 3), float("inf"))),
    ],
)
def test_invalid_calculator_output_is_persisted_failure(
    copper,
    tmp_path,
    monkeypatch,
    field,
    value,
):
    class InvalidCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(
            self, atoms=None, properties=None, system_changes=all_changes
        ):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "energy": 1.0,
                "forces": np.zeros((2, 3)),
                "stress": np.zeros((3, 3)),
                field: value,
            }

    atoms = Atoms("Cu2", cell=[8] * 3, pbc=True)
    monkeypatch.setattr(runner, "ase_read", lambda _: atoms)
    use_calculator(monkeypatch, InvalidCalculator())
    output = tmp_path / "result.json"
    result = run_mlip(copper, ASEMACEConfig(), output_file=output)
    assert not result["success"]
    assert result["energy"] is None
    assert json.loads(output.read_text()) == result
    assert "NaN" not in output.read_text()
    assert "Infinity" not in output.read_text()


@pytest.mark.parametrize("unsupported", [True, False])
def test_stress_unsupported_differs_from_calculator_failure(
    copper,
    monkeypatch,
    unsupported,
):
    class StressCalculator(EMT):
        def get_stress(self, atoms=None):
            if unsupported:
                raise PropertyNotImplementedError("stress unavailable")
            raise RuntimeError("stress calculation crashed")

    atoms = Atoms(
        "Cu2", positions=[[0, 0, 0], [3, 0, 0]], cell=[8] * 3, pbc=True
    )
    monkeypatch.setattr(runner, "ase_read", lambda _: atoms)
    use_calculator(monkeypatch, StressCalculator())
    result = run_mlip(copper, ASEMACEConfig())
    assert result["success"] is unsupported
    assert result["stress"] is None
    if not unsupported:
        assert "stress calculation crashed" in result["error"]


@pytest.mark.parametrize("steps,converged", [(1, False), (300, True)])
def test_optimization_retains_valid_results(
    copper, monkeypatch, steps, converged
):
    use_calculator(monkeypatch, EMT())
    result = run_mlip(
        copper,
        ASEMACEConfig(),
        MLIPCalculationConfig(driver="opt", steps=steps),
    )
    assert result["success"]
    assert result["converged"] is converged
    assert np.isfinite(result["energy"])
    assert result["n_steps"] <= steps


def test_strict_json_preserves_previous_file(tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{"old": true}')
    with pytest.raises(ValueError):
        runner._write_json(path, {"energy": float("nan")})
    assert json.loads(path.read_text()) == {"old": True}


def test_cuda_availability_and_faults(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=cuda, device=str)
    )
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        runner._synchronize_device("cuda")
    runner._synchronize_device("cpu")

    def broken_sync(device):
        raise RuntimeError("device fault")

    cuda.is_available = lambda: True
    cuda.synchronize = broken_sync
    with pytest.raises(RuntimeError, match="device fault"):
        runner._synchronize_device("cuda")
