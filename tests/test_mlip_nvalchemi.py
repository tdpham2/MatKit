"""Exercise ALCHEMI adapter boundaries with CPU-only protocol doubles.

These tests verify MatKit's conversions and dispatch, not GPU kernel behavior
or compatibility with a live installation. That evidence comes from smoke.pbs.
"""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from matkit.mlip import MLIPCalculationConfig, NVAlchemiMACEConfig, run_mlip
from matkit.mlip import runner


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values.copy()

    def squeeze(self, axis):
        return Tensor(self.values.squeeze(axis))

    def tolist(self):
        return self.values.tolist()


@pytest.fixture
def alchemi(monkeypatch):
    modules = {}
    for name in (
        "torch",
        "nvalchemi",
        "nvalchemi.data",
        "nvalchemi.dynamics",
        "nvalchemi.models",
        "nvalchemi.models.mace",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        modules[name] = module
        if "." in name:
            parent, leaf = name.rsplit(".", 1)
            setattr(modules[parent], leaf, module)
    state = SimpleNamespace(
        events=[], model_loads=[], corrupt=None, converged=[0]
    )
    torch = modules["torch"]
    torch.device = str
    torch.float32, torch.float64 = "float32", "float64"
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda device: state.events.append(("sync", device)),
    )

    def zeros(*shape, device, dtype):
        state.events.append(("zeros", shape, device, dtype))
        return Tensor(np.zeros(shape, dtype=dtype))

    torch.zeros = zeros

    class AtomicData:
        @classmethod
        def from_atoms(cls, atoms, *, device, dtype):
            state.events.append(("from_atoms", len(atoms), device, dtype))
            return SimpleNamespace(
                num_nodes=len(atoms),
                device=device,
                positions=Tensor(atoms.positions.astype(dtype)),
                cell=Tensor(atoms.cell.array[None].astype(dtype)),
                pbc=Tensor(atoms.pbc[None]),
                stress=None,
            )

    class Batch:
        @classmethod
        def from_data_list(cls, data):
            result = cls()
            result.data = data
            return result

        def to_data_list(self):
            return self.data

    class Model:
        def eval(self):
            state.events.append(("eval",))

        def make_neighbor_hooks(self):
            return ["neighbor-hook"]

    class MACEWrapper:
        @classmethod
        def from_checkpoint(cls, checkpoint, **kwargs):
            state.model_loads.append((checkpoint, kwargs))
            return Model()

    class BaseDynamics:
        def __init__(self, **kwargs):
            state.events.append((type(self).__name__, kwargs))

        def __enter__(self):
            state.events.append(("enter",))
            return self

        def __exit__(self, *args):
            state.events.append(("exit",))

        def run(self, batch):
            for index, data in enumerate(batch.data):
                assert data.energy.numpy().shape == (1, 1)
                assert data.forces.numpy().shape == (data.num_nodes, 3)
                assert data.velocities.numpy().shape == (data.num_nodes, 3)
                data.energy = Tensor([[index + 1.25]])
                data.forces = Tensor(np.zeros((data.num_nodes, 3)))
                data.stress = Tensor(np.eye(3)[None])
                if isinstance(self, FIRE):
                    data.positions = Tensor(data.positions.numpy() + 0.1)
                if state.corrupt and index == 1:
                    state.corrupt(data)
            return batch

    class FIRE(BaseDynamics):
        pass

    class ConvergenceHook:
        @classmethod
        def from_fmax(cls, value):
            state.events.append(("fmax", value))
            return cls()

        def evaluate(self, batch):
            return None if state.converged is None else Tensor(state.converged)

    modules["nvalchemi.data"].AtomicData = AtomicData
    modules["nvalchemi.data"].Batch = Batch
    modules["nvalchemi.models.mace"].MACEWrapper = MACEWrapper
    for cls in (BaseDynamics, FIRE, ConvergenceHook):
        setattr(modules["nvalchemi.dynamics"], cls.__name__, cls)
    state.model = Model()
    return state


def entries():
    return [
        (
            3,
            "first.xyz",
            Atoms("Cu", positions=[[1, 2, 3]], cell=[8] * 3, pbc=True),
        ),
        (
            7,
            "second.xyz",
            Atoms("Cu2", positions=[[0, 0, 0], [2, 0, 0]], cell=[9] * 3),
        ),
    ]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_model_loading_and_atomic_conversion(alchemi, dtype):
    backend = NVAlchemiMACEConfig(
        "weights.pt", dtype=dtype, compile_model=True, enable_cueq=True
    )
    runner._load_nvalchemi_model(backend)
    assert alchemi.model_loads == [
        (
            "weights.pt",
            {
                "device": "cuda",
                "dtype": dtype,
                "compile_model": True,
                "enable_cueq": True,
            },
        )
    ]
    assert ("eval",) in alchemi.events
    atoms = entries()[1][2]
    data = runner._atoms_to_nvalchemi_data(atoms, backend)
    assert np.array_equal(data.positions.numpy(), atoms.positions)
    assert data.positions.numpy().dtype == np.dtype(dtype)
    assert np.array_equal(data.cell.numpy()[0], atoms.cell.array)
    assert np.array_equal(data.pbc.numpy()[0], atoms.pbc)
    assert data.forces.numpy().shape == (2, 3)
    assert ("zeros", (2, 3), "cuda", dtype) in alchemi.events


@pytest.mark.parametrize("driver", ["energy", "opt"])
def test_dynamics_and_result_mapping(alchemi, driver):
    inputs = entries()
    backend = NVAlchemiMACEConfig("medium", dt=0.2)
    calculation = MLIPCalculationConfig(driver=driver, fmax=0.02, steps=7)
    outputs = runner._run_nvalchemi_chunk(
        alchemi.model, inputs, backend, calculation
    )
    assert [index for index, _ in outputs] == [3, 7]
    for (_, result), (_, _, original) in zip(outputs, inputs):
        assert result["success"]
        assert (
            result["final_structure"]["atomic_numbers"]
            == original.numbers.tolist()
        )
        assert result["final_structure"]["cell"] == original.cell.array.tolist()
        assert result["final_structure"]["pbc"] == original.pbc.tolist()
        expected = original.positions + (0.1 if driver == "opt" else 0)
        assert np.allclose(result["final_structure"]["positions"], expected)
        assert np.array_equal(result["stress"], np.eye(3))
        assert result["force_unit"] == "eV/angstrom"
    name = "FIRE" if driver == "opt" else "BaseDynamics"
    settings = next(event[1] for event in alchemi.events if event[0] == name)
    assert settings["hooks"] == ["neighbor-hook"]
    assert settings["n_steps"] == (7 if driver == "opt" else 1)
    assert [result["energy"] for _, result in outputs] == [1.25, 2.25]
    assert ("enter",) in alchemi.events and ("exit",) in alchemi.events
    assert [result["converged"] for _, result in outputs] == [
        True,
        driver == "energy",
    ]
    if driver == "opt":
        assert settings["dt"] == 0.2
        assert ("fmax", 0.02) in alchemi.events


@pytest.mark.parametrize(
    "field,value",
    [
        ("energy", [[1, 2]]),
        ("energy", [[float("nan")]]),
        ("forces", [[0, 0, 0]]),
        ("stress", np.zeros((2, 3, 3))),
    ],
)
def test_one_invalid_native_result_preserves_other_items(alchemi, field, value):
    alchemi.corrupt = lambda data: setattr(data, field, Tensor(value))
    outputs = runner._run_nvalchemi_chunk(
        alchemi.model,
        entries(),
        NVAlchemiMACEConfig("medium"),
        MLIPCalculationConfig(),
    )
    assert outputs[0][1]["success"]
    assert not outputs[1][1]["success"]
    assert outputs[1][1]["error"]


def test_no_native_convergence(alchemi):
    alchemi.converged = None
    outputs = runner._run_nvalchemi_chunk(
        alchemi.model,
        entries(),
        NVAlchemiMACEConfig("medium"),
        MLIPCalculationConfig(driver="opt"),
    )
    assert all(
        result["success"] and not result["converged"] for _, result in outputs
    )


def test_energy_does_not_apply_optimizer_restriction(alchemi, tmp_path):
    path = tmp_path / "structure.xyz"
    write(path, entries()[0][2])
    result = run_mlip(
        path,
        NVAlchemiMACEConfig("medium"),
        MLIPCalculationConfig(optimizer="bfgs"),
    )
    assert result["success"]
    assert any(event[0] == "BaseDynamics" for event in alchemi.events)
