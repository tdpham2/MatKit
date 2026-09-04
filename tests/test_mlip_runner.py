"""Tests for runtime-selectable, agent-free MLIP execution."""

import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
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
from matkit.mlip import runner as mlip_runner


def _write_copper(path, distance=2.5):
    atoms = Atoms(
        "Cu2",
        positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )
    write(path, atoms)
    return atoms


def _emt_context(counter=None):
    @contextmanager
    def calculator_context(_config):
        if counter is not None:
            counter["entered"] += 1
        yield EMT()

    return calculator_context


def test_backend_config_validation():
    with pytest.raises(ValueError, match="both cluster and root"):
        RootstockConfig(
            checkpoint="mace-mp-0-medium",
            cluster="polaris",
            root="/shared/rootstock",
        )
    with pytest.raises(ValueError, match="dt must be positive"):
        NVAlchemiMACEConfig(checkpoint="medium", dt=0)
    with pytest.raises(ValueError, match="steps must be at least 1"):
        MLIPCalculationConfig(steps=0)


def test_mace_anicc_uses_model_path_signature(monkeypatch):
    received = {}

    def mace_anicc(**kwargs):
        received.update(kwargs)
        return "calculator"

    monkeypatch.setitem(
        sys.modules,
        "mace.calculators",
        SimpleNamespace(mace_anicc=mace_anicc),
    )
    monkeypatch.setitem(
        sys.modules,
        "mace",
        SimpleNamespace(calculators=sys.modules["mace.calculators"]),
    )

    calculator = mlip_runner._create_mace_calculator(
        ASEMACEConfig(
            checkpoint="ani.model",
            calculator_type="mace_anicc",
            device="cuda",
        )
    )

    assert calculator == "calculator"
    assert received == {"device": "cuda", "model_path": "ani.model"}


def test_run_mlip_energy_writes_runtime_neutral_result(tmp_path, monkeypatch):
    input_file = tmp_path / "copper.xyz"
    output_file = tmp_path / "result.json"
    _write_copper(input_file)
    monkeypatch.setattr(mlip_runner, "_ase_backend_context", _emt_context())

    result = run_mlip(
        input_file,
        ASEMACEConfig(checkpoint="unused"),
        output_file=output_file,
    )

    assert result["success"] is True
    assert result["energy"] is not None
    assert len(result["forces"]) == 2
    assert result["stress"] is None
    assert result["n_steps"] == 0
    stored = json.loads(output_file.read_text())
    assert stored == result
    assert stored["backend_info"]["type"] == "ase-mace"
    assert stored["final_structure"]["atomic_numbers"] == [29, 29]


def test_periodic_energy_includes_full_stress(tmp_path, monkeypatch):
    input_file = tmp_path / "periodic.xyz"
    atoms = Atoms(
        "Cu",
        positions=[[0.0, 0.0, 0.0]],
        cell=[3.6, 3.6, 3.6],
        pbc=True,
    )
    write(input_file, atoms)
    monkeypatch.setattr(mlip_runner, "_ase_backend_context", _emt_context())

    result = run_mlip(
        input_file,
        ASEMACEConfig(checkpoint="unused"),
    )

    assert result["success"] is True
    assert np.asarray(result["stress"]).shape == (3, 3)


def test_run_mlip_fixed_cell_optimization(tmp_path, monkeypatch):
    input_file = tmp_path / "copper.xyz"
    initial = _write_copper(input_file, distance=3.0)
    monkeypatch.setattr(mlip_runner, "_ase_backend_context", _emt_context())

    result = run_mlip(
        input_file,
        ASEMACEConfig(checkpoint="unused"),
        MLIPCalculationConfig(driver="opt", steps=2),
    )

    assert result["success"] is True
    assert result["n_steps"] <= 2
    assert result["final_structure"]["cell"] == initial.cell.tolist()


def test_batch_reuses_calculator_and_preserves_failures(tmp_path, monkeypatch):
    first = tmp_path / "first.xyz"
    second = tmp_path / "second.xyz"
    missing = tmp_path / "missing.xyz"
    _write_copper(first)
    _write_copper(second, distance=2.7)
    counter = {"entered": 0}
    monkeypatch.setattr(
        mlip_runner,
        "_ase_backend_context",
        _emt_context(counter),
    )

    summary = run_mlip_batch(
        [first, missing, second],
        ASEMACEConfig(checkpoint="unused"),
        output_dir=tmp_path / "results",
    )

    assert summary["status"] == "partial"
    assert summary["succeeded"] == 2
    assert summary["failed"] == 1
    assert counter["entered"] == 1
    assert [item["index"] for item in summary["items"]] == [0, 1, 2]
    assert [item["status"] for item in summary["items"]] == [
        "success",
        "failure",
        "success",
    ]
    manifest = json.loads(
        (tmp_path / "results" / "batch_manifest.json").read_text()
    )
    assert manifest["status"] == "partial"
    assert all(
        (tmp_path / "results" / name).exists()
        for name in (
            "00000_first.json",
            "00001_missing.json",
            "00002_second.json",
        )
    )


def test_rootstock_context_forwards_options_and_closes(monkeypatch):
    events = []

    class FakeRootstockCalculator:
        def __init__(self, **kwargs):
            events.append(("init", kwargs))

        def __enter__(self):
            events.append(("enter", None))
            return "calculator"

        def __exit__(self, exc_type, exc, traceback):
            events.append(("exit", None))

    monkeypatch.setitem(
        sys.modules,
        "rootstock",
        SimpleNamespace(RootstockCalculator=FakeRootstockCalculator),
    )
    config = RootstockConfig(
        checkpoint="mace-mp-0-medium",
        cluster="polaris",
        setup_kwargs={"default_dtype": "float32"},
        timeout=1200,
        device="cuda",
    )

    with mlip_runner._ase_backend_context(config) as calculator:
        assert calculator == "calculator"

    assert [event[0] for event in events] == ["init", "enter", "exit"]
    kwargs = events[0][1]
    assert kwargs["checkpoint"] == "mace-mp-0-medium"
    assert kwargs["cluster"] == "polaris"
    assert kwargs["setup_kwargs"] == {"default_dtype": "float32"}
    assert kwargs["timeout"] == 1200


def test_nvalchemi_loads_once_and_chunks_in_order(tmp_path, monkeypatch):
    input_files = []
    for index in range(3):
        input_file = tmp_path / f"input_{index}.xyz"
        _write_copper(input_file, distance=2.5 + index * 0.1)
        input_files.append(input_file)

    loaded = []
    chunks = []
    sentinel_model = object()

    def load_model(config):
        loaded.append(config.checkpoint)
        return sentinel_model

    def run_chunk(model, entries, backend, calculation):
        assert model is sentinel_model
        chunks.append([entry[1] for entry in entries])
        return [
            (
                index,
                mlip_runner._success_result(
                    input_file,
                    backend,
                    calculation,
                    atoms,
                    1.25,
                    np.zeros((len(atoms), 3)),
                    None,
                    True,
                    0,
                    0.01,
                ),
            )
            for index, input_file, atoms in entries
        ]

    monkeypatch.setattr(mlip_runner, "_load_nvalchemi_model", load_model)
    monkeypatch.setattr(mlip_runner, "_run_nvalchemi_chunk", run_chunk)
    monkeypatch.setattr(mlip_runner, "_synchronize_device", lambda _: None)
    summary = run_mlip_batch(
        input_files,
        NVAlchemiMACEConfig(checkpoint="medium"),
        output_dir=tmp_path / "results",
        batch_size=3,
        max_atoms=4,
    )

    assert summary["status"] == "completed"
    assert loaded == ["medium"]
    assert [len(chunk) for chunk in chunks] == [2, 1]
    assert chunks[0] + chunks[1] == [
        str(path.resolve()) for path in input_files
    ]


def test_nvalchemi_missing_backend_is_persisted(tmp_path, monkeypatch):
    input_file = tmp_path / "input.xyz"
    output_file = tmp_path / "failure.json"
    _write_copper(input_file)

    def missing_backend(_config):
        raise ImportError("Install the nvalchemi_mace extra")

    monkeypatch.setattr(mlip_runner, "_load_nvalchemi_model", missing_backend)
    result = run_mlip(
        input_file,
        NVAlchemiMACEConfig(checkpoint="medium"),
        output_file=output_file,
    )

    assert result["success"] is False
    assert "nvalchemi_mace" in result["error"]
    assert json.loads(output_file.read_text())["success"] is False


def test_nvalchemi_rejects_non_fire_optimizer(tmp_path):
    input_file = tmp_path / "input.xyz"
    _write_copper(input_file)

    with pytest.raises(ValueError, match="only the FIRE optimizer"):
        run_mlip(
            input_file,
            NVAlchemiMACEConfig(checkpoint="medium"),
            MLIPCalculationConfig(driver="opt", optimizer="bfgs"),
        )
