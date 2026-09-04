"""Durability regressions for completed results and interrupted batches."""

from contextlib import contextmanager
import json
from pathlib import Path

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
    run_mlip_batch,
)
from matkit.mlip import runner


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    paths = []
    for directory in ("a", "b"):
        path = tmp_path / directory / "same.xyz"
        path.parent.mkdir()
        write(
            path, Atoms("Cu2", positions=[[0, 0, 0], [3, 0, 0]], cell=[8] * 3)
        )
        paths.append(path)

    @contextmanager
    def context(_backend):
        yield EMT()

    monkeypatch.setattr(runner, "_ase_backend_context", context)
    return paths


def read_manifest(out):
    return json.loads((out / "batch_manifest.json").read_text())


def test_interrupt_preserves_completed_item(inputs, tmp_path, monkeypatch):
    out = tmp_path / "results"
    original = runner._run_ase_item

    def run_item(input_file, *args):
        manifest = read_manifest(out)
        assert manifest["status"] == "running"
        if input_file == str(inputs[1]):
            assert manifest["succeeded"] == 1
            assert manifest["pending"] == 1
            stored = json.loads(
                Path(manifest["items"][0]["result_file"]).read_text()
            )
            assert stored["success"]
            raise KeyboardInterrupt()
        assert manifest["pending"] == 2
        return original(input_file, *args)

    monkeypatch.setattr(runner, "_run_ase_item", run_item)
    with pytest.raises(KeyboardInterrupt):
        run_mlip_batch(inputs, ASEMACEConfig(), output_dir=out)
    manifest = read_manifest(out)
    assert manifest["status"] == "interrupted"
    assert manifest["error"] == "KeyboardInterrupt"
    assert manifest["succeeded"] == manifest["pending"] == 1
    assert not (out / "00001_same.json").exists()


def test_teardown_failure_preserves_all_results(inputs, tmp_path, monkeypatch):
    @contextmanager
    def broken_context(_backend):
        yield EMT()
        raise RuntimeError("worker cleanup failed")

    monkeypatch.setattr(runner, "_ase_backend_context", broken_context)
    out = tmp_path / "results"
    with pytest.raises(RuntimeError, match="worker cleanup failed"):
        run_mlip_batch(inputs, RootstockConfig("test"), output_dir=out)
    manifest = read_manifest(out)
    assert manifest["status"] == "interrupted"
    assert manifest["succeeded"] == 2
    assert manifest["failed"] == manifest["pending"] == 0
    assert manifest["error"] == "worker cleanup failed"
    for item in manifest["items"]:
        assert item["status"] == "success"
        assert json.loads(Path(item["result_file"]).read_text())["success"]


@pytest.mark.parametrize("failure", ["result", "manifest"])
def test_persistence_failure_does_not_become_calculation_failure(
    inputs,
    tmp_path,
    monkeypatch,
    failure,
):
    out = tmp_path / "results"
    replace = runner.os.replace
    failed = False

    def fail_once(source, destination):
        nonlocal failed
        destination = Path(destination)
        if failure == "result":
            trigger = destination.name == "00001_same.json"
        else:
            trigger = (
                destination.name == "batch_manifest.json"
                and (out / "00001_same.json").exists()
            )
        if trigger and not failed:
            failed = True
            raise OSError("disk write failed")
        return replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", fail_once)
    with pytest.raises(OSError, match="disk write failed"):
        run_mlip_batch(inputs, ASEMACEConfig(), output_dir=out)
    manifest = read_manifest(out)
    assert manifest["status"] == "interrupted"
    assert manifest["failed"] == 0
    assert manifest["succeeded"] == (1 if failure == "result" else 2)
    assert manifest["pending"] == (1 if failure == "result" else 0)
    assert json.loads((out / "00000_same.json").read_text())["success"]
    assert not list(out.glob("*.tmp"))


def test_atomic_replace_failure_retains_previous_json(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    path.write_text('{"previous": true}')

    def fail(*args):
        raise OSError("replace failed")

    monkeypatch.setattr(runner.os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        runner._write_json(path, {"new": True})
    assert json.loads(path.read_text()) == {"previous": True}
    assert list(tmp_path.iterdir()) == [path]


def test_fresh_directory_and_complete_payloads(inputs, tmp_path, monkeypatch):
    out = tmp_path / "results"
    summary = run_mlip_batch(inputs, ASEMACEConfig(), output_dir=out)
    manifest = read_manifest(out)
    assert manifest == {
        k: v
        for k, v in summary.items()
        if k not in ("manifest_file", "results")
    }
    assert manifest["pending"] == manifest["unconverged"] == 0
    assert [Path(item["result_file"]).name for item in manifest["items"]] == [
        "00000_same.json",
        "00001_same.json",
    ]
    for item, result in zip(manifest["items"], summary["results"]):
        assert json.loads(Path(item["result_file"]).read_text()) == result
    original_manifest = (out / "batch_manifest.json").read_bytes()
    monkeypatch.setattr(
        runner,
        "_ase_backend_context",
        lambda _: pytest.fail("existing batch was rerun"),
    )
    with pytest.raises(FileExistsError, match="fresh directory"):
        run_mlip_batch(inputs, ASEMACEConfig(), output_dir=out)
    assert (out / "batch_manifest.json").read_bytes() == original_manifest


def test_concurrent_writer_cannot_claim_directory(inputs, tmp_path):
    out = tmp_path / "results"
    with runner._fresh_batch_directory(out):
        with pytest.raises(FileExistsError):
            run_mlip_batch(inputs, ASEMACEConfig(), output_dir=out)
        assert (out / ".matkit_batch.lock").exists()


def test_valid_unconverged_batch_is_counted(inputs, tmp_path):
    summary = run_mlip_batch(
        inputs,
        ASEMACEConfig(),
        MLIPCalculationConfig(driver="opt", steps=1),
        output_dir=tmp_path / "results",
    )
    assert summary["status"] == "completed"
    assert summary["succeeded"] == summary["unconverged"] == 2
    assert all(item["converged"] is False for item in summary["items"])


def test_startup_failure_retains_input_errors(inputs, tmp_path, monkeypatch):
    @contextmanager
    def broken_start(_backend):
        raise ImportError("missing calculator")
        yield

    monkeypatch.setattr(runner, "_ase_backend_context", broken_start)
    summary = run_mlip_batch(
        [inputs[0], tmp_path / "missing.xyz"],
        ASEMACEConfig(),
        output_dir=tmp_path / "results",
    )
    assert summary["status"] == "failure"
    assert summary["failed"] == 2
    assert "missing calculator" in summary["items"][0]["error"]
    assert "does not exist" in summary["items"][1]["error"]


def test_rootstock_gpu_worker_does_not_require_parent_cuda(
    inputs, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        runner,
        "_synchronize_device",
        lambda _: pytest.fail("Rootstock CUDA belongs to the worker"),
    )
    result = run_mlip_batch(
        inputs,
        RootstockConfig("medium", device="cuda"),
        output_dir=tmp_path / "results",
    )
    assert result["succeeded"] == 2


def test_native_chunk_is_saved_before_next_chunk(inputs, tmp_path, monkeypatch):
    out = tmp_path / "results"
    monkeypatch.setattr(runner, "_load_nvalchemi_model", lambda _: object())
    monkeypatch.setattr(runner, "_synchronize_device", lambda _: None)

    def chunk(model, entries, backend, calculation):
        index, input_file, atoms = entries[0]
        if index:
            assert read_manifest(out)["succeeded"] == 1
            assert json.loads((out / "00000_same.json").read_text())["success"]
            raise KeyboardInterrupt()
        return [
            (
                index,
                runner._success_result(
                    input_file,
                    backend,
                    calculation,
                    atoms,
                    1.0,
                    np.zeros((len(atoms), 3)),
                    None,
                    True,
                    0,
                    0.1,
                ),
            )
        ]

    monkeypatch.setattr(runner, "_run_nvalchemi_chunk", chunk)
    with pytest.raises(KeyboardInterrupt):
        run_mlip_batch(
            inputs, NVAlchemiMACEConfig("medium"), output_dir=out, batch_size=1
        )
    assert read_manifest(out)["status"] == "interrupted"
