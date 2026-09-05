"""Integrity checks for interruption, mutable inputs and worker artifacts."""

import json
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import write

from matkit.api import (
    EvaluateRequest,
    MLIPMethod,
    PoreRequest,
    StructureRef,
    execute,
    inspect_run,
    prepare,
    run_batch,
)
from matkit.api.models import BatchResult, Failure


def test_completed_items_do_not_hide_batch_orchestration_failure():
    result = BatchResult(
        state="interrupted",
        items=[{"accepted": True}],
        failure=Failure(
            code="RuntimeError", stage="teardown", message="worker failed"
        ),
    )
    assert not result.accepted


def test_active_batch_owns_its_prepared_items(sample_cif, tmp_path):
    root = tmp_path / "batch"
    root.mkdir()
    leaf = root / "00000"
    prepare(
        PoreRequest(structure=StructureRef(path=sample_cif)), output_dir=leaf
    )
    (root / ".matkit.lock").write_text("1")
    (leaf / ".matkit.batch").write_text("1")
    with pytest.raises(FileExistsError, match="active batch"):
        execute(leaf)


def test_changed_model_cannot_be_reused_across_batch(tmp_path, monkeypatch):
    from matkit.api import runtime

    path = tmp_path / "cu.extxyz"
    write(path, Atoms("Cu", cell=[10] * 3))
    model = tmp_path / "model.pt"
    model.write_bytes(b"model-one")
    request = EvaluateRequest(
        structure=StructureRef(path=str(path)),
        method=MLIPMethod(checkpoint=str(model)),
    )
    original = runtime.prepare

    def prepare_item(*args, **kwargs):
        result = original(*args, **kwargs)
        model.write_bytes(b"model-two")
        return result

    monkeypatch.setattr(runtime, "prepare", prepare_item)
    with pytest.raises(ValueError, match="Supporting inputs changed"):
        run_batch([request, request], output_dir=tmp_path / "batch")
    batch = json.loads((tmp_path / "batch" / "batch_manifest.json").read_text())
    assert batch["state"] == "interrupted"
    assert all(not item["accepted"] for item in batch["items"])


def test_atomic_committed_result_survives_corrupt_manifest(
    sample_cif, tmp_path
):
    from tests.test_api_runtime import fake_execution

    root = tmp_path / "run"
    prepare(
        PoreRequest(structure=StructureRef(path=sample_cif)), output_dir=root
    )
    result = execute(root, execution=fake_execution("zeopp"))
    (root / "run.json").write_text("corrupt")
    assert inspect_run(root).payload == result.payload


def test_environment_values_are_not_retrievable_artifacts(sample_cif, tmp_path):
    from tests.test_api_runtime import fake_execution

    root = tmp_path / "run"
    prepare(
        PoreRequest(structure=StructureRef(path=sample_cif)), output_dir=root
    )
    result = execute(
        root,
        execution=fake_execution(
            "zeopp",
            mode="subprocess",
            environment={"MATKIT_TEST_CREDENTIAL": "fixture-secret"},
        ),
    )
    assert result.accepted
    for item in result.artifacts:
        if Path(item.path).suffix == ".json":
            assert "fixture-secret" not in (root / item.path).read_text()
