"""Committed science and worker failure must remain independently visible."""

import json
from pathlib import Path
import sys
import time

from click.testing import CliRunner
import pytest

from matkit.api import (
    PoreRequest,
    RunResult,
    StructureRef,
    inspect_run,
    prepare,
    run,
    run_batch,
)
from matkit.api import runtime
from matkit.api.structures import sha256
from matkit.cli import main
from tests.test_api_runtime import fake_execution


def worker_command(behavior):
    fixture = Path(__file__).parent / "fixtures" / "post_commit_worker.py"

    def command(root, execution, batch):
        args = [sys.executable, str(fixture), str(root), behavior]
        return [*args, "--batch"] if batch else args

    return command


def assert_preserved(root):
    committed = RunResult.model_validate_json(
        (root / "result.json").read_text()
    )
    inspected = inspect_run(root)
    assert committed.accepted
    assert inspected.state == "interrupted"
    assert not inspected.accepted
    assert inspected.failure.stage == "orchestration"
    assert inspected.numerical_validity == "valid"
    assert inspected.payload == committed.payload
    assert (root / "worker.finished").exists()
    assert not (root / ".matkit.lock").exists()
    assert all(sha256(root / a.path) == a.sha256 for a in inspected.artifacts)
    return inspected


@pytest.mark.parametrize("behavior", ["hang", "bad_exit", "teardown_error"])
def test_post_commit_worker_failures(
    behavior, sample_cif, tmp_path, monkeypatch
):
    monkeypatch.setattr(runtime, "_worker_command", worker_command(behavior))
    root = tmp_path / "run"
    result = run(
        PoreRequest(structure=StructureRef(path=sample_cif)),
        output_dir=root,
        execution=fake_execution("zeopp", mode="subprocess", timeout_s=3),
    )
    assert result == assert_preserved(root)
    if behavior == "hang":
        assert result.failure.code == "TimeoutExpired"


def test_cli_reports_post_commit_failure(sample_cif, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_worker_command", worker_command("bad_exit"))
    root = tmp_path / "run"
    prepare(
        PoreRequest(structure=StructureRef(path=sample_cif)), output_dir=root
    )
    profile = tmp_path / "execution.json"
    profile.write_text(fake_execution("zeopp").model_dump_json())
    runner = CliRunner()
    response = runner.invoke(
        main,
        ["execute", str(root), "--execution", str(profile)],
    )
    assert response.exit_code == 1, response.output
    assert json.loads(response.stdout)["state"] == "interrupted"
    assert_preserved(root)
    inspected = runner.invoke(main, ["inspect", str(root)])
    assert inspected.exit_code == 0, inspected.output
    assert json.loads(inspected.stdout)["state"] == "interrupted"


def test_keyboard_interrupt_preserves_committed_result(
    sample_cif, tmp_path, monkeypatch
):
    monkeypatch.setattr(runtime, "_worker_command", worker_command("hang"))
    root = tmp_path / "run"
    original_popen = runtime.subprocess.Popen

    def launch(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        original_wait = process.wait
        interrupted = False

        def wait(timeout=None):
            nonlocal interrupted
            if not interrupted:
                deadline = time.monotonic() + 5
                while not (root / "worker.finished").exists():
                    if time.monotonic() >= deadline:
                        raise AssertionError("worker did not commit")
                    time.sleep(0.025)
                interrupted = True
                raise KeyboardInterrupt("fixture cancellation")
            return original_wait(timeout=timeout)

        process.wait = wait
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    with pytest.raises(KeyboardInterrupt, match="fixture cancellation"):
        run(
            PoreRequest(structure=StructureRef(path=sample_cif)),
            output_dir=root,
            execution=fake_execution("zeopp", mode="subprocess"),
        )
    assert_preserved(root)


@pytest.mark.parametrize("behavior", ["hang", "bad_exit"])
def test_batch_retains_completed_items_after_worker_failure(
    behavior, sample_cif, tmp_path, monkeypatch
):
    monkeypatch.setattr(runtime, "_worker_command", worker_command(behavior))
    request = PoreRequest(structure=StructureRef(path=sample_cif))
    root = tmp_path / "batch"
    result = run_batch(
        [request, request],
        output_dir=root,
        execution=fake_execution("zeopp", mode="subprocess", timeout_s=3),
    )
    assert result.state == "interrupted"
    assert not result.accepted
    assert result.failure is not None
    assert all(item["accepted"] for item in result.items)
    assert all(
        inspect_run(root / item["bundle"]).accepted for item in result.items
    )
