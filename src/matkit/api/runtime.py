"""Synchronous operations, supervised workers and durable batches."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import adapters
from .bundles import (
    atomic_json,
    claim,
    collect_artifacts,
    commit_result,
    environment_versions,
    inspect_run,
    prepare,
    refresh_artifacts,
    staged_request,
    verify_inputs,
)
from .models import (
    AdsorptionRequest,
    BatchResult,
    CalculatorRequest,
    EvaluateRequest,
    ExecutionConfig,
    Failure,
    PoreRequest,
    RelaxRequest,
    parse_request,
)
from .structures import load_structure


def _begin(root, record, execution):
    verify_inputs(root, record)
    request = staged_request(root)
    record = record.model_copy(deep=True)
    record.state = "running"
    record.provenance["execution_environment"] = environment_versions()
    record.resolved["execution"] = execution.model_dump(mode="json")
    if isinstance(request, CalculatorRequest):
        backend = adapters.backend_config(request, execution)
        record.resolved["calculator"] = backend.to_dict()
        if request.method.calculator_type == "mace_anicc":
            record.resolved["calculator"]["dtype"] = None
            record.provenance["precision_source"] = "calculator factory"
        if request.adapter.type == "rootstock":
            record.provenance["calculator_environment"] = (
                "Rootstock worker; resolved environment unavailable"
            )
        # Checkpoint aliases may resolve inside a worker; do not invent a hash.
        record.provenance["model_identity_evidence"] = (
            "content hash"
            if record.provenance.get("model_sha256")
            else "requested identifier only"
        )
    atomic_json(root / "run.json", record)
    return request, record


def _finish(root, record, output, started):
    payload, checks, timings = output
    finished = record.model_copy(
        update={
            "state": "completed",
            "numerical_validity": "valid",
            "payload": payload,
            "checks": checks,
            "timings": {**timings, "wall_s": time.perf_counter() - started},
            "artifacts": collect_artifacts(root),
            "failure": None,
        }
    )
    return commit_result(root, finished)


def _fail(root, record, exc, stage, interrupted=False):
    # Never replace an already committed scientific result after a write or
    # teardown failure. The orchestration error remains visible to the caller.
    if (root / "result.json").exists():
        try:
            committed = inspect_run(root)
            failure = Failure(
                code=type(exc).__name__,
                stage="orchestration",
                message=str(exc) or type(exc).__name__,
            )
            atomic_json(
                root / "run.json",
                committed.model_copy(
                    update={"state": "interrupted", "failure": failure}
                ),
            )
        except Exception:
            pass  # The committed result still survives a filesystem failure.
        raise exc
    try:
        artifacts = collect_artifacts(root)
    except Exception:
        artifacts = record.artifacts
    failed = record.model_copy(
        update={
            "state": "interrupted" if interrupted else "failed",
            "numerical_validity": "invalid"
            if stage == "calculation"
            else "unknown",
            "failure": Failure(
                code=type(exc).__name__,
                stage=stage,
                message=str(exc) or type(exc).__name__,
            ),
            "artifacts": artifacts,
        }
    )
    return commit_result(root, failed)


def _execute_claimed(root, execution):
    record = inspect_run(root)
    if record.state != "prepared":
        raise FileExistsError(
            "Only prepared runs can execute; use a fresh bundle for reruns"
        )
    started = time.perf_counter()
    stage = "validation"
    try:
        request, record = _begin(root, record, execution)
        stage = "calculation"
        if isinstance(request, CalculatorRequest):
            atoms, _, _ = load_structure(request.structure)
            setup = time.perf_counter()
            with adapters.calculator_session(request, execution) as session:
                setup_s = time.perf_counter() - setup
                _, legacy = next(
                    adapters.evaluate_items(
                        [(0, request.structure.path, atoms)],
                        request,
                        execution,
                        session,
                    )
                )
                output = adapters.calculator_payload(root, request, legacy)
                output[2]["setup_s"] = setup_s
                result = _finish(root, record, output, started)
            return result
        return _finish(
            root,
            record,
            adapters.run_external(root, request, execution),
            started,
        )
    except Exception as exc:
        return _fail(
            root,
            record,
            exc,
            stage,
            interrupted=isinstance(exc, subprocess.TimeoutExpired),
        )
    except BaseException as exc:
        _fail(root, record, exc, stage, interrupted=True)
        raise


def _worker_command(root, execution, batch):
    python = execution.python or sys.executable
    command = [python, "-m", "matkit.worker", str(root)]
    if batch:
        command.append("--batch")
    return command


def _interrupt_run(root, exc):
    """Record interruption without replacing committed science."""
    record = inspect_run(root)
    if not (root / "result.json").exists():
        return _fail(root, record, exc, "worker", interrupted=True)
    interrupted = record.model_copy(
        update={
            "state": "interrupted",
            "failure": Failure(
                code=type(exc).__name__,
                stage="orchestration",
                message=str(exc) or type(exc).__name__,
            ),
        }
    )
    atomic_json(root / "run.json", interrupted)
    return interrupted


def _supervise_claimed(root, execution, batch=False):
    atomic_json(
        root / "execution.json",
        execution.model_copy(update={"environment": {}}),
    )
    with (
        (root / "worker.stdout.log").open("w") as stdout,
        (root / "worker.stderr.log").open("w") as stderr,
    ):
        process = subprocess.Popen(
            _worker_command(root, execution, batch),
            env={
                **os.environ,
                **execution.environment,
                "MATKIT_WORKER_PROCESS": "1",
            },
            stdout=stdout,
            stderr=stderr,
            start_new_session=os.name == "posix",
        )
        try:
            code = process.wait(timeout=execution.timeout_s)
            adapters.stop_process(process)
        except BaseException as exc:
            adapters.stop_process(process)
            if batch:
                _interrupt_batch(root, exc)
            else:
                result = refresh_artifacts(root, _interrupt_run(root, exc))
            if isinstance(exc, subprocess.TimeoutExpired):
                if batch:
                    return read_batch(root)
                return result
            raise
    if batch:
        result = read_batch(root)
        if result.state == "running" or code != (0 if result.accepted else 1):
            _interrupt_batch(
                root, RuntimeError(f"Worker exited with code {code}")
            )
        return read_batch(root)
    result = inspect_run(root)
    if result.state in {"prepared", "running"}:
        return _fail(
            root,
            result,
            RuntimeError(
                f"Worker exited with code {code}; see worker.stderr.log"
            ),
            "worker",
            interrupted=True,
        )
    if code != (0 if result.accepted else 1):
        result = _interrupt_run(
            root,
            RuntimeError(
                f"Worker exited with code {code}; see worker.stderr.log"
            ),
        )
    return refresh_artifacts(root, result)


def execute(
    bundle: str | Path, *, execution: ExecutionConfig | dict | None = None
):
    root = Path(bundle).expanduser().resolve()
    if (root / ".matkit.batch").exists() and (
        root.parent / ".matkit.lock"
    ).exists():
        raise FileExistsError("This item is owned by an active batch")
    config = ExecutionConfig.model_validate(execution or {})
    with claim(root):
        record = inspect_run(root)
        if record.state != "prepared":
            raise FileExistsError("Only prepared bundles can execute")
        verify_inputs(root, record)
        if (
            config.mode == "subprocess"
            or config.python
            or config.environment
            or config.timeout_s
        ):
            try:
                return _supervise_claimed(root, config)
            except OSError as exc:
                return _fail(root, record, exc, "worker")
        return _execute_claimed(root, config)


def run(request, *, output_dir, execution=None):
    prepare(request, output_dir=output_dir)
    return execute(output_dir, execution=execution)


def _operation(value, cls):
    if isinstance(value, dict):
        value = cls.model_validate(value)
    if not isinstance(value, cls):
        raise ValueError(f"Expected {cls.__name__}")
    return value


def evaluate(request: EvaluateRequest | dict, *, output_dir, execution=None):
    return run(
        _operation(request, EvaluateRequest),
        output_dir=output_dir,
        execution=execution,
    )


def relax(request: RelaxRequest | dict, *, output_dir, execution=None):
    return run(
        _operation(request, RelaxRequest),
        output_dir=output_dir,
        execution=execution,
    )


def analyze_pores(request: PoreRequest | dict, *, output_dir, execution=None):
    return run(
        _operation(request, PoreRequest),
        output_dir=output_dir,
        execution=execution,
    )


def prepare_adsorption(request: AdsorptionRequest | dict, *, output_dir):
    return prepare(
        _operation(request, AdsorptionRequest), output_dir=output_dir
    )


def run_adsorption(
    request: AdsorptionRequest | dict, *, output_dir, execution=None
):
    return run(
        _operation(request, AdsorptionRequest),
        output_dir=output_dir,
        execution=execution,
    )


def analyze_adsorption(bundle: str | Path):
    """Inspect a run, or collect a manually executed prepared gRASPA bundle.

    Manual execution must save work/raspa.log and exit.json containing the
    engine's returncode. Prefer ``matkit execute`` to record these reliably.
    """
    root = Path(bundle).expanduser().resolve()
    with claim(root):
        record = inspect_run(root)
        if record.operation != "adsorption":
            raise ValueError("Expected an adsorption bundle")
        if record.state not in {"prepared", "running"}:
            return record
        request = staged_request(root)
        # Verify the original inventory, excluding newly created logs.
        verify_inputs(root, record)
        exit_record = json.loads((root / "exit.json").read_text())
        started = time.perf_counter()
        try:
            if exit_record.get("returncode") != 0:
                raise ValueError("Engine did not report a zero exit code")
            return _finish(
                root, record, adapters.parse_adsorption(root, request), started
            )
        except Exception as exc:
            return _fail(root, record, exc, "collection")


def read_batch(root):
    return BatchResult.model_validate_json(
        (Path(root) / "batch_manifest.json").read_text()
    )


def _checkpoint_batch(root, state=None, failure=None):
    batch = read_batch(root)
    for item in batch.items:
        item_dir = root / item["bundle"]
        if (item_dir / "run.json").exists():
            result = inspect_run(item_dir)
            item.update(
                state=result.state,
                accepted=result.accepted,
                result_file=f"{item['bundle']}/result.json"
                if (item_dir / "result.json").exists()
                else None,
                failure=result.failure.model_dump(mode="json")
                if result.failure
                else None,
            )
    if state:
        batch.state = state
    if failure:
        batch.failure = failure
    atomic_json(root / "batch_manifest.json", batch)
    return batch


def _interrupt_batch(root, exc):
    failure = Failure(
        code=type(exc).__name__,
        stage="batch",
        message=str(exc) or type(exc).__name__,
    )
    batch = read_batch(root)
    for item in batch.items:
        item_dir = root / item["bundle"]
        if (item_dir / "run.json").exists() and not (
            item_dir / "result.json"
        ).exists():
            _fail(
                item_dir, inspect_run(item_dir), exc, "batch", interrupted=True
            )
    return _checkpoint_batch(root, "interrupted", failure)


def _execute_batch_claimed(root, execution):
    batch = read_batch(root)
    entries, records, requests = [], {}, {}
    started = time.perf_counter()
    try:
        for item in batch.items:
            item_root = root / item["bundle"]
            if item["state"] != "prepared":
                continue
            index = item["index"]
            record = inspect_run(item_root)
            try:
                request, record = _begin(item_root, record, execution)
                atoms, _, _ = load_structure(request.structure)
                requests[index], records[index] = request, record
                entries.append((index, request.structure.path, atoms))
            except Exception as exc:
                _fail(item_root, record, exc, "validation")
        _checkpoint_batch(root)
        if entries:
            supporting_hashes = [
                {
                    path: digest
                    for path, digest in record.provenance[
                        "prepared_hashes"
                    ].items()
                    if path.startswith("inputs/")
                    and not path.startswith("inputs/structure")
                }
                for record in records.values()
            ]
            if any(
                hashes != supporting_hashes[0]
                for hashes in supporting_hashes[1:]
            ):
                raise ValueError(
                    "Supporting inputs changed while preparing the batch"
                )
            request = requests[entries[0][0]]
            if isinstance(request, CalculatorRequest):
                setup = time.perf_counter()
                try:
                    with adapters.calculator_session(
                        request, execution
                    ) as session:
                        setup_s = time.perf_counter() - setup
                        for index, legacy in adapters.evaluate_items(
                            entries, request, execution, session
                        ):
                            item_root = root / batch.items[index]["bundle"]
                            try:
                                output = adapters.calculator_payload(
                                    item_root, requests[index], legacy
                                )
                                output[2]["setup_s"] = setup_s
                            except Exception as exc:
                                _fail(
                                    item_root,
                                    records[index],
                                    exc,
                                    "calculation",
                                )
                            else:
                                _finish(
                                    item_root, records[index], output, started
                                )
                            _checkpoint_batch(root)
                except Exception as exc:
                    # Startup failures affect pending items. Teardown and
                    # persistence failures must preserve completed results.
                    if any(
                        (root / item["bundle"] / "result.json").exists()
                        for item in batch.items
                        if item["index"] in records
                    ):
                        raise
                    for index in records:
                        _fail(
                            root / batch.items[index]["bundle"],
                            records[index],
                            exc,
                            "calculator_setup",
                        )
            else:
                for index, _, _ in entries:
                    item_root = root / batch.items[index]["bundle"]
                    try:
                        output = adapters.run_external(
                            item_root, requests[index], execution
                        )
                    except Exception as exc:
                        _fail(item_root, records[index], exc, "calculation")
                    else:
                        _finish(item_root, records[index], output, started)
                    _checkpoint_batch(root)
        batch = _checkpoint_batch(root)
        states = [item["state"] for item in batch.items]
        state = (
            "completed"
            if all(s == "completed" for s in states)
            else "partial"
            if "completed" in states
            else "failed"
        )
        return _checkpoint_batch(root, state)
    except BaseException as exc:
        _interrupt_batch(root, exc)
        raise


def run_batch(requests, *, output_dir, execution=None):
    requests = [parse_request(request) for request in requests]
    if not requests:
        raise ValueError("A batch requires at least one request")
    comparison = requests[0].model_dump(exclude={"structure"})
    if any(
        r.model_dump(exclude={"structure"}) != comparison for r in requests[1:]
    ):
        raise ValueError(
            "Batches require homogeneous operations, models and settings"
        )
    config = ExecutionConfig.model_validate(execution or {})
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with claim(root):
        if any(p.name != ".matkit.lock" for p in root.iterdir()):
            raise FileExistsError("Use a fresh batch output directory")
        items = [
            {
                "index": i,
                "bundle": f"{i:05d}",
                "state": "pending",
                "accepted": False,
                "failure": None,
            }
            for i in range(len(requests))
        ]
        atomic_json(
            root / "batch_manifest.json",
            BatchResult(state="running", items=items),
        )
        try:
            for i, request in enumerate(requests):
                try:
                    prepare(request, output_dir=root / items[i]["bundle"])
                    (root / items[i]["bundle"] / ".matkit.batch").write_text(
                        "1"
                    )
                    items[i]["state"] = "prepared"
                except Exception as exc:
                    items[i].update(
                        state="failed",
                        failure={
                            "code": type(exc).__name__,
                            "stage": "preparation",
                            "message": str(exc),
                        },
                    )
                atomic_json(
                    root / "batch_manifest.json",
                    BatchResult(state="running", items=items),
                )
            if (
                config.mode == "subprocess"
                or config.python
                or config.environment
                or config.timeout_s
            ):
                return _supervise_claimed(root, config, batch=True)
            return _execute_batch_claimed(root, config)
        except BaseException as exc:
            _interrupt_batch(root, exc)
            raise
