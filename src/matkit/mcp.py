"""Optional stdio tools over MatKit's scientific API and supervised workers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
from uuid import uuid4

import click
from pydantic import Field, JsonValue

from matkit.api import (
    AdsorptionRequest,
    EvaluateRequest,
    ExecutionConfig,
    PoreRequest,
    RelaxRequest,
    inspect_run,
    list_capabilities,
    prepare,
)
from matkit.api.bundles import (
    artifact,
    atomic_json,
    claim,
    collect_artifacts,
    commit_result,
    contained_path,
)
from matkit.api.models import Artifact, Failure, Model, ScientificCheck
from matkit.api.runtime import _fail, _worker_command
from matkit.api.structures import sha256
from matkit.operation_cli import resolve_request_paths


class ArtifactLink(Artifact):
    uri: str


class ToolResult(Model):
    run_id: str
    state: str
    numerical_validity: str
    accepted: bool
    checks: list[ScientificCheck]
    summary: dict[str, JsonValue] = Field(default_factory=dict)
    artifacts: list[ArtifactLink]
    failure: Failure | None = None


def _links(root, record):
    inventory = [*record.artifacts]
    for name in ("run.json", "result.json"):
        if (root / name).is_file():
            inventory.append(artifact(root, root / name, "record"))
    return [
        ArtifactLink(
            **ref.model_dump(),
            uri=f"matkit://runs/{record.run_id}/artifacts/{ref.sha256}",
        )
        for ref in inventory
    ]


def _summary(root):
    record = inspect_run(root)
    summary = {}
    if record.payload:
        values = record.payload.model_dump(mode="json")
        summary = {
            key: value
            for key, value in values.items()
            if key not in {"forces", "stress", "results"}
        }
        if "results" in values:
            summary["results"] = {
                name: {
                    key: value
                    for key, value in result.items()
                    if not isinstance(value, list)
                }
                for name, result in values["results"].items()
            }
    return ToolResult(
        run_id=record.run_id,
        state=record.state,
        numerical_validity=record.numerical_validity,
        accepted=record.accepted,
        checks=record.checks,
        summary=summary,
        artifacts=_links(root, record),
        failure=record.failure,
    )


async def _execute_bounded(root, execution):
    import anyio

    with claim(root):
        # Environment values travel through the process environment, not through
        # artifacts retrievable by MCP clients.
        atomic_json(
            root / "execution.json",
            execution.model_copy(update={"environment": {}}),
        )
        with (
            (root / "worker.stdout.log").open("w") as stdout,
            (root / "worker.stderr.log").open("w") as stderr,
        ):
            process = None
            try:
                process = await anyio.open_process(
                    _worker_command(root, execution, False),
                    env={
                        **os.environ,
                        **execution.environment,
                        "MATKIT_WORKER_PROCESS": "1",
                    },
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=os.name == "posix",
                )
                with anyio.fail_after(execution.timeout_s):
                    code = await process.wait()
                record = inspect_run(root)
                if record.state in {"prepared", "running"}:
                    _fail(
                        root,
                        record,
                        RuntimeError(
                            f"Worker exited with code {code}; "
                            "see worker.stderr.log"
                        ),
                        "worker",
                        interrupted=True,
                    )
            except BaseException as exc:
                with anyio.CancelScope(shield=True):
                    if process is not None and process.returncode is None:
                        if os.name == "posix":
                            os.killpg(process.pid, signal.SIGTERM)
                        else:
                            process.terminate()
                        with anyio.move_on_after(5) as grace:
                            await process.wait()
                        if grace.cancel_called:
                            if os.name == "posix":
                                os.killpg(process.pid, signal.SIGKILL)
                            else:
                                process.kill()
                            await process.wait()
                    if not (root / "result.json").exists():
                        _fail(
                            root,
                            inspect_run(root),
                            exc,
                            "worker",
                            interrupted=True,
                        )
                if not isinstance(exc, (TimeoutError, OSError)):
                    raise
            finally:
                if process is not None:
                    with anyio.CancelScope(shield=True):
                        if os.name == "posix":
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        await process.aclose()
                        record = inspect_run(root)
                        if record.state not in {"prepared", "running"}:
                            commit_result(
                                root,
                                record.model_copy(
                                    update={
                                        "artifacts": collect_artifacts(root)
                                    }
                                ),
                            )
        record = inspect_run(root)
        if record.state not in {"prepared", "running"}:
            commit_result(
                root,
                record.model_copy(
                    update={"artifacts": collect_artifacts(root)}
                ),
            )
    return _summary(root)


def create_server(
    *, run_root, input_roots, profiles=None, timeout_s=60.0, tools=None
):
    from mcp.server import MCPServer

    run_root = Path(run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    roots = [Path(path).expanduser().resolve() for path in input_roots]
    if not roots:
        raise ValueError("At least one input root is required")
    profile_map = {
        name: ExecutionConfig.model_validate(value)
        for name, value in (profiles or {"default": {}}).items()
    }
    if not profile_map:
        raise ValueError("At least one execution profile is required")
    ExecutionConfig(timeout_s=timeout_s)
    server = MCPServer("MatKit")

    def allowed(path, *, output=False):
        path = Path(path).expanduser().resolve()
        locations = [run_root] if output else [*roots, run_root]
        if not any(path.is_relative_to(base) for base in locations):
            raise ValueError(
                "Input path is outside the server's configured roots"
            )
        return path

    def run_directory(run_id):
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("Invalid run identifier")
        return contained_path(run_root, run_id)

    def prepare_tool(request):
        values = resolve_request_paths(
            request.model_dump(mode="json"), roots[0]
        )
        allowed(values["structure"]["path"])
        sidecar = Path(values["structure"]["path"]).with_suffix(
            ".metadata.json"
        )
        if sidecar.exists():
            allowed(sidecar)
        for key in ("radii_file", "template_dir"):
            if values.get(key):
                allowed(values[key])
        checkpoint = values.get("method", {}).get("checkpoint")
        if checkpoint and (
            Path(checkpoint).is_absolute() or Path(checkpoint).exists()
        ):
            allowed(checkpoint)
        adapter = values.get("adapter")
        if isinstance(adapter, dict):
            for key in ("root", "weights"):
                if adapter.get(key):
                    allowed(adapter[key])
            if adapter.get("cache_root"):
                allowed(adapter["cache_root"], output=True)
        temporary = run_root / uuid4().hex
        record = prepare(values, output_dir=temporary)
        destination = run_directory(record.run_id)
        temporary.rename(destination)
        return destination

    async def calculate(request, profile):
        if profile not in profile_map:
            raise ValueError(f"Unknown execution profile: {profile}")
        config = profile_map[profile]
        timeout = (
            min(timeout_s, config.timeout_s) if config.timeout_s else timeout_s
        )
        config = config.model_copy(
            update={"mode": "subprocess", "timeout_s": timeout}
        )
        root = prepare_tool(request)
        return await _execute_bounded(root, config)

    async def matkit_evaluate(
        request: EvaluateRequest, profile: str = "default"
    ) -> ToolResult:
        """Evaluate requested energy/forces/stress.

        Check model/species support. Use prepared bundles and CLI execution
        for long runs.
        """
        return await calculate(request, profile)

    async def matkit_relax(
        request: RelaxRequest, profile: str = "default"
    ) -> ToolResult:
        """Relax positions at fixed cell.

        Check force_convergence before using the geometry and recompute
        invalidated charges/properties.
        """
        return await calculate(request, profile)

    async def matkit_pores(
        request: PoreRequest, profile: str = "default"
    ) -> ToolResult:
        """Analyze a periodic structure using Zeo++.

        Choose radii/probe settings explicitly. Execution does not establish
        sampling accuracy.
        """
        return await calculate(request, profile)

    def matkit_prepare_adsorption(request: AdsorptionRequest) -> ToolResult:
        """Stage a charged periodic CIF for single-component gRASPA CUDA.

        Execute the bundle with MatKit CLI inside an allocation; this tool
        does not submit a job.
        """
        return _summary(prepare_tool(request))

    def matkit_prepare(
        request: EvaluateRequest | RelaxRequest | PoreRequest,
    ) -> ToolResult:
        """Prepare a long calculation for later CLI execution.

        Preparation does not load models or execution engines.
        """
        return _summary(prepare_tool(request))

    def matkit_inspect(run_id: str) -> ToolResult:
        """Inspect a prepared/executed run.

        Check scientific outcomes separately from execution state;
        artifacts are retrievable MCP resources.
        """
        return _summary(run_directory(run_id))

    def matkit_capabilities() -> dict:
        """Discover implementations, requirements and execution profiles.

        MOFforge owns construction and structural edits.
        """
        return {
            "capabilities": list_capabilities(),
            "profiles": list(profile_map),
            "timeout_s": timeout_s,
        }

    catalog = {
        function.__name__: function
        for function in (
            matkit_evaluate,
            matkit_relax,
            matkit_pores,
            matkit_prepare_adsorption,
            matkit_prepare,
            matkit_inspect,
            matkit_capabilities,
        )
    }
    selected = set(tools) if tools is not None else set(catalog)
    if not selected <= catalog.keys():
        raise ValueError(f"Unknown tools: {sorted(selected - catalog.keys())}")
    for name in sorted(selected):
        server.tool()(catalog[name])

    @server.resource(
        "matkit://runs/{run_id}/artifacts/{digest}",
        mime_type="application/octet-stream",
    )
    def read_artifact(run_id: str, digest: str) -> bytes:
        root = run_directory(run_id)
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Invalid artifact hash")
        for ref in _links(root, inspect_run(root)):
            if ref.sha256 == digest:
                path = contained_path(root, ref.path)
                if sha256(path) != digest:
                    raise ValueError("Artifact changed since it was recorded")
                return path.read_bytes()
        raise ValueError("Artifact is not in the run manifest")

    return server


@click.command()
@click.option("--run-root", required=True, type=click.Path(path_type=Path))
@click.option(
    "--input-root",
    multiple=True,
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--profiles", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--timeout", type=click.FloatRange(min=0, min_open=True), default=60.0
)
@click.option("--tools", help="Comma-separated tool allowlist.")
def main(run_root, input_root, profiles, timeout, tools):
    """Serve selected MatKit tools over stdio (requires matkit[mcp])."""
    try:
        profile_map = json.loads(profiles.read_text()) if profiles else None
        server = create_server(
            run_root=run_root,
            input_roots=input_root,
            profiles=profile_map,
            timeout_s=timeout,
            tools=tools.split(",") if tools is not None else None,
        )
        server.run(transport="stdio")
    except ImportError as exc:
        raise click.ClickException(
            "Install matkit[mcp] to run the MCP server"
        ) from exc
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
