"""Thin Click commands over the shared scientific API."""

from pathlib import Path
import json

import click


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise click.UsageError(
            f"Cannot read JSON specification: {exc}"
        ) from exc


def resolve_request_paths(value, base):
    """Resolve inputs relative to their spec; keep model aliases intact."""
    if not isinstance(value, dict):
        raise ValueError("Expected a request object")
    value = json.loads(json.dumps(value))

    def path(container, key, optional=False):
        if container.get(key):
            candidate = Path(container[key]).expanduser()
            if not candidate.is_absolute():
                candidate = base / candidate
            if not optional or candidate.is_file():
                container[key] = str(candidate.resolve())

    if isinstance(value.get("structure"), dict):
        path(value["structure"], "path")
    for key in ("radii_file", "template_dir"):
        path(value, key)
    if isinstance(value.get("method"), dict):
        path(value["method"], "checkpoint", optional=True)
    if isinstance(value.get("adapter"), dict):
        for key in ("root", "cache_root", "weights"):
            path(value["adapter"], key)
    return value


def execution_profile(path):
    from matkit.api import ExecutionConfig

    values = read_json(path) if path else {}
    if not isinstance(values, dict):
        raise click.UsageError("Execution profile must be a JSON object")
    # CLI engines always run in a worker, keeping engine stdout out of JSON.
    return ExecutionConfig.model_validate({**values, "mode": "subprocess"})


def invoke(function, *args, prepared=False, **kwargs):
    try:
        result = function(*args, **kwargs)
    except (ValueError, TypeError, FileNotFoundError, FileExistsError) as exc:
        raise click.UsageError(str(exc)) from exc
    except OSError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.model_dump_json(indent=2))
    if prepared:
        return
    accepted = (
        result.accepted
        if hasattr(result, "accepted")
        else all(item["accepted"] for item in result.items)
    )
    if not accepted:
        raise click.exceptions.Exit(1)


def register_commands(main):
    def operation_command(name, operation, preparation=False):
        @click.command(name)
        @click.option(
            "--spec",
            required=True,
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
        )
        @click.option(
            "--outdir", required=True, type=click.Path(path_type=Path)
        )
        @click.option(
            "--execution",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
        )
        def command(spec, outdir, execution):
            from matkit.api import parse_request, prepare, run

            try:
                data = resolve_request_paths(
                    read_json(spec), spec.resolve().parent
                )
                data.setdefault("operation", operation)
                request = parse_request(data)
                if request.operation != operation:
                    raise ValueError(
                        f"This command requires operation={operation}"
                    )
                if preparation:
                    if execution is not None:
                        raise ValueError(
                            "Preparation does not use an execution profile"
                        )
                    invoke(prepare, request, output_dir=outdir, prepared=True)
                else:
                    invoke(
                        run,
                        request,
                        output_dir=outdir,
                        execution=execution_profile(execution),
                    )
            except (ValueError, TypeError) as exc:
                raise click.UsageError(str(exc)) from exc

        return command

    for operation in ("evaluate", "relax", "pores"):
        main.add_command(operation_command(operation, operation))

    @main.group("adsorption")
    def adsorption():
        """Prepare, run, or collect single-component gRASPA calculations."""

    adsorption.add_command(operation_command("prepare", "adsorption", True))
    adsorption.add_command(operation_command("run", "adsorption"))

    @adsorption.command("analyze")
    @click.argument(
        "bundle", type=click.Path(exists=True, file_okay=False, path_type=Path)
    )
    def analyze(bundle):
        from matkit.api import analyze_adsorption

        invoke(analyze_adsorption, bundle)

    @main.command("prepare")
    @click.option(
        "--spec",
        required=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
    )
    @click.option("--outdir", required=True, type=click.Path(path_type=Path))
    def prepare_command(spec, outdir):
        """Stage any operation without loading its execution engine."""
        from matkit.api import prepare

        try:
            request = resolve_request_paths(
                read_json(spec), spec.resolve().parent
            )
            invoke(prepare, request, output_dir=outdir, prepared=True)
        except (ValueError, TypeError) as exc:
            raise click.UsageError(str(exc)) from exc

    @main.command("execute")
    @click.argument(
        "bundle", type=click.Path(exists=True, file_okay=False, path_type=Path)
    )
    @click.option(
        "--execution",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
    )
    def execute_command(bundle, execution):
        """Execute a prepared bundle in this allocation/environment."""
        from matkit.api import execute

        try:
            invoke(execute, bundle, execution=execution_profile(execution))
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc

    @main.command("batch")
    @click.option(
        "--spec",
        required=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
    )
    @click.option("--outdir", required=True, type=click.Path(path_type=Path))
    @click.option(
        "--execution",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
    )
    def batch_command(spec, outdir, execution):
        """Execute a JSON list of homogeneous requests with calculator reuse."""
        from matkit.api import run_batch

        try:
            values = read_json(spec)
            if not isinstance(values, list):
                raise ValueError("Batch specification must be a list")
            requests = [
                resolve_request_paths(value, spec.resolve().parent)
                for value in values
            ]
            invoke(
                run_batch,
                requests,
                output_dir=outdir,
                execution=execution_profile(execution),
            )
        except (ValueError, TypeError) as exc:
            raise click.UsageError(str(exc)) from exc

    @main.command("inspect")
    @click.argument(
        "bundle", type=click.Path(exists=True, file_okay=False, path_type=Path)
    )
    def inspect_command(bundle):
        """Read committed results without rerunning a calculation."""
        from matkit.api import inspect_run
        from matkit.api.runtime import read_batch

        function = (
            read_batch
            if (bundle / "batch_manifest.json").exists()
            else inspect_run
        )
        # Inspection succeeding is independent of the calculation's outcome.
        invoke(function, bundle, prepared=True)

    @main.command("capabilities")
    @click.option(
        "--json", "as_json", is_flag=True, help="Print JSON (also the default)."
    )
    def capabilities(as_json):
        """List implementations, availability, restrictions and evidence."""
        from matkit.api import list_capabilities

        click.echo(json.dumps(list_capabilities(), indent=2, allow_nan=False))
