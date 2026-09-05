"""Small adapters over the maintained calculator and engine implementations."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import signal
import subprocess

import numpy as np

from .models import (
    AdsorptionPayload,
    AdsorptionRequest,
    AlchemiAdapter,
    EvaluationPayload,
    EvaluateRequest,
    MACEAdapter,
    PorePayload,
    PoreRequest,
    RelaxRequest,
    ScientificCheck,
)
from .structures import final_structure, load_structure, sha256


def backend_config(request, execution):
    from matkit.mlip.config import (
        ASEMACEConfig,
        NVAlchemiMACEConfig,
        RootstockConfig,
    )

    method = request.method
    adapter = request.adapter
    device = execution.device or (
        "cuda" if isinstance(adapter, AlchemiAdapter) else "cpu"
    )
    if isinstance(adapter, MACEAdapter):
        return ASEMACEConfig(
            **method.model_dump(),
            device=device,
            dtype=adapter.dtype or "float64",
        )
    if isinstance(adapter, AlchemiAdapter):
        return NVAlchemiMACEConfig(
            checkpoint=method.checkpoint,
            device=device,
            dtype=adapter.dtype,
            dt=adapter.dt or 0.1,
            compile_model=adapter.compile_model,
            enable_cueq=adapter.enable_cueq,
        )
    return RootstockConfig(
        checkpoint=method.checkpoint,
        device=device,
        **adapter.model_dump(exclude={"type"}),
    )


def calculation_config(request):
    from matkit.mlip.config import MLIPCalculationConfig

    if isinstance(request, RelaxRequest):
        return MLIPCalculationConfig(
            driver="opt",
            optimizer=request.optimizer,
            fmax=request.fmax,
            steps=request.steps,
        )
    return MLIPCalculationConfig()


@contextmanager
def calculator_session(request, execution):
    from matkit.mlip import runner

    backend = backend_config(request, execution)
    if isinstance(request.adapter, AlchemiAdapter):
        model = runner._load_nvalchemi_model(backend)
        yield backend, model
    else:
        with runner._ase_backend_context(backend) as calculator:
            yield backend, calculator


def evaluate_items(entries, request, execution, session):
    """Yield item results while keeping the calculator/model alive."""
    from matkit.mlip import runner

    backend, calculator = session
    calculation = calculation_config(request)
    if isinstance(request.adapter, AlchemiAdapter):
        chunks = runner._chunks_by_capacity(
            entries, request.adapter.batch_size, request.adapter.max_atoms
        )
        for chunk in chunks:
            try:
                results = runner._run_nvalchemi_chunk(
                    calculator, chunk, backend, calculation
                )
                if [i for i, _ in results] != [i for i, _, _ in chunk]:
                    raise ValueError(
                        "ALCHEMI returned inconsistent result ordering"
                    )
            except Exception as exc:
                results = [(i, exc) for i, _, _ in chunk]
            yield from results
    else:
        properties = (
            request.properties if isinstance(request, EvaluateRequest) else None
        )
        for index, input_file, atoms in entries:
            try:
                result = runner._run_ase_item(
                    input_file,
                    atoms,
                    calculator,
                    backend,
                    calculation,
                    requested_properties=properties,
                )
            except Exception as exc:
                result = exc
            yield index, result


def calculator_payload(root, request, legacy):
    if isinstance(legacy, Exception):
        raise legacy
    if not legacy["success"]:
        raise ValueError(legacy["error"])
    original_atoms, structure, source_hash = load_structure(request.structure)
    properties = (
        request.properties
        if isinstance(request, EvaluateRequest)
        else ["potential_energy", "forces"]
    )
    names = {
        "potential_energy": "energy",
        "forces": "forces",
        "stress": "stress",
    }
    for name in properties:
        if legacy.get(names[name]) is None:
            raise ValueError(f"Requested property {name} is unavailable")
    final = legacy["final_structure"]
    atoms = original_atoms.copy()
    if final["atomic_numbers"] != atoms.numbers.tolist():
        raise ValueError("Calculator changed atom correspondence")
    atoms.positions = final["positions"]
    atoms.cell = final["cell"]
    atoms.pbc = final["pbc"]
    float32 = (
        isinstance(request.adapter, AlchemiAdapter)
        and request.adapter.dtype == "float32"
    )
    relative_tolerance = 1e-6 if float32 else 0
    absolute_tolerance = 1e-6 if float32 else 1e-7
    if not np.allclose(
        atoms.cell.array,
        original_atoms.cell.array,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    ) or not np.array_equal(atoms.pbc, original_atoms.pbc):
        raise ValueError("Fixed-cell calculation changed cell or periodicity")
    if isinstance(request, EvaluateRequest) and not np.allclose(
        atoms.positions,
        original_atoms.positions,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    ):
        raise ValueError("Property evaluation changed atomic positions")
    out = root / "work" / "final_structure.extxyz"
    final_structure(atoms, structure, source_hash, out)
    checks = []
    converged = None
    if isinstance(request, RelaxRequest):
        max_force = float(np.linalg.norm(legacy["forces"], axis=1).max())
        converged = (
            bool(legacy["converged"])
            and max_force <= request.fmax * (1 + 1e-6) + 1e-12
        )
        checks.append(
            ScientificCheck(
                name="force_convergence",
                status="passed" if converged else "failed",
                required=True,
                detail=(
                    f"Requested fmax={request.fmax} eV/angstrom; "
                    f"optimizer reports convergence={converged}"
                ),
            )
        )
    payload = EvaluationPayload(
        potential_energy=legacy.get("energy")
        if "potential_energy" in properties
        else None,
        forces=legacy.get("forces") if "forces" in properties else None,
        stress=legacy.get("stress") if "stress" in properties else None,
        converged=converged,
        n_steps=legacy.get("n_steps"),
        final_structure="work/final_structure.extxyz",
    )
    return payload, checks, {"calculation_s": legacy["calculation_time_s"]}


def stop_process(process, *, group=True):
    if group and os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if not (group and os.name == "posix"):
            process.kill()
    finally:
        if group and os.name == "posix":
            # A child can outlive a terminated group leader.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()


def external_command(root, execution, engine, arguments):
    from .bundles import atomic_json

    command = execution.executables.get(engine)
    if command is None:
        binary = shutil.which(
            "network" if engine == "zeopp" else "simulate",
            path=execution.environment.get("PATH"),
        )
        if binary is None:
            raise FileNotFoundError(
                f"Configure the {engine} executable in the execution profile"
            )
        command = [binary]
    binary = shutil.which(command[0], path=execution.environment.get("PATH"))
    if binary is None:
        raise FileNotFoundError(f"Executable unavailable: {command[0]}")
    command = [str(Path(binary).resolve()), *command[1:], *arguments]
    atomic_json(
        root / "command.json",
        {
            "argv": command,
            "cwd": "work",
            "executable_sha256": sha256(Path(command[0])),
            "argument_file_hashes": {
                token: sha256(candidate)
                for token in command
                if (candidate := Path(token)).is_file()
            },
        },
    )
    # Worker children share its process group so supervision covers the engine.
    own_group = os.environ.get("MATKIT_WORKER_PROCESS") != "1"
    log_name = "raspa.log" if engine == "graspa" else "engine.stdout.log"
    with (
        (root / "work" / log_name).open("w") as stdout,
        (root / "work" / "engine.stderr.log").open("w") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=root / "work",
            env={**os.environ, **execution.environment},
            stdout=stdout,
            stderr=stderr,
            start_new_session=own_group and os.name == "posix",
        )
        try:
            code = process.wait(timeout=execution.timeout_s)
        except BaseException:
            stop_process(process, group=own_group)
            raise
    atomic_json(root / "exit.json", {"returncode": code})
    if code != 0:
        raise RuntimeError(
            f"{engine} exited with code {code}; see work/engine.stderr.log"
        )


def run_pores(root, request, execution):
    from matkit.zeopp.zeopp import _analysis_arguments

    args = _analysis_arguments(
        request.analyses,
        request.probe_radius,
        request.channel_radius,
        request.num_samples,
        request.high_accuracy,
        request.radii_file,
        "structure.cif",
    )
    external_command(root, execution, "zeopp", args)
    return parse_pores(root, request)


def parse_pores(root, request):
    from matkit.zeopp.zeopp import _output_path, _parse_output

    results = {
        analysis: _parse_output(
            _output_path(root / "work", "structure", analysis), analysis
        )
        for analysis in request.analyses
    }
    return (
        PorePayload(results=results),
        [
            ScientificCheck(
                name="sampling_quality",
                status="unknown",
                detail="Execution does not establish sampling accuracy",
            )
        ],
        {},
    )


def parse_adsorption(root, request):
    from matkit.graspa import get_output_data

    data = get_output_data(
        str(root / "work"),
        unit=request.unit,
        eos=request.fugacity_coefficient == "PR-EOS",
    )
    if not data["success"]:
        raise ValueError("gRASPA output is incomplete")
    payload = AdsorptionPayload(
        component=request.adsorbate,
        uptake=data["uptake"],
        unit=data["unit"],
        uncertainty=data["error"],
        heat_of_adsorption=data["qst"],
        heat_uncertainty=data["error_qst"],
    )
    checks = [
        ScientificCheck(
            name="sampling_quality",
            status="unknown",
            detail=(
                "Engine statistics do not establish equilibration "
                "or independent samples"
            ),
        )
    ]
    return payload, checks, {"engine_reported_s": data["calc_time_in_s"]}


def run_external(root, request, execution):
    if isinstance(request, PoreRequest):
        return run_pores(root, request, execution)
    if isinstance(request, AdsorptionRequest):
        external_command(root, execution, "graspa", [])
        return parse_adsorption(root, request)
    raise TypeError("Not an external-engine request")
