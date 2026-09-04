"""Plain-Python execution core for runtime-selectable MLIPs."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import read as ase_read

from matkit.mlip.config import (
    ASEMACEConfig,
    MLIPBackendConfig,
    MLIPCalculationConfig,
    NVAlchemiMACEConfig,
    RootstockConfig,
    _positive_integer,
)
from matkit.types import MLIPBatchSummary, MLIPResult


def _finite_array(name: str, value, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have shape {shape} and finite values")
    return array


def _validate_atoms(atoms) -> None:
    if not len(atoms):
        raise ValueError("Structure must contain at least one atom")
    _finite_array("positions", atoms.positions, (len(atoms), 3))
    cell = _finite_array("cell", atoms.cell.array, (3, 3))
    periodic = cell[atoms.pbc]
    if len(periodic) and np.linalg.matrix_rank(periodic) != len(periodic):
        raise ValueError(
            "Periodic cell vectors must be nonzero and independent"
        )


def _atoms_payload(atoms) -> dict[str, Any]:
    """Convert the portable portion of an ASE Atoms object to JSON data."""
    return {
        "atomic_numbers": atoms.get_atomic_numbers().tolist(),
        "positions": atoms.get_positions().tolist(),
        "cell": atoms.cell.array.tolist(),
        "pbc": atoms.pbc.tolist(),
    }


def _optional_stress(atoms):
    if not atoms.pbc.any() or atoms.cell.rank != 3:
        return None
    try:
        return atoms.get_stress(voigt=False)
    except PropertyNotImplementedError:
        return None


def _synchronize_device(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for requested device {device}")
    torch.cuda.synchronize(torch.device(device))


def _create_mace_calculator(config: ASEMACEConfig):
    try:
        import mace.calculators as mace_calculators
    except ImportError as exc:
        raise ImportError(
            "Direct MACE requires the 'mlip' extra: pip install matkit[mlip]"
        ) from exc

    try:
        factory = getattr(mace_calculators, config.calculator_type)
    except AttributeError as exc:
        raise ImportError(
            f"Installed mace-torch does not provide {config.calculator_type}."
        ) from exc

    if config.calculator_type == "mace_anicc":
        if config.dispersion:
            raise ValueError(
                "Dispersion options are supported only by "
                "calculator_type='mace_mp'."
            )
        return factory(device=config.device, model_path=config.checkpoint)

    kwargs: dict[str, Any] = {
        "model": config.checkpoint,
        "device": config.device,
        "default_dtype": config.dtype,
    }
    if config.calculator_type == "mace_mp":
        kwargs["dispersion"] = config.dispersion
        if config.dispersion:
            kwargs.update(
                {
                    "damping": config.damping,
                    "dispersion_xc": config.dispersion_xc,
                    "dispersion_cutoff": config.dispersion_cutoff,
                }
            )
    elif config.dispersion:
        raise ValueError(
            "Dispersion options are supported only by calculator_type="
            "'mace_mp'."
        )
    return factory(**kwargs)


@contextmanager
def _ase_backend_context(
    config: ASEMACEConfig | RootstockConfig,
) -> Iterator[Any]:
    """Create one ASE calculator and retain it for the whole request."""
    if isinstance(config, ASEMACEConfig):
        yield _create_mace_calculator(config)
        return

    try:
        from rootstock import RootstockCalculator
    except ImportError as exc:
        raise ImportError(
            "Rootstock requires its optional extra: "
            "pip install matkit[rootstock]"
        ) from exc

    kwargs = {
        "checkpoint": config.checkpoint,
        "cluster": config.cluster,
        "root": config.root,
        "cache_root": config.cache_root,
        "device": config.device,
        "setup_kwargs": config.setup_kwargs,
        "timeout": config.timeout,
        "weights": config.weights,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    with RootstockCalculator(**kwargs) as calculator:
        yield calculator


def _optimizer_class(name: str):
    from ase.optimize import BFGS, FIRE, GPMin, LBFGS, MDMin

    return {
        "bfgs": BFGS,
        "lbfgs": LBFGS,
        "gpmin": GPMin,
        "fire": FIRE,
        "mdmin": MDMin,
    }[name]


def _success_result(
    input_file: str,
    backend: MLIPBackendConfig,
    calculation: MLIPCalculationConfig,
    atoms,
    energy: float,
    forces,
    stress,
    converged: bool,
    n_steps: int | None,
    calculation_time: float,
) -> dict[str, Any]:
    _validate_atoms(atoms)
    energy = _finite_array("energy", energy, ()).item()
    forces = _finite_array("forces", forces, (len(atoms), 3))
    if stress is not None:
        stress = _finite_array("stress", stress, (3, 3))
    return {
        "schema_version": 1,
        "success": True,
        "error": "",
        "input_structure_file": input_file,
        "backend_info": backend.to_dict(),
        "calculation_input": calculation.to_dict(),
        "energy": float(energy),
        "energy_unit": "eV",
        "forces": None if forces is None else forces.tolist(),
        "force_unit": "eV/angstrom",
        "stress": None if stress is None else stress.tolist(),
        "stress_unit": "eV/angstrom^3",
        "converged": bool(converged),
        "n_steps": n_steps,
        "final_structure": _atoms_payload(atoms),
        "calculation_time_s": calculation_time,
    }


def _failure_result(
    input_file: str,
    backend: MLIPBackendConfig,
    calculation: MLIPCalculationConfig,
    error: Exception | str,
    calculation_time: float = 0.0,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "success": False,
        "error": str(error),
        "input_structure_file": input_file,
        "backend_info": backend.to_dict(),
        "calculation_input": calculation.to_dict(),
        "energy": None,
        "energy_unit": "eV",
        "forces": None,
        "force_unit": "eV/angstrom",
        "stress": None,
        "stress_unit": "eV/angstrom^3",
        "converged": False,
        "n_steps": None,
        "final_structure": None,
        "calculation_time_s": calculation_time,
    }


def _run_ase_item(
    input_file: str,
    atoms,
    calculator,
    backend: ASEMACEConfig | RootstockConfig,
    calculation: MLIPCalculationConfig,
) -> dict[str, Any]:
    atoms.calc = calculator
    _synchronize_device(backend.device)
    started = time.perf_counter()
    converged = True
    n_steps = 0
    if calculation.driver == "opt":
        optimizer = _optimizer_class(calculation.optimizer)(atoms, logfile=None)
        converged = bool(
            optimizer.run(fmax=calculation.fmax, steps=calculation.steps)
        )
        n_steps = optimizer.nsteps

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = _optional_stress(atoms)
    _synchronize_device(backend.device)
    elapsed = time.perf_counter() - started
    return _success_result(
        input_file,
        backend,
        calculation,
        atoms,
        energy,
        forces,
        stress,
        converged,
        n_steps,
        elapsed,
    )


def _load_nvalchemi_model(config: NVAlchemiMACEConfig):
    try:
        import torch
        from nvalchemi.models.mace import MACEWrapper
    except ImportError as exc:
        raise ImportError(
            "NVIDIA ALCHEMI MACE requires the 'nvalchemi_mace' "
            "extra and a matching CUDA extra."
        ) from exc

    model = MACEWrapper.from_checkpoint(
        config.checkpoint,
        device=torch.device(config.device),
        dtype=getattr(torch, config.dtype),
        enable_cueq=config.enable_cueq,
        compile_model=config.compile_model,
    )
    model.eval()
    return model


def _atoms_to_nvalchemi_data(atoms, config: NVAlchemiMACEConfig):
    try:
        import torch
        from nvalchemi.data import AtomicData
    except ImportError as exc:
        raise ImportError(
            "NVIDIA ALCHEMI MACE requires the 'nvalchemi_mace' "
            "extra and a matching CUDA extra."
        ) from exc

    dtype = getattr(torch, config.dtype)
    data = AtomicData.from_atoms(
        atoms,
        device=torch.device(config.device),
        dtype=dtype,
    )
    data.forces = torch.zeros(
        data.num_nodes, 3, device=data.device, dtype=dtype
    )
    data.energy = torch.zeros(1, 1, device=data.device, dtype=dtype)
    data.velocities = torch.zeros(
        data.num_nodes, 3, device=data.device, dtype=dtype
    )
    return data


def _nvalchemi_result(
    input_file: str,
    original_atoms,
    data,
    backend: NVAlchemiMACEConfig,
    calculation: MLIPCalculationConfig,
    converged: bool,
    started: float,
) -> dict[str, Any]:
    final_atoms = original_atoms.copy()
    final_atoms.positions = data.positions.detach().cpu().numpy()
    if data.cell is not None:
        final_atoms.cell = data.cell.squeeze(0).detach().cpu().numpy()
    if data.pbc is not None:
        final_atoms.pbc = data.pbc.squeeze(0).detach().cpu().numpy()

    energy_values = data.energy.detach().cpu().numpy()
    if energy_values.size != 1:
        raise ValueError("ALCHEMI must return one energy per structure")
    energy = energy_values.reshape(()).item()
    forces = None
    if data.forces is not None:
        forces = data.forces.detach().cpu().numpy()
    stress = None
    if data.stress is not None:
        stress = data.stress.detach().cpu().numpy()
        if stress.shape == (1, 3, 3):
            stress = stress[0]
    return _success_result(
        input_file,
        backend,
        calculation,
        final_atoms,
        energy,
        forces,
        stress,
        converged,
        None if calculation.driver == "opt" else 0,
        time.perf_counter() - started,
    )


def _run_nvalchemi_chunk(
    model,
    entries: Sequence[tuple[int, str, Any]],
    backend: NVAlchemiMACEConfig,
    calculation: MLIPCalculationConfig,
) -> list[tuple[int, dict[str, Any]]]:
    try:
        from nvalchemi.data import Batch
        from nvalchemi.dynamics import BaseDynamics, ConvergenceHook, FIRE
    except ImportError as exc:
        raise ImportError(
            "NVIDIA ALCHEMI MACE requires the 'nvalchemi_mace' "
            "extra and a matching CUDA extra."
        ) from exc

    started = time.perf_counter()
    data_list = [
        _atoms_to_nvalchemi_data(atoms, backend) for _, _, atoms in entries
    ]
    batch = Batch.from_data_list(data_list)
    hooks = model.make_neighbor_hooks()
    convergence = None
    if calculation.driver == "energy":
        dynamics = BaseDynamics(model=model, hooks=hooks, n_steps=1)
    else:
        convergence = ConvergenceHook.from_fmax(calculation.fmax)
        dynamics = FIRE(
            model=model,
            dt=backend.dt,
            hooks=hooks,
            convergence_hook=convergence,
            n_steps=calculation.steps,
        )

    _synchronize_device(backend.device)
    with dynamics:
        batch = dynamics.run(batch)
    _synchronize_device(backend.device)

    converged_indices = set(range(len(entries)))
    if convergence is not None:
        indices = convergence.evaluate(batch)
        converged_indices = (
            set() if indices is None else set(indices.detach().cpu().tolist())
        )

    return [
        (
            original_index,
            _nvalchemi_result(
                input_file,
                atoms,
                data,
                backend,
                calculation,
                chunk_index in converged_indices,
                started,
            ),
        )
        for chunk_index, ((original_index, input_file, atoms), data) in (
            enumerate(zip(entries, batch.to_data_list()))
        )
    ]


def _chunks_by_capacity(
    entries: Sequence[tuple[int, str, Any]],
    batch_size: int,
    max_atoms: int | None,
) -> Iterator[list[tuple[int, str, Any]]]:
    chunk: list[tuple[int, str, Any]] = []
    atom_count = 0
    for entry in entries:
        n_atoms = len(entry[2])
        exceeds_atoms = (
            max_atoms is not None
            and bool(chunk)
            and atom_count + n_atoms > max_atoms
        )
        if len(chunk) >= batch_size or exceeds_atoms:
            yield chunk
            chunk = []
            atom_count = 0
        chunk.append(entry)
        atom_count += n_atoms
    if chunk:
        yield chunk


def _read_inputs(
    input_files: Sequence[str | Path],
    backend: MLIPBackendConfig,
    calculation: MLIPCalculationConfig,
) -> tuple[list[tuple[int, str, Any]], list[dict[str, Any] | None]]:
    prepared = []
    results: list[dict[str, Any] | None] = [None] * len(input_files)
    for index, value in enumerate(input_files):
        input_file = str(Path(value).expanduser().resolve())
        try:
            if not Path(input_file).is_file():
                raise FileNotFoundError(
                    f"Input structure file does not exist: {value}"
                )
            atoms = ase_read(input_file)
            _validate_atoms(atoms)
            prepared.append((index, input_file, atoms))
        except Exception as exc:
            results[index] = _failure_result(
                input_file, backend, calculation, exc
            )
    return prepared, results


def _execute_inputs(
    input_files: Sequence[str | Path],
    backend: MLIPBackendConfig,
    calculation: MLIPCalculationConfig,
    batch_size: int,
    max_atoms: int | None,
) -> tuple[list[dict[str, Any]], float]:
    if not input_files:
        raise ValueError("At least one input structure file is required")
    _positive_integer("batch_size", batch_size)
    if max_atoms is not None:
        _positive_integer("max_atoms", max_atoms)
    if (
        isinstance(backend, NVAlchemiMACEConfig)
        and calculation.driver == "opt"
        and calculation.optimizer != "fire"
    ):
        raise ValueError("NVIDIA ALCHEMI supports only the FIRE optimizer")

    prepared, results = _read_inputs(input_files, backend, calculation)
    if not prepared:
        return [result for result in results if result is not None], 0.0

    setup_started = time.perf_counter()
    if isinstance(backend, (ASEMACEConfig, RootstockConfig)):
        try:
            with _ase_backend_context(backend) as calculator:
                _synchronize_device(backend.device)
                setup_time = time.perf_counter() - setup_started
                for index, input_file, atoms in prepared:
                    try:
                        results[index] = _run_ase_item(
                            input_file,
                            atoms,
                            calculator,
                            backend,
                            calculation,
                        )
                    except Exception as exc:
                        results[index] = _failure_result(
                            input_file, backend, calculation, exc
                        )
        except Exception as exc:
            setup_time = time.perf_counter() - setup_started
            for index, input_file, _ in prepared:
                results[index] = _failure_result(
                    input_file, backend, calculation, exc
                )
    else:
        try:
            model = _load_nvalchemi_model(backend)
            _synchronize_device(backend.device)
            setup_time = time.perf_counter() - setup_started
            for chunk in _chunks_by_capacity(prepared, batch_size, max_atoms):
                try:
                    chunk_results = _run_nvalchemi_chunk(
                        model, chunk, backend, calculation
                    )
                    if len(chunk_results) != len(chunk):
                        raise RuntimeError(
                            "NVIDIA ALCHEMI returned a different number of "
                            "results than input structures."
                        )
                    for index, result in chunk_results:
                        results[index] = result
                except Exception as exc:
                    for index, input_file, _ in chunk:
                        results[index] = _failure_result(
                            input_file, backend, calculation, exc
                        )
        except Exception as exc:
            setup_time = time.perf_counter() - setup_started
            for index, input_file, _ in prepared:
                results[index] = _failure_result(
                    input_file, backend, calculation, exc
                )

    final_results = []
    for result in results:
        if result is None:
            raise RuntimeError("Internal MLIP result ordering error")
        final_results.append(result)
    return final_results, setup_time


def _write_json(path: Path, data: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False), encoding="utf-8"
    )
    return str(path.resolve())


def run_mlip(
    input_file: str | Path,
    backend: MLIPBackendConfig,
    calculation: MLIPCalculationConfig | None = None,
    output_file: str | Path | None = None,
) -> MLIPResult:
    """Run one energy calculation or fixed-cell optimization."""
    calculation = calculation or MLIPCalculationConfig()
    results, setup_time = _execute_inputs(
        [input_file], backend, calculation, batch_size=1, max_atoms=None
    )
    result = results[0]
    result["setup_time_s"] = setup_time
    if output_file is not None:
        path = Path(output_file).expanduser().resolve()
        result["output_results_file"] = str(path)
        _write_json(path, result)
    return result


def run_mlip_batch(
    input_files: Sequence[str | Path],
    backend: MLIPBackendConfig,
    calculation: MLIPCalculationConfig | None = None,
    output_dir: str | Path = "mlip_results",
    batch_size: int = 16,
    max_atoms: int | None = None,
) -> MLIPBatchSummary:
    """Run an ordered MLIP batch and persist results plus a manifest."""
    calculation = calculation or MLIPCalculationConfig()
    started = time.perf_counter()
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    results, setup_time = _execute_inputs(
        input_files,
        backend,
        calculation,
        batch_size=batch_size,
        max_atoms=max_atoms,
    )

    items = []
    for index, result in enumerate(results):
        stem = Path(result["input_structure_file"]).stem
        result_path = output_path / f"{index:05d}_{stem}.json"
        result["setup_time_s"] = setup_time
        result_file = _write_json(result_path, result)
        items.append(
            {
                "index": index,
                "input_structure_file": result["input_structure_file"],
                "status": "success" if result["success"] else "failure",
                "result_file": result_file,
                "error": result["error"],
            }
        )

    succeeded = sum(item["status"] == "success" for item in items)
    failed = len(items) - succeeded
    if items and failed == 0:
        status = "completed"
    elif succeeded:
        status = "partial"
    else:
        status = "failure"
    manifest = {
        "schema_version": 1,
        "status": status,
        "backend_info": backend.to_dict(),
        "calculation_input": calculation.to_dict(),
        "setup_time_s": setup_time,
        "wall_time_s": time.perf_counter() - started,
        "total": len(items),
        "succeeded": succeeded,
        "failed": failed,
        "items": items,
    }
    manifest_file = _write_json(output_path / "batch_manifest.json", manifest)
    return {
        **manifest,
        "manifest_file": manifest_file,
        "results": results,
    }
