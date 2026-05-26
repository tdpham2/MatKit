from pathlib import Path

from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.optimize import BFGS

try:
    from ase.filters import ExpCellFilter
except ImportError:
    try:
        from ase.constraints import ExpCellFilter
    except ImportError:
        from ase.filters import FrechetCellFilter as ExpCellFilter
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase import units


def _create_calculator(
    model: str = "uma-s-1p2",
    task_name: str = "omat",
    device: str = "cpu",
):
    """Create a FAIRChemCalculator from a pretrained UMA model.

    Args:
        model: UMA model name (e.g. 'uma-s-1p2', 'uma-m-1p1').
        task_name: Task head name ('omat', 'oc20', 'oc22', 'oc25',
            'omol', 'odac', 'omc').
        device: Compute device ('cpu' or 'cuda').

    Returns:
        Configured FAIRChemCalculator instance.
    """
    from fairchem.core import pretrained_mlip, FAIRChemCalculator

    predictor = pretrained_mlip.get_predict_unit(model, device=device)
    return FAIRChemCalculator(predictor, task_name=task_name)


def _run_opt_uma_with_calc(
    atoms,
    calc,
    run_type: str = "geo_opt",
    steps: int = 1000,
    fmax: float = 1e-3,
    fmax_cell: float | None = None,
    write_traj: bool = False,
    traj_prefix: str | None = None,
) -> dict:
    """Core optimization logic with a pre-built calculator.

    Args:
        atoms: ASE Atoms object to optimize (modified in place).
        calc: Pre-built ASE calculator.
        run_type: 'geo_opt', 'cell_opt', or 'geo_opt_cell_opt'.
        steps: Maximum optimization steps per stage.
        fmax: Force convergence criterion (eV/A) for geometry.
        fmax_cell: Force convergence for cell optimization.
            Falls back to fmax if None.
        write_traj: Whether to write trajectory files.
        traj_prefix: Base path for trajectory files (no
            extension). Required if write_traj is True.

    Returns:
        Dict with 'converged' (bool), 'final_energy' (float),
        'n_steps' (int).
    """
    atoms.calc = calc
    if fmax_cell is None:
        fmax_cell = fmax

    total_steps = 0

    if run_type == "geo_opt":
        traj = f"{traj_prefix}.traj" if write_traj else None
        dyn = BFGS(atoms, trajectory=traj)
        converged = dyn.run(fmax=fmax, steps=steps)
        total_steps = dyn.nsteps
    elif run_type == "cell_opt":
        ecf = ExpCellFilter(atoms)
        traj = f"{traj_prefix}.traj" if write_traj else None
        dyn = BFGS(ecf, trajectory=traj)
        converged = dyn.run(fmax=fmax_cell, steps=steps)
        total_steps = dyn.nsteps
    elif run_type == "geo_opt_cell_opt":
        traj1 = f"{traj_prefix}_geo_opt.traj" if write_traj else None
        dyn1 = BFGS(atoms, trajectory=traj1)
        dyn1.run(fmax=fmax, steps=steps)
        total_steps += dyn1.nsteps

        ecf = ExpCellFilter(atoms)
        traj2 = f"{traj_prefix}_cell_opt.traj" if write_traj else None
        dyn2 = BFGS(ecf, trajectory=traj2)
        converged = dyn2.run(fmax=fmax_cell, steps=steps)
        total_steps += dyn2.nsteps
    else:
        raise ValueError(
            f"run_type '{run_type}' is not supported. "
            "Options: 'geo_opt', 'cell_opt', "
            "'geo_opt_cell_opt'."
        )

    return {
        "converged": bool(converged),
        "final_energy": float(atoms.get_potential_energy()),
        "n_steps": total_steps,
    }


def run_opt_uma(
    fname: str,
    run_type: str = "geo_opt",
    steps: int = 1000,
    fmax: float = 1e-3,
    model: str = "uma-s-1p2",
    task_name: str = "omat",
    device: str = "cpu",
    write_traj: bool = False,
    output_fname: str = None,
) -> None:
    """Run geometry and/or cell optimization using a UMA model.

    Args:
        fname: Path to input structure file (CIF, XYZ, POSCAR,
            etc.).
        run_type: Type of optimization - 'geo_opt', 'cell_opt',
            or 'geo_opt_cell_opt'.
        steps: Maximum number of optimization steps.
        fmax: Force convergence criterion in eV/Angstrom.
        model: UMA model name ('uma-s-1p2', 'uma-m-1p1').
        task_name: Task head ('omat', 'oc20', 'oc22', 'oc25',
            'omol', 'odac', 'omc').
        device: Compute device ('cpu' or 'cuda').
        write_traj: Whether to write ASE trajectory files.
        output_fname: Output filename (auto-generated if None).
    """
    atoms = ase_read(fname)
    calc = _create_calculator(model, task_name, device)

    fpath = Path(fname)
    filename = str(fpath.with_suffix(""))
    ext = fpath.suffix

    if output_fname is None:
        output_fname = f"{filename}_opt_{model}{ext}"

    _run_opt_uma_with_calc(
        atoms,
        calc,
        run_type=run_type,
        steps=steps,
        fmax=fmax,
        write_traj=write_traj,
        traj_prefix=filename,
    )

    ase_write(output_fname, atoms)


def run_sp_uma(
    fname: str,
    model: str = "uma-s-1p2",
    task_name: str = "omat",
    device: str = "cpu",
) -> dict:
    """Run single-point energy/forces/stress calculation with UMA.

    Args:
        fname: Path to input structure file.
        model: UMA model name ('uma-s-1p2', 'uma-m-1p1').
        task_name: Task head ('omat', 'oc20', 'oc22', 'oc25',
            'omol', 'odac', 'omc').
        device: Compute device ('cpu' or 'cuda').

    Returns:
        Dict with 'energy' (float), 'forces' (list),
        and 'stress' (list or None).
    """
    atoms = ase_read(fname)
    atoms.calc = _create_calculator(model, task_name, device)

    result = {
        "energy": atoms.get_potential_energy(),
        "forces": atoms.get_forces().tolist(),
    }

    try:
        result["stress"] = atoms.get_stress().tolist()
    except Exception:
        result["stress"] = None

    return result


def run_md_uma(
    fname: str,
    model: str = "uma-s-1p2",
    task_name: str = "omat",
    device: str = "cpu",
    temperature: float = 300.0,
    timestep: float = 1.0,
    steps: int = 1000,
    ensemble: str = "nvt",
    friction: float = 0.01,
    write_traj: bool = False,
    output_fname: str = None,
    log_interval: int = 10,
) -> None:
    """Run molecular dynamics using a UMA model.

    Args:
        fname: Path to input structure file.
        model: UMA model name ('uma-s-1p2', 'uma-m-1p1').
        task_name: Task head ('omat', 'oc20', 'oc22', 'oc25',
            'omol', 'odac', 'omc').
        device: Compute device ('cpu' or 'cuda').
        temperature: Target temperature in Kelvin.
        timestep: MD timestep in femtoseconds.
        steps: Number of MD steps.
        ensemble: 'nve' (VelocityVerlet) or 'nvt' (Langevin).
        friction: Langevin friction coefficient (for NVT only).
        write_traj: Whether to write ASE trajectory file.
        output_fname: Output filename for final structure
            (auto-generated if None).
        log_interval: Steps between log entries.
    """
    atoms = ase_read(fname)
    atoms.calc = _create_calculator(model, task_name, device)

    fpath = Path(fname)
    filename = str(fpath.with_suffix(""))
    ext = fpath.suffix

    if output_fname is None:
        output_fname = f"{filename}_md_{model}{ext}"

    traj = f"{filename}_md.traj" if write_traj else None

    if ensemble == "nvt":
        dyn = Langevin(
            atoms,
            timestep=timestep * units.fs,
            temperature_K=temperature,
            friction=friction,
            trajectory=traj,
            loginterval=log_interval,
        )
    elif ensemble == "nve":
        dyn = VelocityVerlet(
            atoms,
            timestep=timestep * units.fs,
            trajectory=traj,
            loginterval=log_interval,
        )
    else:
        raise ValueError(
            f"ensemble '{ensemble}' is not supported. Options: 'nve', 'nvt'."
        )

    dyn.run(steps=steps)
    ase_write(output_fname, atoms)


def _gpu_worker(
    gpu_id,
    job_queue,
    result_queue,
    task_name,
    steps,
    fmax,
    fmax_cell,
    output_dir,
    write_traj,
    overwrite,
    device,
):
    """Worker process pinned to a single GPU.

    Must be module-level for multiprocessing spawn pickling.
    Caches calculators per (model, task_name) to avoid
    reloading models between jobs on the same GPU.
    """
    import os

    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    calc_cache = {}
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    while True:
        job = job_queue.get()
        if job is None:
            break

        input_file, model, run_type = job
        stem = Path(input_file).stem
        tag = f"{stem}__{run_type}__{model}"
        output_cif = outdir / f"{tag}.cif"

        if output_cif.exists() and not overwrite:
            result_queue.put(
                {
                    "structure": stem,
                    "model": model,
                    "run_type": run_type,
                    "task_name": task_name,
                    "status": "skipped",
                    "output_file": str(output_cif),
                    "converged": None,
                    "final_energy": None,
                    "n_steps": None,
                    "error_message": None,
                }
            )
            continue

        try:
            cache_key = (model, task_name)
            if cache_key not in calc_cache:
                calc_cache[cache_key] = _create_calculator(
                    model, task_name, device
                )
            calc = calc_cache[cache_key]

            atoms = ase_read(input_file)

            traj_dir = outdir / "trajectories"
            traj_prefix = None
            if write_traj:
                traj_dir.mkdir(parents=True, exist_ok=True)
                traj_prefix = str(traj_dir / tag)

            result = _run_opt_uma_with_calc(
                atoms,
                calc,
                run_type=run_type,
                steps=steps,
                fmax=fmax,
                fmax_cell=fmax_cell,
                write_traj=write_traj,
                traj_prefix=traj_prefix,
            )

            ase_write(str(output_cif), atoms)

            result_queue.put(
                {
                    "structure": stem,
                    "model": model,
                    "run_type": run_type,
                    "task_name": task_name,
                    "status": "success",
                    "output_file": str(output_cif),
                    "converged": result["converged"],
                    "final_energy": result["final_energy"],
                    "n_steps": result["n_steps"],
                    "error_message": None,
                }
            )
            print(
                f"[GPU {gpu_id}] Done: {tag} "
                f"(E={result['final_energy']:.4f}, "
                f"converged={result['converged']})"
            )
        except Exception as e:
            result_queue.put(
                {
                    "structure": stem,
                    "model": model,
                    "run_type": run_type,
                    "task_name": task_name,
                    "status": "failure",
                    "output_file": None,
                    "converged": None,
                    "final_energy": None,
                    "n_steps": None,
                    "error_message": str(e),
                }
            )
            print(f"[GPU {gpu_id}] ERROR: {tag}: {e}")


def run_opt_uma_batch(
    input_path: str,
    output_dir: str,
    models: list | str = "uma-s-1p2",
    run_types: list | str = "geo_opt",
    task_name: str = "omat",
    steps: int = 1000,
    fmax: float = 1e-3,
    fmax_cell: float | None = None,
    num_gpus: int | None = None,
    device: str = "cuda",
    write_traj: bool = False,
    overwrite: bool = False,
) -> str:
    """Run UMA optimization in batch across multiple GPUs.

    Creates a Cartesian product of (input_files x models x
    run_types) and distributes jobs across GPU workers using
    a shared queue. Each worker caches calculators to avoid
    reloading models between jobs.

    Args:
        input_path: Path to a single structure file or a
            directory of CIF files.
        output_dir: Directory for output CIFs and results.jsonl.
        models: Model name(s). String or list of strings.
        run_types: Run type(s). String or list of strings.
            Options: 'geo_opt', 'cell_opt', 'geo_opt_cell_opt'.
        task_name: FAIRChem task head.
        steps: Maximum optimization steps per stage.
        fmax: Force convergence criterion (eV/A) for geometry.
        fmax_cell: Force convergence for cell optimization.
            Falls back to fmax if None.
        num_gpus: Number of GPUs. Auto-detected if None.
        device: 'cuda' or 'cpu'.
        write_traj: Write ASE trajectory files.
        overwrite: Overwrite existing output files.

    Returns:
        Path to the results.jsonl file.
    """
    import itertools
    import json
    from multiprocessing import get_context

    # Normalize inputs
    if isinstance(models, str):
        models = [models]
    if isinstance(run_types, str):
        run_types = [run_types]

    input_p = Path(input_path)
    if input_p.is_dir():
        input_files = sorted(str(f) for f in input_p.glob("*.cif"))
        if not input_files:
            raise FileNotFoundError(f"No CIF files found in: {input_path}")
    else:
        input_files = [str(input_p)]

    jobs = list(itertools.product(input_files, models, run_types))
    if not jobs:
        print("No jobs to run.")
        return ""

    # Auto-detect GPU count
    if num_gpus is None:
        if device == "cuda":
            try:
                import torch

                num_gpus = torch.cuda.device_count()
            except ImportError:
                num_gpus = 1
            if num_gpus == 0:
                num_gpus = 1
        else:
            num_gpus = 1

    n_workers = min(num_gpus, len(jobs))

    print(
        f"Running {len(jobs)} jobs across {n_workers} workers (device={device})"
    )

    ctx = get_context("spawn")
    job_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for job in jobs:
        job_queue.put(job)
    for _ in range(n_workers):
        job_queue.put(None)

    workers = []
    for gpu_id in range(n_workers):
        proc = ctx.Process(
            target=_gpu_worker,
            args=(
                gpu_id,
                job_queue,
                result_queue,
                task_name,
                steps,
                fmax,
                fmax_cell,
                output_dir,
                write_traj,
                overwrite,
                device,
            ),
        )
        proc.start()
        workers.append(proc)

    for proc in workers:
        proc.join()

    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    results.sort(key=lambda r: (r["structure"], r["model"], r["run_type"]))

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "results.jsonl"

    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = len(results) - success - skipped

    with open(summary_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    print(
        f"Batch complete: {success} success, {skipped} skipped, "
        f"{failed} failure out of {len(results)} jobs"
    )
    print(f"Results: {summary_path}")

    return str(summary_path)
