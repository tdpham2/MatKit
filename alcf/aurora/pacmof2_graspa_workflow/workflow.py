"""
PACMOF2 + gRASPA-SYCL Parsl Workflow for ALCF Aurora
=====================================================

Pipeline: Folder of CIF files -> PACMOF2 (assign charges) -> gRASPA-SYCL (GCMC)

This workflow:
  1. Reads all .cif files from an input directory
  2. Runs PACMOF2 to assign partial atomic charges (neutral MOFs)
  3. Runs gRASPA-SYCL GCMC simulations on the charged CIFs
  4. Collects results into a summary JSONL file

Usage:
    python workflow.py --input_dir /path/to/cifs --output_dir /path/to/output

Requirements:
    - parsl
    - pacmof2  (pip install from https://github.com/snurr-group/pacmof2)
    - matkit   (this project)
    - ase, pymatgen
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import parsl
from parsl import python_app
from parsl.config import Config
from parsl.providers import LocalProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from parsl.addresses import address_by_interface

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Default gRASPA-SYCL binary path on Aurora
# ------------------------------------------------------------------ #
GRASPA_SYCL_BIN = (
    "/lus/flare/projects/IQC/thang/SC_Hari/gRASPA/graspa-sycl/bin/sycl.out"
)

# Environment variables prepended to every gRASPA invocation
GRASPA_ENV_PREFIX = (
    "export OMP_NUM_THREADS=1; export ZE_FLAT_DEVICE_HIERARCHY=FLAT; "
)


def _build_graspa_cmd(graspa_bin: str) -> str:
    """Build the full shell command for a gRASPA-SYCL invocation.

    Parameters
    ----------
    graspa_bin : str
        Path to the gRASPA-SYCL binary (sycl.out).

    Returns
    -------
    str
        Shell command string with environment setup prepended.
    """
    return f"{GRASPA_ENV_PREFIX}{graspa_bin}"


# ------------------------------------------------------------------ #
#  Aurora Parsl configuration
# ------------------------------------------------------------------ #
def get_aurora_config(
    run_dir: str = None,
    account: str = "IQC",
    queue: str = "default",
    walltime: str = "01:00:00",
    nodes_per_block: int = 1,
    max_blocks: int = 1,
    worker_init: str = None,
) -> Config:
    """Build a Parsl Config for ALCF Aurora.

    Parameters
    ----------
    run_dir : str
        Directory for Parsl run files (logs, etc.).
    account : str
        PBS project/account name.
    queue : str
        PBS queue name.
    walltime : str
        Maximum walltime per PBS job.
    nodes_per_block : int
        Number of nodes per Parsl block.
    max_blocks : int
        Maximum number of blocks (PBS jobs) Parsl can launch.
    worker_init : str
        Shell commands to run before each worker starts.

    Returns
    -------
    parsl.Config
    """
    if run_dir is None:
        run_dir = os.getcwd()

    if worker_init is None:
        worker_init = "module load frameworks\nexport TMPDIR=/tmp\n"

    # Determine number of nodes from PBS environment if available
    nodefile = os.environ.get("PBS_NODEFILE")
    if nodefile and os.path.exists(nodefile):
        with open(nodefile) as f:
            num_nodes = len(set(f.read().strip().splitlines()))
    else:
        num_nodes = nodes_per_block
        logger.warning(
            "PBS_NODEFILE not found; defaulting to %d node(s).",
            num_nodes,
        )

    return Config(
        executors=[
            HighThroughputExecutor(
                label="aurora_gpu",
                heartbeat_period=30,
                heartbeat_threshold=240,
                available_accelerators=12,
                max_workers_per_node=9,
                address=address_by_interface("bond0"),
                provider=LocalProvider(
                    nodes_per_block=num_nodes,
                    launcher=MpiExecLauncher(
                        bind_cmd="--cpu-bind", overrides="--ppn 1"
                    ),
                    init_blocks=1,
                    worker_init=worker_init,
                    max_blocks=1,
                    min_blocks=0,
                ),
            )
        ],
        run_dir=run_dir,
    )


# ------------------------------------------------------------------ #
#  Parsl Apps
# ------------------------------------------------------------------ #
@python_app
def run_pacmof2(cif_path: str, output_dir: str) -> str:
    """Assign partial atomic charges to a CIF using PACMOF2.

    Parameters
    ----------
    cif_path : str
        Path to the input CIF file.
    output_dir : str
        Directory to write the charged CIF file.

    Returns
    -------
    str
        Path to the output CIF file with charges.
    """
    from pathlib import Path

    from pacmof2 import get_charges

    cif = Path(cif_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    get_charges(
        path_to_cif=str(cif),
        output_path=str(out),
        identifier="_pacmof",
        multiple_cifs=False,
        net_charge=0,
    )

    charged_cif = out / f"{cif.stem}_pacmof.cif"
    if not charged_cif.exists():
        raise FileNotFoundError(
            f"PACMOF2 did not produce expected output: {charged_cif}"
        )
    return str(charged_cif)


@python_app
def run_graspa_sycl(
    charged_cif_path: str,
    output_dir: str,
    adsorbate: str = "H2",
    temperature: float = 298.0,
    pressure: float = 1e5,
    n_cycle: int = 10000,
    graspa_bin: str = GRASPA_SYCL_BIN,
) -> dict:
    """Set up and run a gRASPA-SYCL GCMC simulation.

    Parameters
    ----------
    charged_cif_path : str
        Path to the CIF file with partial atomic charges.
    output_dir : str
        Directory to write simulation files and results.
    adsorbate : str
        Adsorbate molecule name (e.g., 'H2', 'CO2').
    temperature : float
        Simulation temperature in Kelvin.
    pressure : float
        Simulation pressure in Pascals.
    n_cycle : int
        Number of Monte Carlo cycles.
    graspa_bin : str
        Path to the gRASPA-SYCL binary (sycl.out).

    Returns
    -------
    dict
        Simulation results including uptake, status, and timing.
    """
    import subprocess
    from pathlib import Path

    from matkit.graspa_sycl import setup_simulation, get_output_data

    cif = Path(charged_cif_path)
    sim_dir = (
        Path(output_dir) / f"{cif.stem}--{adsorbate}-{temperature}-{pressure:g}"
    )
    sim_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "structure": cif.stem,
        "adsorbate": adsorbate,
        "temperature_K": temperature,
        "pressure_Pa": pressure,
        "status": "failure",
        "uptake": None,
        "error": None,
        "unit": "mol/kg",
        "calc_time_s": None,
        "sim_dir": str(sim_dir),
    }

    # Build the full shell command with env vars from the binary path
    graspa_cmd = _build_graspa_cmd(graspa_bin)

    try:
        # Set up simulation input files
        setup_simulation(
            cif=str(cif),
            outpath=str(sim_dir),
            adsorbate=adsorbate,
            temperature=temperature,
            pressure=pressure,
            n_cycle=n_cycle,
        )

        # Run the gRASPA-SYCL binary
        log_file = sim_dir / "raspa.log"
        err_file = sim_dir / "raspa.err"
        with open(log_file, "w") as fp, open(err_file, "w") as fe:
            subprocess.run(
                graspa_cmd,
                cwd=str(sim_dir),
                stdout=fp,
                stderr=fe,
                shell=True,
            )

        # Parse output
        output = get_output_data(
            output_path=str(sim_dir),
            unit="mol/kg",
            output_fname="raspa.log",
            adsorbate=adsorbate,
        )

        result["uptake"] = output.get("uptake", 0)
        result["error"] = output.get("error", 0)
        result["calc_time_s"] = output.get("calc_time_in_s")
        result["status"] = "success" if output.get("success") else "failure"

    except Exception as e:
        result["status"] = "failure"
        result["error_message"] = str(e)

    return result


# ------------------------------------------------------------------ #
#  Main Workflow
# ------------------------------------------------------------------ #
def run_workflow(
    input_dir: str,
    output_dir: str,
    adsorbate: str = "H2",
    temperature: float = 298.0,
    pressure: float = 1e5,
    n_cycle: int = 10000,
    graspa_bin: str = GRASPA_SYCL_BIN,
) -> list[dict]:
    """Run the PACMOF2 -> gRASPA-SYCL pipeline on a folder of CIF files.

    Parameters
    ----------
    input_dir : str
        Directory containing input CIF files.
    output_dir : str
        Base output directory for charged CIFs and simulation results.
    adsorbate : str
        Adsorbate molecule name.
    temperature : float
        Simulation temperature in Kelvin.
    pressure : float
        Simulation pressure in Pascals.
    n_cycle : int
        Number of Monte Carlo cycles.
    graspa_bin : str
        Path to the gRASPA-SYCL binary (sycl.out).

    Returns
    -------
    list[dict]
        List of result dicts, one per CIF file.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    cif_files = sorted(input_path.glob("*.cif"))
    if not cif_files:
        raise ValueError(f"No .cif files found in {input_dir}")

    logger.info("Found %d CIF files in %s", len(cif_files), input_dir)

    # Directory layout:
    #   output_dir/
    #     charged_cifs/        <- PACMOF2 output
    #     simulations/         <- gRASPA simulation directories
    #     results.jsonl        <- summary file
    charged_dir = output_path / "charged_cifs"
    sim_base_dir = output_path / "simulations"
    charged_dir.mkdir(parents=True, exist_ok=True)
    sim_base_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- #
    #  Stage 1: PACMOF2 — assign charges (all submitted in parallel)
    # -------------------------------------------------------------- #
    logger.info("Stage 1: Submitting PACMOF2 tasks...")
    pacmof_futures = {}
    for cif in cif_files:
        fut = run_pacmof2(
            cif_path=str(cif),
            output_dir=str(charged_dir),
        )
        pacmof_futures[cif.stem] = fut

    # -------------------------------------------------------------- #
    #  Stage 2: gRASPA-SYCL — GCMC simulations (chained to PACMOF2)
    # -------------------------------------------------------------- #
    logger.info("Stage 2: Submitting gRASPA-SYCL tasks (chained to PACMOF2)...")
    graspa_futures = {}
    for mof_name, pacmof_fut in pacmof_futures.items():
        # Parsl automatically resolves the future: the gRASPA task
        # will not start until the PACMOF2 task completes and provides
        # the charged CIF path.
        fut = run_graspa_sycl(
            charged_cif_path=pacmof_fut,
            output_dir=str(sim_base_dir),
            adsorbate=adsorbate,
            temperature=temperature,
            pressure=pressure,
            n_cycle=n_cycle,
            graspa_bin=graspa_bin,
        )
        graspa_futures[mof_name] = fut

    # -------------------------------------------------------------- #
    #  Collect results
    # -------------------------------------------------------------- #
    logger.info("Waiting for all tasks to complete...")
    results = []
    for mof_name, fut in graspa_futures.items():
        try:
            result = fut.result()
        except Exception as e:
            result = {
                "structure": mof_name,
                "status": "failure",
                "error_message": str(e),
            }
        results.append(result)

    # Write summary
    summary_path = output_path / "results.jsonl"
    success_count = 0
    with open(summary_path, "w", encoding="utf-8") as f:
        for res in results:
            if res.get("status") == "success":
                success_count += 1
            f.write(json.dumps(res) + "\n")

    logger.info(
        "Workflow complete: %d/%d succeeded. Results: %s",
        success_count,
        len(results),
        summary_path,
    )

    return results


# ------------------------------------------------------------------ #
#  CLI Entry Point
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="PACMOF2 + gRASPA-SYCL Parsl workflow for Aurora",
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing input CIF files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Base output directory for results.",
    )
    parser.add_argument(
        "--adsorbate",
        default="H2",
        help="Adsorbate molecule (default: H2).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=298.0,
        help="Simulation temperature in K (default: 298.0).",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=1e5,
        help="Simulation pressure in Pa (default: 1e5).",
    )
    parser.add_argument(
        "--n_cycle",
        type=int,
        default=10000,
        help="Number of MC cycles (default: 10000).",
    )
    parser.add_argument(
        "--graspa_bin",
        default=GRASPA_SYCL_BIN,
        help="Path to the gRASPA-SYCL binary (sycl.out).",
    )
    parser.add_argument(
        "--account",
        default="IQC",
        help="PBS account/project name (default: IQC).",
    )
    parser.add_argument(
        "--queue",
        default="default",
        help="PBS queue name (default: default).",
    )
    parser.add_argument(
        "--walltime",
        default="01:00:00",
        help="PBS walltime (default: 01:00:00).",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes per block (default: 1).",
    )

    args = parser.parse_args()

    # Initialize Parsl
    config = get_aurora_config(
        run_dir=args.output_dir,
        account=args.account,
        queue=args.queue,
        walltime=args.walltime,
        nodes_per_block=args.nodes,
    )
    parsl.load(config)

    t0 = time.time()
    run_workflow(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        adsorbate=args.adsorbate,
        temperature=args.temperature,
        pressure=args.pressure,
        n_cycle=args.n_cycle,
        graspa_bin=args.graspa_bin,
    )
    elapsed = time.time() - t0

    logger.info("Total wall time: %.1f seconds", elapsed)
    parsl.dfk().cleanup()


if __name__ == "__main__":
    main()
