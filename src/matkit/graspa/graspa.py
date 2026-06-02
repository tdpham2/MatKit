from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

from ase.io import read as ase_read

from matkit.types import GRASPAResult
from matkit.utils.cif import sanitize_cif_stem
from matkit.utils.template import copy_template, render_template
from matkit.utils.unitcell_calculator import calculate_cell_size


def get_output_data(
    output_path: str,
    unit: str = "mol/kg",
    output_fname: str = "raspa.log",
    eos: bool = False,
) -> GRASPAResult:
    """Parse gRASPA simulation output and extract uptake data.

    Args:
        output_path: Path to directory containing simulation output.
        unit: Unit for uptake values ('mol/kg', 'mg/g', or 'g/L').
        output_fname: Name of the output log file.
        eos: Whether equation of state was used (changes line indices).

    Returns:
        Dict with keys: success, uptake, error, unit, qst, error_qst,
        qst_unit, calc_time_in_s.

    Raises:
        ValueError: If the output file cannot be parsed or unit is invalid.
    """
    result = {
        "success": False,
        "uptake": None,
        "error": None,
        "unit": unit,
        "qst": None,
        "error_qst": None,
        "qst_unit": "kJ/mol",
    }
    uptake_lines = []
    try:
        time_line = None
        with open(Path(output_path) / output_fname, "r") as rf:
            for line in rf:
                if "Overall: Average" in line:
                    uptake_lines.append(line.strip())
                if "Work" in line:
                    time_line = line.strip()

        if time_line is None:
            raise ValueError("Could not find timing line in output.")
        if not uptake_lines:
            raise ValueError("Could not find uptake lines in output.")

        result_qst = uptake_lines[0].split(",")
        qst = float(result_qst[0].split()[-1])
        error_qst = float(result_qst[1].split()[-1])
        result["qst"] = qst
        result["error_qst"] = error_qst

        if not eos:
            result_mol_kg = uptake_lines[6].split(",")
            uptake_mol_kg = float(result_mol_kg[0].split()[-1])
            error_mol_kg = float(result_mol_kg[1].split()[-1])

            result_mg_g = uptake_lines[4].split(",")
            uptake_mg_g = float(result_mg_g[0].split()[-1])
            error_mg_g = float(result_mg_g[1].split()[-1])

            result_g_L = uptake_lines[7].split(",")
            uptake_g_L = float(result_g_L[0].split()[-1])
            error_g_L = float(result_g_L[1].split()[-1])

        else:
            result_mol_kg = uptake_lines[11].split(",")
            uptake_mol_kg = float(result_mol_kg[0].split()[-1])
            error_mol_kg = float(result_mol_kg[1].split()[-1])

            result_mg_g = uptake_lines[5].split(",")
            uptake_mg_g = float(result_mg_g[0].split()[-1])
            error_mg_g = float(result_mg_g[1].split()[-1])

            result_g_L = uptake_lines[13].split(",")
            uptake_g_L = float(result_g_L[0].split()[-1])
            error_g_L = float(result_g_L[1].split()[-1])

        if unit == "mol/kg":
            result["uptake"] = uptake_mol_kg
            result["error"] = error_mol_kg
        elif unit == "mg/g":
            result["uptake"] = uptake_mg_g
            result["error"] = error_mg_g
        elif unit == "g/L":
            result["uptake"] = uptake_g_L
            result["error"] = error_g_L
        else:
            raise ValueError(f"Unit {unit} is not supported yet.")

        time = float(time_line.split()[2])
        result["calc_time_in_s"] = time
        result["success"] = True
        return result
    except Exception as e:
        raise ValueError(f"Failed to parse output: {e}") from e


DEFAULT_ADSORBATE_PARAMS = {
    "TranslationProbability": 1.0,
    "RotationProbability": 1.0,
    "ReinsertionProbability": 1.0,
    "SwapProbability": 2.0,
    "CreateNumberOfMolecules": 0,
    "IdealGasRosenbluthWeight": 1.0,
    "FugacityCoefficient": "PR-EOS",
    "MolFraction": 1.0,
}


def generate_component_blocks(adsorbates: list[dict]) -> str:
    """Generate gRASPA component block text for simulation input.

    Args:
        adsorbates: List of dicts, each with a 'MoleculeName' key and
            optional parameter overrides (e.g., SwapProbability).

    Returns:
        Formatted component block string for simulation.input.

    Raises:
        ValueError: If any adsorbate dict is missing 'MoleculeName'.
    """
    lines = []
    for i, ad in enumerate(adsorbates):
        if "MoleculeName" not in ad:
            raise ValueError("Each adsorbate must have a 'MoleculeName' key")

        # Start the block
        lines.append(
            f"Component {i} MoleculeName              {ad['MoleculeName']}"
        )

        for key, default_val in DEFAULT_ADSORBATE_PARAMS.items():
            # Skip this key if the user explicitly set it to None
            if key in ad and ad[key] is None:
                continue
            # If user provided a value, use it; else use default
            val = ad.get(key, default_val)
            lines.append(f"             {key:28} {val}")

        lines.append("")  # Spacer between components

    return "\n".join(lines)


def setup_simulation(
    cif: str,
    outpath: str,
    adsorbates: list[dict],
    temperature: float = 298.0,
    pressure: float = 1e5,
    cutoff: float = 12.8,
    n_cycle: int = 1000,
    template_dir: str = "template",
    cell_size: list[int] | None = None,
) -> bool:
    """Set up a gRASPA GCMC simulation.

    Copies template files to the output directory, substitutes
    simulation parameters into the input file, and adds component
    blocks for the specified adsorbates.

    Args:
        cif: Path to the input CIF structure file.
        outpath: Directory to write simulation files to.
        adsorbates: List of adsorbate dicts with 'MoleculeName' key.
        temperature: Simulation temperature in Kelvin.
        pressure: Simulation pressure in Pascals.
        cutoff: Van der Waals cutoff radius in Angstrom.
        n_cycle: Number of Monte Carlo cycles.
        template_dir: Template subdirectory name under files/.
        cell_size: Pre-computed unit cell dimensions [uc_x, uc_y, uc_z].
            If provided, skips reading the CIF to calculate cell size.

    Returns:
        True on success.

    Raises:
        FileNotFoundError: If the CIF file does not exist.
    """
    outdir = Path(outpath)

    cifpath = Path(cif)
    if not cifpath.exists():
        raise FileNotFoundError(f"CIF file does not exist: {cif}")
    safe_stem = sanitize_cif_stem(cifpath.stem)

    template_path = Path(__file__).parent / "files" / template_dir
    copy_template(template_path, outdir)
    shutil.copy(cifpath, outdir / f"{safe_stem}.cif")

    # Use pre-computed cell size or read CIF
    if cell_size is not None:
        uc_x, uc_y, uc_z = cell_size
    else:
        atoms = ase_read(cifpath)
        uc_x, uc_y, uc_z = calculate_cell_size(atoms)

    input_path = outdir / "simulation.input"
    render_template(
        input_path,
        {
            "NCYCLE": str(n_cycle),
            "TEMPERATURE": str(temperature),
            "PRESSURE": str(pressure),
            "CUTOFF": str(cutoff),
            "CIFFILE": safe_stem,
            "UC_X": str(uc_x),
            "UC_Y": str(uc_y),
            "UC_Z": str(uc_z),
        },
    )

    # Replace component block placeholder
    component_block = generate_component_blocks(adsorbates)
    content = input_path.read_text()
    content = content.replace("__COMPONENTS__", component_block)
    input_path.write_text(content)

    return True


def _setup_single_cif(
    cif: Path,
    out_path: Path,
    adsorbates: list[dict],
    temperatures: list[float],
    pressures: list[float],
    cutoff: float,
    n_cycle: int,
    template_dir: str,
) -> list[dict]:
    """Set up all T x P simulations for a single CIF file.

    Reads the CIF once to compute cell size, then creates a simulation
    directory for each (temperature, pressure) combination.
    """
    atoms = ase_read(cif)
    cell_size = calculate_cell_size(atoms)
    safe_stem = sanitize_cif_stem(cif.stem)

    entries = []
    for temp, pres in product(temperatures, pressures):
        sim_dir = out_path / safe_stem / f"T{temp}_P{pres:g}"
        setup_simulation(
            cif=str(cif),
            outpath=str(sim_dir),
            adsorbates=adsorbates,
            temperature=temp,
            pressure=pres,
            cutoff=cutoff,
            n_cycle=n_cycle,
            template_dir=template_dir,
            cell_size=cell_size,
        )
        entry = {
            "sim_dir": str(sim_dir),
            "cif": cif.name,
            "temperature": temp,
            "pressure": pres,
            "adsorbates": [ad["MoleculeName"] for ad in adsorbates],
        }
        if safe_stem != cif.stem:
            entry["original_cif_stem"] = cif.stem
            entry["safe_cif_stem"] = safe_stem
        entries.append(entry)
    return entries


def setup_batch(
    cif_dir: str,
    outpath: str,
    adsorbates: list[dict],
    temperatures: list[float],
    pressures: list[float],
    cutoff: float = 12.8,
    n_cycle: int = 1000,
    template_dir: str = "template",
    max_workers: int | None = None,
) -> list[dict]:
    """Set up gRASPA simulations for all CIF x T x P.

    Discovers all .cif files in cif_dir and creates a simulation directory
    for each (CIF, temperature, pressure) combination. Each CIF is read
    once to compute cell size, then all T x P combinations reuse the
    cached result. CIFs are processed in parallel using threads.

    Writes a simulations.jsonl manifest to outpath.

    Args:
        cif_dir: Directory containing input CIF files.
        outpath: Base output directory for all simulation directories.
        adsorbates: List of adsorbate dicts with 'MoleculeName' key.
        temperatures: List of simulation temperatures in Kelvin.
        pressures: List of simulation pressures in Pascals.
        cutoff: Van der Waals cutoff radius in Angstrom.
        n_cycle: Number of Monte Carlo cycles.
        template_dir: Template subdirectory name under files/.
        max_workers: Max threads for parallel CIF processing.
            Defaults to None (lets ThreadPoolExecutor choose).

    Returns:
        List of manifest dicts, each with keys: sim_dir, cif,
        temperature, pressure, adsorbates.

    Raises:
        ValueError: If cif_dir does not exist or contains no .cif files.
    """
    cif_path = Path(cif_dir)
    out_path = Path(outpath)

    if not cif_path.is_dir():
        raise ValueError(f"CIF directory does not exist: {cif_dir}")

    cif_files = sorted(cif_path.glob("*.cif"))
    if not cif_files:
        raise ValueError(f"No .cif files found in {cif_dir}")

    manifest = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                _setup_single_cif,
                cif,
                out_path,
                adsorbates,
                temperatures,
                pressures,
                cutoff,
                n_cycle,
                template_dir,
            )
            for cif in cif_files
        ]
        for future in futures:
            manifest.extend(future.result())

    manifest_path = out_path / "simulations.jsonl"
    with manifest_path.open("w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")

    rename_map = {
        entry["original_cif_stem"]: entry["safe_cif_stem"]
        for entry in manifest
        if "original_cif_stem" in entry
    }
    if rename_map:
        _write_cif_mapping(out_path, rename_map)

    return manifest


def _write_cif_mapping(out_path: Path, mapping: dict[str, str]) -> None:
    """Write {original_stem: safe_stem} to out_path/cif_mapping.json.

    Merges with any existing file so re-running setup_batch with
    additional CIFs is additive.
    """
    mapping_path = out_path / "cif_mapping.json"
    existing: dict[str, str] = {}
    if mapping_path.exists():
        existing = json.loads(mapping_path.read_text())
    existing.update(mapping)
    mapping_path.write_text(json.dumps(existing, indent=2, sort_keys=True))
