from pathlib import Path
import shutil
from matkit.utils.unitcell_calculator import calculate_cell_size
from ase.io import read as ase_read


def get_output_data(
    output_path: str,
    unit: str = "mol/kg",
    output_fname: str = "raspa.log",
    eos: bool = False,
) -> dict:
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
        with open(Path(output_path) / output_fname, "r") as rf:
            for line in rf:
                if "Overall: Average" in line:
                    uptake_lines.append(line.strip())
                if "Work" in line:
                    time_line = line.strip()

        result_qst = uptake_lines[0].split(",")
        qst = result_qst[0].split()[-1]
        error_qst = result_qst[1].split()[-1]
        result["qst"] = qst
        result["error_qst"] = error_qst

        if not eos:
            result_mol_kg = uptake_lines[6].split(",")
            uptake_mol_kg = result_mol_kg[0].split()[-1]
            error_mol_kg = result_mol_kg[1].split()[-1]

            result_mg_g = uptake_lines[4].split(",")
            uptake_mg_g = result_mg_g[0].split()[-1]
            error_mg_g = result_mg_g[1].split()[-1]

            result_g_L = uptake_lines[7].split(",")
            uptake_g_L = result_g_L[0].split()[-1]
            error_g_L = result_g_L[1].split()[-1]

        else:
            result_mol_kg = uptake_lines[11].split(",")
            uptake_mol_kg = result_mol_kg[0].split()[-1]
            error_mol_kg = result_mol_kg[1].split()[-1]

            result_mg_g = uptake_lines[5].split(",")
            uptake_mg_g = result_mg_g[0].split()[-1]
            error_mg_g = result_mg_g[1].split()[-1]

            result_g_L = uptake_lines[13].split(",")
            uptake_g_L = result_g_L[0].split()[-1]
            error_g_L = result_g_L[1].split()[-1]

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
        raise ValueError(f"{e}")


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

    Returns:
        True on success.

    Raises:
        FileNotFoundError: If the CIF file does not exist.
    """
    outdir = Path(outpath)
    outdir.mkdir(parents=True, exist_ok=True)

    cifpath = Path(cif)
    if not cifpath.exists():
        raise FileNotFoundError(f"CIF file does not exist: {cif}")
    cifname = cifpath.stem
    shutil.copy(cifpath, outdir / f"{cifname}.cif")

    # Copy template files
    template_path = Path(__file__).parent / "files" / template_dir
    for item in template_path.iterdir():
        if item.is_dir():
            shutil.copytree(item, outdir, dirs_exist_ok=True)
        else:
            shutil.copy2(item, outdir)

    # Read CIF to get cell size
    atoms = ase_read(cifpath)
    uc_x, uc_y, uc_z = calculate_cell_size(atoms)

    # Read template and replace placeholders
    input_path = outdir / "simulation.input"
    with input_path.open("r") as f:
        template = f.read()

    subs = {
        "NCYCLE": str(n_cycle),
        "TEMPERATURE": str(temperature),
        "PRESSURE": str(pressure),
        "CUTOFF": str(cutoff),
        "CIFFILE": cifname,
        "UC_X": str(uc_x),
        "UC_Y": str(uc_y),
        "UC_Z": str(uc_z),
    }

    for key, val in subs.items():
        template = template.replace(key, val)

    # Replace component block placeholder
    component_block = generate_component_blocks(adsorbates)
    template = template.replace("__COMPONENTS__", component_block)
    # Write final input
    with input_path.open("w") as f:
        f.write(template)

    return True
