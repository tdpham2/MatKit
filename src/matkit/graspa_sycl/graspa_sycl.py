from pathlib import Path
import shutil
from matkit.utils.unitcell_calculator import calculate_cell_size
from ase.io import read as ase_read

_file_dir = Path(__file__).parent / "files" / "template"


def setup_simulation(
    cif: str,
    outpath: str,
    adsorbate: str = "CO2",
    temperature: float = 298,
    pressure: float = 1e5,
    cutoff: float = 12.8,
    n_cycle: int = 1000,
) -> bool:
    """Set up a gRASPA SYCL (Intel GPU) GCMC simulation.

    Args:
        cif: Path to the input CIF structure file.
        outpath: Directory to write simulation files to.
        adsorbate: Adsorbate molecule name (e.g., 'CO2', 'H2').
        temperature: Simulation temperature in Kelvin.
        pressure: Simulation pressure in Pascals.
        cutoff: Van der Waals cutoff radius in Angstrom.
        n_cycle: Number of Monte Carlo cycles.

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
    for item in _file_dir.iterdir():
        if item.is_dir():
            shutil.copytree(item, outdir, dirs_exist_ok=True)
        else:
            shutil.copy2(item, outdir)
    shutil.copy(cif, outdir)
    # Editing input file.
    atoms = ase_read(cif)
    [uc_x, uc_y, uc_z] = calculate_cell_size(atoms)

    with (
        open(f"{outdir}/simulation.input", "r") as f_in,
        open(f"{outdir}/simulation.input.tmp", "w") as f_out,
    ):
        for line in f_in:
            if "NCYCLE" in line:
                line = line.replace("NCYCLE", str(n_cycle))
            if "ADSORBATE" in line:
                line = line.replace("ADSORBATE", adsorbate)
            if "TEMPERATURE" in line:
                line = line.replace("TEMPERATURE", str(temperature))
            if "PRESSURE" in line:
                line = line.replace("PRESSURE", str(pressure))
            if "UC_X UC_Y UC_Z" in line:
                line = line.replace("UC_X UC_Y UC_Z", f"{uc_x} {uc_y} {uc_z}")
            if "CUTOFF" in line:
                line = line.replace("CUTOFF", str(cutoff))
            if "CIFFILE" in line:
                line = line.replace("CIFFILE", cifname)
            f_out.write(line)

    shutil.move(f"{outdir}/simulation.input.tmp", f"{outdir}/simulation.input")

    return True


def get_output_data(
    output_path: str,
    calc_time: bool = False,
    unit: str = "mol/kg",
    output_fname: str = "raspa.log",
    cifname: str = None,
    adsorbate: str = "CO2",
) -> dict:
    """Parse gRASPA SYCL simulation output and extract uptake data.

    Args:
        output_path: Path to directory containing simulation output.
        calc_time: Whether to extract calculation time.
        unit: Unit for uptake values ('mol/kg' or 'g/L').
        output_fname: Name of the output log file.
        cifname: CIF filename in output dir (auto-detected if None).
        adsorbate: Adsorbate name (needed for g/L molar mass lookup).

    Returns:
        Dict with keys: success, uptake, error, unit, calc_time_in_s.

    Raises:
        ValueError: If parsing fails or adsorbate is unsupported for g/L.
    """
    result = {"success": False, "uptake": 0, "error": 0, "unit": unit}
    output_dir = Path(output_path)
    try:
        with open(output_dir / output_fname, "r") as rf:
            for line in rf:
                if "UnitCells" in line:
                    unitcell_line = line.strip()
                elif "Overall: Average:" in line:
                    uptake_line = line.strip()
                elif "Work" in line:
                    time_line = line.strip()
        result["calc_time_in_s"] = float(time_line.split()[2])

        if cifname is None:
            cif_list = list(output_dir.glob("*.cif"))
            if len(cif_list) != 1:
                raise ValueError(
                    f"Found {len(cif_list)} CIF files in "
                    f"{output_path}, expected 1."
                )
            cifpath = str(cif_list[0])
        else:
            cifpath = str(output_dir / cifname)

        uptake_total_molecule = float(uptake_line.split()[2][:-1])
        error_total_molecule = float(uptake_line.split()[4][:-1])
        unitcell = unitcell_line.split()[4:]
        unitcell = [int(float(i)) for i in unitcell]
        atoms = ase_read(cifpath)

        if unit == "mol/kg":
            framework_mass = (
                sum(atoms.get_masses())
                * unitcell[0]
                * unitcell[1]
                * unitcell[2]
            )

            uptake_mol_kg = uptake_total_molecule / framework_mass * 1000
            error_mol_kg = error_total_molecule / framework_mass * 1000
            result["uptake"] = uptake_mol_kg
            result["error"] = error_mol_kg
            result["success"] = True
        elif unit == "g/L":
            framework_vol = (
                atoms.get_volume() * unitcell[0] * unitcell[1] * unitcell[2]
            )
            framework_vol_in_L = framework_vol * 1e-27
            if adsorbate == "CO2":
                molar_mass = 44.0098
            elif adsorbate == "H2":
                molar_mass = 2.02
            else:
                raise ValueError(f"Adsorbate {adsorbate} is not supported.")
            uptake_g_L = (
                uptake_total_molecule
                / (6.022 * 1e23)
                * molar_mass
                / framework_vol_in_L
            )
            error_g_L = (
                error_total_molecule
                / (6.022 * 1e23)
                * molar_mass
                / framework_vol_in_L
            )
            result["uptake"] = uptake_g_L
            result["error"] = error_g_L

        return result
    except Exception as e:
        raise ValueError(e)
