from pathlib import Path
import shutil
import os
from matkit.utils.unitcell_calculator import calculate_cell_size
from ase.io import read as ase_read

_file_dir = Path(__file__).parent / "files" / "template"


def setup_input_simulation(
    cifs: list[str],
    outpath: str,
    adsorbate: str = "CO2",
    temperature: float = 298,
    pressure: float = 1e5,
    cutoff: float = 12.8,
    n_cycle: int = 1000,
):
    outpath = Path(outpath)

    for cif in cifs:
        cifpath = Path(cif)
        if not cifpath.exists():
            raise FileNotFoundError(f"Source directory does not exist: {cif}")

        cifname = cif.split("/")[-1][:-4]
        outdir = Path(os.path.join(outpath, cifname))
        outdir.mkdir(parents=True, exist_ok=True)
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
                    line = line.replace(
                        "UC_X UC_Y UC_Z", f"{uc_x} {uc_y} {uc_z}"
                    )
                if "CUTOFF" in line:
                    line = line.replace("CUTOFF", str(cutoff))
                if "CIFFILE" in line:
                    line = line.replace("CIFFILE", cifname)
                f_out.write(line)

        shutil.move(
            f"{outdir}/simulation.input.tmp", f"{outdir}/simulation.input"
        )

    return True


def get_output_data(output_path, calc_time=False, unit="mol/kg"):
    result = {"success": False, "uptake": 0, "error": 0, "unit": unit}
    with open(output_path, "r") as file:
        for line in file:
            if "Average loading absolute [mol/kg framework]" in line:
                mol_kg_line = line.strip()
            elif "Average loading absolute [milligram/gram framework]" in line:
                mg_g_line = line.strip()
            elif "Framework Density" in line:
                density_line = line.strip()

    if not all([mol_kg_line, mg_g_line, density_line]):
        raise ValueError("One or more expected lines were not found.")

    uptake_mol_kg = float(mol_kg_line.split()[5])
    error_mol_kg = float(mol_kg_line.split()[7])

    density_kg_m3 = float(density_line.split()[2])
    uptake_mg_g = float(mg_g_line.split()[5])
    error_mg_g = float(mg_g_line.split()[7])

    if unit == "mol/kg":
        result["uptake"] = uptake_mol_kg
        result["error"] = error_mol_kg
    elif unit == "g/L":
        # Unit conversion to g/L
        uptake_g_L = uptake_mg_g * density_kg_m3 / 1000
        error_g_L = error_mg_g * density_kg_m3 / 1000
        result["uptake"] = uptake_g_L
        result["error"] = error_g_L
    else:
        raise ValueError("Unit {unit} is not supported yet.")

    if calc_time:
        from datetime import datetime

        with open(output_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        start_raw = lines[6]
        end_raw = lines[-3]

        # Parse the raw datetime strings
        start_time = datetime.strptime(start_raw, "%a %b %d %H:%M:%S %Y")
        end_time = datetime.strptime(end_raw, "%a %b %d %H:%M:%S %Y")

        duration_seconds = int((end_time - start_time).total_seconds())
        result["calc_time_in_s"] = duration_seconds
    return result
