from pathlib import Path
import shutil
import os
from matkit.utils.unitcell_calculator import calculate_cell_size
from ase.io import read as ase_read


def setup_input_simulation(
    cif: str,
    outpath: str,
    adsorbate: str = "CO2",
    temperature: float = 298,
    pressure: float = 1e5,
    cutoff: float = 12.8,
    n_cycle: int = 1000,
    template_dir: str = "template",
):
    outpath = Path(outpath)
    _file_dir = Path(__file__).parent / "files" / template_dir

    cifpath = Path(cif)
    if not cifpath.exists():
        raise FileNotFoundError(f"Source directory does not exist: {cif}")

    cifname = cif.split("/")[-1][:-4]
    outdir = Path(outpath)
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
            if template_dir != "template_mixture":
                if "ADSORBATE" in line:
                    line = line.replace("ADSORBATE", adsorbate)

            f_out.write(line)

    shutil.move(f"{outdir}/simulation.input.tmp", f"{outdir}/simulation.input")

    return True


def get_output_data(
    output_path: str, unit="mol/kg", output_fname: str = "raspa.log", eos: bool = False
):
    result = {"success": False, "uptake": 0, "error": 0, "unit": unit}
    uptake_lines = []
    try:
        with open(os.path.join(output_path, output_fname), "r") as rf:
            for line in rf:
                if "Overall: Average" in line:
                    uptake_lines.append(line.strip())
                if "Work" in line:
                    time_line = line.strip()

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

        return result
