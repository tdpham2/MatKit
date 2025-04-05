from pathlib import Path
import shutil
import os
from matkit.utils.unitcell_calculator import calculate_cell_size
from ase.io import read as ase_read
import glob

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


def get_output_data(
    output_path: str,
    calc_time: bool = False,
    unit: str = "mol/kg",
    output_fname: str = "raspa.log",
    cifname: str = None,
):
    result = {"success": False, "uptake": 0, "error": 0, "unit": unit}
    try:
        with open(os.path.join(output_path, output_fname), "r") as file:
            for line in file:
                if "UnitCells" in line:
                    unitcell_line = line.strip()
                elif "Overall: Average:" in line:
                    uptake_line = line.strip()
                elif "Work" in line:
                    time_line = line.strip()

        if cifname is None:
            cif_list = glob.glob(os.path.join(output_path, "*.cif"))
            if len(cif_list) != 1:
                raise ValueError(f"There are {len(cif_list)} in {output_path}.")
            else:
                cifpath = os.path.join(output_path, cif_list[0])
        else:
            cifpath = os.path.join(output_path, cifname)

        atoms = ase_read(cifpath)
        masses = sum(atoms.get_masses())

        uptake_total_molecule = float(uptake_line.split()[2][:-1])
        error_total_molecule = float(uptake_line.split()[4][:-1])
        unitcell = unitcell_line.split()[4:]
        unitcell = [int(float(i)) for i in unitcell]
        total_masses = masses * unitcell[0] * unitcell[1] * unitcell[2]
        uptake_mol_kg = uptake_total_molecule / total_masses * 1000
        error_mol_kg = error_total_molecule / total_masses * 1000
        result["uptake"] = uptake_mol_kg
        result["error"] = error_mol_kg
        result["calc_time_in_s"] = float(time_line.split()[2])
        result["success"] = True

        return result
    except Exception as e:
        raise ValueError(e)
        return result
