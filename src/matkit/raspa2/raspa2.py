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
