from mace.calculators import mace_mp
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter


def run_opt_mace(
    fname: str,
    run_type: str = "geo_opt",
    steps: float = 1000,
    fmax: float = 1e-3,
    model: str = "medium",
    device: str = "cpu",
    dispersion: bool = True,
    default_dtype: str = "float64",
    write_traj: bool = False,
    output_fname: str = None,
):
    try:
        atoms = ase_read(fname)
        atoms.calc = mace_mp(
            model=model,
            dispersion=dispersion,
            default_dtype=default_dtype,
            device=device,
        )
        if output_fname is None:
            filename = fname.split(".")[0]
            ext = fname.split(".")[-1]

            output_fname = filename + f"_opt_{model}" + "." + ext
        if run_type == "geo_opt":
            if write_traj is True:
                dyn = BFGS(atoms, trajectory=f"{filename}.traj")
            else:
                dyn = BFGS(atoms)
            dyn.run(fmax=fmax, steps=steps)
        elif run_type == "cell_opt":
            ecf = ExpCellFilter(atoms)
            if write_traj is True:
                dyn = BFGS(ecf, trajectory=f"{filename}.traj")
            else:
                dyn = BFGS(ecf)
            dyn.run(fmax=fmax, steps=steps)
        elif run_type == "geo_opt_cell_opt":
            # First do a geometry optimization
            if write_traj is True:
                dyn1 = BFGS(atoms, trajectory=f"{filename}_geo_opt.traj")
                dyn1.run(fmax=fmax, steps=steps)
                ecf = ExpCellFilter(atoms)
                dyn = BFGS(ecf, trajectory=f"{filename}_cell_opt.traj")
                dyn1.run(fmax=fmax, steps=steps)

            else:
                dyn1 = BFGS(atoms)
                dyn1.run(fmax=fmax, steps=steps)
                ecf = ExpCellFilter(atoms)
                dyn = BFGS(ecf)
                dyn1.run(fmax=fmax, steps=steps)

        else:
            print(
                (f"run_type {run_type} is not supported."),
                ("Options are 'geo_opt', cell_opt' and 'geo_opt_cell_opt'"),
            )
            return "Error"

        ase_write(output_fname, atoms)

    except Exception as e:
        print(e)
