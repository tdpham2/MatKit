import numpy as np
import ase
import math
from numbers import Real


def calculate_cell_size(atoms: ase.Atoms, cutoff: float = 12.8) -> list[int]:
    """Method to calculate Unitcells (for periodic boundary condition).

    Args:
        atoms (ase.Atoms): ASE atom object
        cutoff (float, optional): Cutoff in Angstrom. Defaults to 12.8.

    Returns:
        list[int, int, int]: Unit cell in x, y and z
    """
    if (
        isinstance(cutoff, bool)
        or not isinstance(cutoff, Real)
        or not math.isfinite(cutoff)
        or cutoff <= 0
    ):
        raise ValueError("cutoff must be positive and finite")
    unit_cell = atoms.cell[:]
    if (
        not np.isfinite(unit_cell).all()
        or np.linalg.matrix_rank(unit_cell) != 3
    ):
        raise ValueError("Cell replication requires three independent vectors")
    # Unit cell vectors
    a = unit_cell[0]
    b = unit_cell[1]
    c = unit_cell[2]
    # minimum distances between unit cell faces
    wa = np.divide(
        np.linalg.norm(np.dot(np.cross(b, c), a)),
        np.linalg.norm(np.cross(b, c)),
    )
    wb = np.divide(
        np.linalg.norm(np.dot(np.cross(c, a), b)),
        np.linalg.norm(np.cross(c, a)),
    )
    wc = np.divide(
        np.linalg.norm(np.dot(np.cross(a, b), c)),
        np.linalg.norm(np.cross(a, b)),
    )

    uc_x = int(np.ceil(cutoff / (0.5 * wa)))
    uc_y = int(np.ceil(cutoff / (0.5 * wb)))
    uc_z = int(np.ceil(cutoff / (0.5 * wc)))

    return [uc_x, uc_y, uc_z]
