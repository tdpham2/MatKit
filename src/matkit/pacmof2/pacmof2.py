"""PACMOF2 charge prediction wrapper for matkit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from pacmof2 import get_charges

from matkit.types import PACMOF2Result


def run_charge_prediction(
    cif_path: str,
    output_dir: str,
    identifier: str = "_pacmof",
    net_charge: Union[int, float, dict] = 0,
    adjust_charge_method: str = "mean",
) -> PACMOF2Result:
    """Run PACMOF2 charge prediction on CIF file(s).

    Args:
        cif_path: Path to a single CIF file or a directory
            containing CIF files.
        output_dir: Directory where output CIF files with
            predicted charges will be written.
        identifier: Suffix appended to output filenames
            (default: "_pacmof").
        net_charge: Net charge for ionic MOFs. Use 0 for
            neutral MOFs (default), an int/float for a single
            structure, or a dict mapping CIF filenames to
            charges for batch ionic processing.
        adjust_charge_method: Method to enforce net charge
            constraint. Either "mean" (default) or
            "magnitude".

    Returns:
        PACMOF2Result dict with keys: success, output_dir,
        num_structures, error.
    """
    cif_path = Path(cif_path)
    output_dir = Path(output_dir)

    if not cif_path.exists():
        raise FileNotFoundError(f"CIF path not found: {cif_path}")

    multiple_cifs = cif_path.is_dir()

    if multiple_cifs:
        n = len(list(cif_path.glob("*.cif")))
        if n == 0:
            raise FileNotFoundError(
                f"No .cif files found in {cif_path}"
            )
    else:
        n = 1

    # If net_charge is a string path, load the JSON file
    if isinstance(net_charge, str):
        nc_path = Path(net_charge)
        if nc_path.is_file():
            with open(nc_path) as f:
                net_charge = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    get_charges(
        path_to_cif=str(cif_path),
        output_path=str(output_dir),
        identifier=identifier,
        multiple_cifs=multiple_cifs,
        adjust_charge_method=adjust_charge_method,
        net_charge=net_charge,
    )

    return PACMOF2Result(
        success=True,
        output_dir=str(output_dir),
        num_structures=n,
        error=None,
    )
