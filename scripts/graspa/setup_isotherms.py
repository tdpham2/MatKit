"""Set up gRASPA isotherm simulations from a YAML config file.

Usage:
    python setup_isotherms.py <config.yaml>
    python setup_isotherms.py  # defaults to isotherm_config.yaml
"""

import sys
from pathlib import Path

import yaml

from matkit.graspa import setup_batch

PRESSURE_TO_PA = {
    "pa": 1.0,
    "bar": 1e5,
    "kpa": 1e3,
    "atm": 101325.0,
}


def main():
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    else:
        config_path = Path(__file__).parent / "isotherm_config.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    pressure_unit = cfg.get("pressure_unit", "Pa").lower()
    if pressure_unit not in PRESSURE_TO_PA:
        raise ValueError(
            f"Unknown pressure_unit '{cfg['pressure_unit']}'. "
            f"Supported: {list(PRESSURE_TO_PA.keys())}"
        )
    factor = PRESSURE_TO_PA[pressure_unit]
    pressures_pa = [p * factor for p in cfg["pressures"]]

    adsorbates = [{"MoleculeName": name} for name in cfg["adsorbates"]]

    print(f"Config: {config_path}")
    print(f"CIF dir: {cfg['cif_dir']}")
    print(f"Adsorbates: {cfg['adsorbates']}")
    print(f"Temperatures: {cfg['temperatures']}")
    print(
        f"Pressures: {len(pressures_pa)} points "
        f"({cfg['pressures'][0]}-{cfg['pressures'][-1]} "
        f"{cfg.get('pressure_unit', 'Pa')})"
    )

    manifest = setup_batch(
        cif_dir=cfg["cif_dir"],
        outpath=cfg["outdir"],
        adsorbates=adsorbates,
        temperatures=cfg["temperatures"],
        pressures=pressures_pa,
        cutoff=cfg.get("cutoff", 12.8),
        n_cycle=cfg.get("cycles", 1000),
        max_workers=cfg.get("max_workers"),
    )

    print(f"\nSet up {len(manifest)} simulations in {cfg['outdir']}")
    print(f"Manifest: {cfg['outdir']}/simulations.jsonl")


if __name__ == "__main__":
    main()
