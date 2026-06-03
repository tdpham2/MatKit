"""Set up pygRASPA isotherm simulations (ML-potential GCMC) from YAML.

Usage:
    python setup_isotherms.py <config.yaml>
    python setup_isotherms.py  # defaults to isotherm_config.yaml

Supports two modes via the 'mode' field in the YAML config:
    - single: each adsorbate runs as independent single-component sims
    - mixture: all adsorbates run together in one sim (requires MolFraction)
"""

import sys
from pathlib import Path

import yaml

from matkit.pygraspa import setup_batch

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

    mode = cfg.get("mode", "single")
    ml = cfg["ml"]
    model_path = ml["model_path"]
    model_type = ml.get("model_type", "FAIRChem-esen")
    task = ml.get("task")
    ecomp_lookup = ml["ecomps"]  # {adsorbate_name: float}
    run_mode = ml.get("run_mode", "run-auto")
    save_poscar = ml.get("save_poscar", False)

    print(f"Config: {config_path}")
    print(f"Mode: {mode}")
    print(f"ML model: {model_path} ({model_type})")
    print(f"CIF dir: {cfg['cif_dir']}")
    print(f"Temperatures: {cfg['temperatures']}")
    print(
        f"Pressures: {len(pressures_pa)} points "
        f"({cfg['pressures'][0]}-{cfg['pressures'][-1]} "
        f"{cfg.get('pressure_unit', 'Pa')})"
    )

    common_kwargs = dict(
        cif_dir=cfg["cif_dir"],
        temperatures=cfg["temperatures"],
        pressures=pressures_pa,
        cutoff=cfg.get("cutoff", 12.8),
        n_cycle=cfg.get("cycles", 1000),
        max_workers=cfg.get("max_workers"),
        model_path=model_path,
        model_type=model_type,
        task=task,
        mode=run_mode,
        save_poscar=save_poscar,
    )

    if mode == "single":
        print(f"Adsorbates: {cfg['adsorbates']} (each runs independently)")
        total = 0
        for ads_name in cfg["adsorbates"]:
            if ads_name not in ecomp_lookup:
                raise ValueError(
                    f"E_comp for adsorbate {ads_name!r} not provided "
                    "under ml.ecomps in the YAML config."
                )
            ads_outdir = str(Path(cfg["outdir"]) / ads_name)
            adsorbates = [{"MoleculeName": ads_name}]
            manifest = setup_batch(
                outpath=ads_outdir,
                adsorbates=adsorbates,
                E_comps=[ecomp_lookup[ads_name]],
                template_dir=cfg.get("template_dir", "template"),
                **common_kwargs,
            )
            print(
                f"  {ads_name}: {len(manifest)} simulations in {ads_outdir}"
            )
            total += len(manifest)
        print(f"\nTotal: {total} simulations")

    elif mode == "mixture":
        adsorbates = []
        for ad in cfg["adsorbates"]:
            entry = {"MoleculeName": ad["name"]}
            for k, v in ad.items():
                if k != "name":
                    entry[k] = v
            adsorbates.append(entry)
        names = [ad["MoleculeName"] for ad in adsorbates]
        print(f"Adsorbates: {names} (mixture)")

        ecomps = []
        for name in names:
            if name not in ecomp_lookup:
                raise ValueError(
                    f"E_comp for adsorbate {name!r} not provided "
                    "under ml.ecomps."
                )
            ecomps.append(ecomp_lookup[name])

        template_dir = cfg.get("template_dir", "template_mixture_isotherm")
        manifest = setup_batch(
            outpath=cfg["outdir"],
            adsorbates=adsorbates,
            E_comps=ecomps,
            template_dir=template_dir,
            **common_kwargs,
        )
        print(f"\nSet up {len(manifest)} simulations in {cfg['outdir']}")

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'single' or 'mixture'.")

    print(f"Manifest: {cfg['outdir']}/simulations.jsonl")


if __name__ == "__main__":
    main()
