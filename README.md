# MatKit

**MatKit** is a modular Python toolkit for building, managing, and validating simulation input files for molecular and materials modeling software. It is designed to accelerate research workflows for gas adsorption simulations in Metal-Organic Frameworks (MOFs) and related porous materials.

## Supported Simulation Engines

- **gRASPA** (CUDA) -- Grand Canonical Monte Carlo on NVIDIA GPUs
- **gRASPA SYCL** -- Grand Canonical Monte Carlo on Intel GPUs
- **RASPA2** -- Classical GCMC simulations
- **RASPA3** -- Force field format conversion from RASPA2
- **Zeo++** -- Pore geometry analysis (pore diameters, surface area, volume, channels)
- **MACE-MP** -- ML interatomic potential geometry/cell optimization
- **ORCA** -- Quantum chemistry (planned)

## Features

- Template-based simulation input generation
- CIF/XYZ/POSCAR file handling via ASE
- Automatic unit cell replication calculation for periodic boundary conditions
- Force field management and RASPA2-to-RASPA3 format conversion
- Simulation output parsing with unit conversion (mol/kg, mg/g, g/L)
- Pore geometry analysis via Zeo++ (Di/Df/Dif, surface area, volume, PSD, channels)
- Solvent/ion removal from MOF structures using graph-based connectivity
- SMILES to CIF linker generation for ToBaCCo MOF construction
- CLI interface for all major operations
- Random CIF sampling for high-throughput screening

## Installation

MatKit requires Python >= 3.10 and can be installed with `pip`:

```bash
git clone https://github.com/tdpham2/MatKit.git
cd MatKit
pip install -e .
```

### Optional Dependencies

```bash
# For SMILES conversion (rdkit)
pip install -e ".[rdkit]"

# For ML interatomic potentials (MACE)
pip install -e ".[mlip]"

# All optional dependencies
pip install -e ".[all]"

# Development dependencies (pytest, ruff)
pip install -e ".[dev]"
```

## CLI Usage

```bash
# Setup a gRASPA simulation
matkit graspa setup --cif structure.cif --outdir sim_output --adsorbate CO2 --temp 298 --pressure 1e5

# Analyze gRASPA results
matkit graspa analyze --path sim_output --unit mol/kg

# Setup a RASPA2 simulation
matkit raspa2 setup --cif structure.cif --outdir sim_output --adsorbate CO2

# Setup a gRASPA SYCL simulation
matkit graspa_sycl setup --cif structure.cif --outdir sim_output --adsorbate CO2

# Create a ToBaCCo linker from SMILES
matkit tobacco create --smiles "Nc1nc[nH]n1" --site N3 --site N5 --out linker.cif

# Run Zeo++ pore analysis (requires network binary)
matkit zeopp run --cif structure.cif --analysis res --radii UFF.rad
matkit zeopp run --cif structure.cif --analysis res --analysis sa --radii UFF.rad --num-samples 100000

# Parse existing Zeo++ output files
matkit zeopp analyze --path output_dir/
```

## Python API

```python
from matkit.graspa import setup_simulation, get_output_data
from matkit.utils import calculate_cell_size, remove_solvent, sample_cifs
from matkit.raspa3 import save_force_field

# Setup a gRASPA simulation
adsorbates = [{"MoleculeName": "CO2"}, {"MoleculeName": "N2"}]
setup_simulation("structure.cif", "output/", adsorbates, temperature=298.0)

# Parse simulation results
result = get_output_data("output/", unit="mol/kg")

# Remove solvent from a MOF CIF
remove_solvent("mof_with_solvent.cif", "mof_clean.cif")

# Convert RASPA2 force field to RASPA3 JSON
save_force_field("pseudo_atoms.def", "force_field.def", "output/")

# Run Zeo++ pore analysis
from matkit.zeopp import run_zeopp, get_output_data

result = run_zeopp(
    "structure.cif",
    analyses=["res", "sa"],
    radii_file="UFF.rad",
    num_samples=100000,
)
print(result["results"]["res"])   # {'Di': 18.5, 'Df': 8.0, 'Dif': 10.9, ...}
print(result["results"]["sa"])    # {'ASA': 4004.7, 'ASA_m2_g': 3918.3, ...}

# Parse existing Zeo++ output files
result = get_output_data("output_dir/")
```

## License

MIT License - Copyright 2025 Thang Pham (Argonne National Laboratory)
