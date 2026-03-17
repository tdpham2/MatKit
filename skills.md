# MatKit - AI Development Guide

## Project Overview

MatKit is a modular Python toolkit (v0.1.0) for building, managing, and validating simulation input files for molecular and materials modeling software. It focuses on Grand Canonical Monte Carlo (GCMC) gas adsorption simulations in Metal-Organic Frameworks (MOFs) and porous materials. Developed at Argonne National Laboratory (ANL).

## Architecture

### Module-per-engine pattern

Each simulation engine has its own subpackage under `src/matkit/`:

```
src/matkit/
  cli.py                    # Click-based CLI entry point
  graspa/                   # gRASPA CUDA simulations
  graspa_sycl/              # gRASPA SYCL (Intel GPU) simulations
  raspa2/                   # RASPA2 simulations
  raspa3/                   # RASPA2 -> RASPA3 format conversion
  tobacco/                  # SMILES -> CIF linker generation for ToBaCCo
  mlip/                     # MACE-MP ML interatomic potential optimization
  orca/                     # ORCA quantum chemistry (stub)
  io/                       # File format converters (SMILES, PubChem JSON)
  utils/                    # Shared utilities (unit cell calc, solvent removal, CIF sampling)
```

### Common patterns across simulation modules

Each simulation engine module follows a two-function pattern:
- `setup_simulation()` / `setup_input_simulation()` - Copies template files, replaces placeholders (NCYCLE, TEMPERATURE, PRESSURE, CUTOFF, CIFFILE, UC_X/Y/Z, ADSORBATE), and generates input files
- `get_output_data()` - Parses simulation output log files and returns a dict with uptake, error, unit, and success fields

Template files are stored in `<module>/files/template*/` directories and use simple string replacement (not Jinja2).

### CLI structure

```
matkit <engine> <action>
  graspa setup|analyze
  graspa_sycl setup|analyze
  raspa2 setup|analyze
  tobacco create
```

## Tech Stack

- **Language**: Python >= 3.10
- **Core deps**: ase (atomic simulation), click (CLI), networkx (graph analysis), numpy
- **Optional deps**: rdkit (SMILES), mace-torch (MLIP), openbabel CLI (obabel)
- **Build**: setuptools via pyproject.toml (PEP 621)
- **Linting**: ruff (E, F rules, 80 char line length)
- **Testing**: pytest (tests/ directory)

## Key Conventions

- Use `pathlib.Path` for file path operations (not string slicing)
- Use `ase.io.read` / `ase.io.write` for all atomic structure I/O
- All `__init__.py` files re-export public APIs with `__all__`
- Template placeholders are UPPERCASE (e.g., NCYCLE, TEMPERATURE)
- Output parsing returns dicts with keys: success, uptake, error, unit
- Unit conversion supports: mol/kg, mg/g, g/L

## Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/

# Format
ruff format src/
```

## Important Notes

- The `data/` and `projects/` directories are gitignored (large data files)
- Template files in `files/template*/` are included as package data
- The `utils/chargemol/` scripts at repo root are standalone legacy scripts, not part of the package
- The `alcf/` directory contains HPC-specific installation documentation for Argonne systems
- ORCA module (`src/matkit/orca/`) is a non-functional stub awaiting implementation
