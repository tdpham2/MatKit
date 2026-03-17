# Changelog

All notable changes to MatKit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CLI interface (`matkit` command) with subcommands for graspa, graspa_sycl, raspa2, and tobacco
- `__init__.py` re-exports for all subpackages (io, mlip, orca, utils, graspa, graspa_sycl, raspa2, raspa3, tobacco)
- Optional dependency groups in pyproject.toml: `rdkit`, `mlip`, `all`, `dev`
- Comprehensive test suite with 42 pytest tests covering utils, raspa3, graspa, and CLI
- GitHub Actions CI workflow (lint, test on Python 3.10-3.12, build)
- `skills.md` for AI-assisted development context
- `CHANGELOG.md`
- `conftest.py` for pytest configuration

### Fixed
- **MACE optimizer bug**: `geo_opt_cell_opt` mode called `dyn1.run()` instead of `dyn.run()` for cell optimization step, meaning the cell was never actually optimized
- **Missing f-string**: `raspa2.py` error message `"Unit {unit} is not supported"` was missing `f` prefix
- **Unreachable code**: Removed dead `return result` after `raise ValueError` in `graspa.py` and `graspa_sycl.py`
- **Shell injection vulnerability**: Replaced `subprocess.call(shell=True)` with safe `subprocess.run([...])` in tobacco module
- **Cross-platform compatibility**: Replaced `sed` subprocess call with Python string operations in `update_cif_with_connection_site()`
- **Unsafe file removal**: Replaced `subprocess.call("rm ...")` with `Path.unlink()` in tobacco module
- Incorrect error message "Source directory does not exist" changed to "CIF file does not exist"
- Fragile CIF name extraction (`str.split("/")[-1][:-4]`) replaced with `Path.stem`
- Invalid type hint `list[int, int, int]` corrected to `list[int]` in `unitcell_calculator.py`
- `steps` parameter type changed from `float` to `int` in `mace_opt.py`
- Removed unused imports (`os`, `glob`, `Path`) across multiple modules
- Cleaned up intermediate file leaks in `create_linker()`

### Changed
- Updated README.md with accurate feature list, CLI usage, and Python API examples
- Added `numpy>=1.22` to core dependencies
- Removed `tests/` from `.gitignore` to allow version-controlled tests
- Standardized `os.path` usage to `pathlib.Path` across all modules

## [0.1.0] - 2025-01-29

### Added
- Initial release
- gRASPA (CUDA) simulation setup and output parsing
- gRASPA SYCL simulation setup and output parsing
- RASPA2 simulation setup and output parsing
- RASPA3 force field format conversion (RASPA2 -> JSON)
- ToBaCCo linker generation from SMILES
- MACE-MP geometry/cell optimization
- Unit cell replication calculator
- Solvent/ion removal from MOF CIF files
- CIF file sampler
- SMILES to XYZ converter
- PubChem JSON to XYZ converter
