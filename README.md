# MatKit

**MatKit** is a modular Python toolkit for building, managing, and validating simulation input files for molecular and materials modeling software. It is designed to accelerate research workflows for gas adsorption simulations in Metal-Organic Frameworks (MOFs) and related porous materials.

## Supported Simulation Engines

- **gRASPA** (CUDA) -- Grand Canonical Monte Carlo on NVIDIA GPUs
- **gRASPA SYCL** -- Grand Canonical Monte Carlo on Intel GPUs
- **RASPA2** -- Classical GCMC simulations
- **RASPA3** -- Force field format conversion from RASPA2
- **Zeo++** -- Pore geometry analysis (pore diameters, surface area, volume, channels)
- **MLIPs** -- experimental direct MACE, Rootstock, and NVIDIA ALCHEMI adapters
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

# Lightweight access to cluster-managed Rootstock models
pip install -e ".[rootstock]"

# NVIDIA ALCHEMI MACE support (install a matching CUDA extra too)
pip install -e ".[nvalchemi_mace]"

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

# Run MACE directly through ASE
matkit mlip run --input structure.cif --backend ase-mace \
  --checkpoint medium --device cuda --dtype float32

# Run a Rootstock checkpoint already deployed on Polaris
matkit mlip run --input structure.cif --backend rootstock \
  --checkpoint mace-mp-0-medium --cluster polaris --device cuda

# Run a native NVIDIA ALCHEMI batch
matkit mlip run-batch --input-dir cifs --backend nvalchemi-mace \
  --checkpoint medium --device cuda --batch-size 16
```

### GPU examples

[`examples/mlip_gpu.py`](examples/mlip_gpu.py) runs one backend per Python
process so GPU runtime state is isolated. It accepts one or more ASE-readable
structure files and writes a manifest plus one JSON result per input.

```bash
# Direct MACE calculator through ASE
python examples/mlip_gpu.py --backend ase-mace structure.cif

# Rootstock-managed MACE checkpoint on Polaris
python examples/mlip_gpu.py --backend rootstock \
  --cluster polaris --checkpoint mace-mp-0-medium structure.cif

# NVIDIA ALCHEMI MACE with native GPU batching
python examples/mlip_gpu.py --backend nvalchemi-mace \
  --checkpoint medium --batch-size 16 structures/*.cif
```

All three commands force `device="cuda"` and use `float32` where the backend
exposes a dtype. Pass `--driver opt` for a fixed-cell geometry optimization.

The new MLIP adapters remain **experimental** until a real execution is
recorded for the backend, capability, and environment. CPU tests exercise
validation and adapter contracts; they do not establish GPU compatibility or
scientific accuracy. See the [Polaris smoke recipe](alcf/polaris/mlip/README.md)
for opt-in energy, optimization, and native batch checks.

### MLIP outcomes and batch files

`success` means a calculation produced valid numerical results. Optimization
convergence is reported separately as `converged`; usable unconverged results
are retained. The CLI and GPU example exit **1** if any item fails or any
requested optimization remains unconverged, **2** for invalid arguments, and
**0** only when the requested calculations succeed. JSON summaries go to stdout;
MatKit's failure diagnostics go to stderr. Optional engines may also emit logs.

Batch result files and `batch_manifest.json` are replaced atomically as each
item completes (after each native ALCHEMI chunk). The manifest starts `running`
with pending items, then finishes `completed`, `partial`, or `failure`. It
includes `pending`, `unconverged`, per-item `converged`, and a run-level `error`.
An execution-complete batch can still contain unconverged optimizations.
Catchable orchestration failures record `interrupted`; abrupt termination can
leave `running`. Previously committed result files remain available. An
unrecoverable filesystem failure may also prevent the last manifest update.

Use a **fresh output directory for every batch**. Existing manifests or active
batch locks are refused; automatic resume and retries are deferred. The GPU
example accepts `--output-dir` and the CLI accepts `--outdir` for reruns.

Explicit options that do not apply to the selected backend/driver are errors:

- `--optimizer`, `--fmax`, and `--steps` require `--driver opt`.
- `--dt`, `--compile-model`, `--enable-cueq`, `--batch-size`, and `--max-atoms`
  belong to ALCHEMI; `--dt` also requires optimization. `max_atoms` limits chunk
  grouping; a single larger structure is still processed alone.
- Rootstock precision uses a model-supported `--setup-kwarg`, not `--dtype`.
  Rootstock workers own their GPU environment; the caller does not need CUDA.
- The `mace_anicc` factory controls precision and rejects an explicit CLI dtype.
  The configuration records requested settings; comprehensive resolved model
  provenance is future work.

MLIP optional packages may require a newer Python than MatKit's core. ALCHEMI
0.2 requires Python 3.11–3.13 and a matching CUDA stack; the Polaris recipe uses
Python 3.12/CUDA 12. Its MACE extra pins 0.3.15, which conflicts with ChemGraph's
currently required MACE >=0.3.16. Use a MatKit-specific environment rather than
combining these stacks. The `all` extra does not include Rootstock or ALCHEMI.

## Python API

The experimental [unified operation API](docs/unified-api.md) provides shared
Python/CLI requests, relocatable calculation bundles, and optional MCP tools.
It covers MLIP evaluation/relaxation, Zeo++ analysis, and single-component
gRASPA CUDA execution. See the [capability inventory](docs/capabilities.md)
for implementation status, environment requirements, and validation limits.

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

# Agent-free, runtime-selectable MLIP execution
from matkit.mlip import (
    ASEMACEConfig,
    MLIPCalculationConfig,
    run_mlip,
)

result = run_mlip(
    "structure.cif",
    ASEMACEConfig(
        checkpoint="medium",
        device="cuda",
        dtype="float32",
    ),
    MLIPCalculationConfig(driver="energy"),
    output_file="mace_result.json",
)
```

## Development roadmap

The [MatKit roadmap](docs/plans/matkit-roadmap.md) records the unified API
baseline and the correctness follow-up for engine output, scientific acceptance,
and isotherm units. Next come real-engine reference evidence, a validated charge
handoff and scripted porous-material workflow, recovery and scaling, optional
MOFforge integration, and ChemGraph evaluations.

[MOFforge](https://github.com/tdpham2/mofforge) prepares and manipulates structures;
MatKit owns scientific relaxation, charge prediction, pore calculations, and
simulations; ChemGraph coordinates workflows across both toolkits. MOFforge
integration remains optional and is a future milestone.

Single-isotherm data may use bar or Pa; pressures are converted into the first
recognized unit before sorting. Each curve must use one temperature and
consistent uptake and heat-of-adsorption units. Duplicate physical pressure
points are rejected; aggregate replicate measurements explicitly before plotting.

## License

MIT License - Copyright 2025 Thang Pham (Argonne National Laboratory)
