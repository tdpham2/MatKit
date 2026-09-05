# Running calculations with MatKit

These examples use public Python APIs and the existing MatKit CLI. MatKit
stages inputs, launches engines, parses outputs, and saves calculation bundles.
The Python programs contain no custom engine launchers, output parsers,
checkpoint writers, or charge-generation helpers.

| Program | Scenarios | Main API |
| --- | --- | --- |
| [zeopp.py](zeopp.py) | Pore analyses; multiple structures | `analyze_pores`, `run_batch` |
| [graspa.py](graspa.py) | Pure-component preparation, execution, pressure sweep | `prepare_adsorption`, `run_adsorption` |
| [graspa_mixture_setup.py](graspa_mixture_setup.py) | CO2/N2 mixture preparation | `matkit.graspa.setup_simulation` |
| [mlip.py](mlip.py) | Energy/forces, fixed-cell relaxation, batches; three backends | `evaluate`, `relax`, `run_batch` |

## Install and supply inputs

Run commands from the repository root after installing MatKit in your selected
environment:

```bash
python -m pip install -e .                 # Core, Zeo++, gRASPA preparation/CLI
python -m pip install -e '.[mlip]'          # Direct MACE, when needed
python -m pip install -e '.[rootstock]'     # Rootstock client, when needed
matkit capabilities --json
```

Zeo++'s `network` and CUDA gRASPA's `simulate` executables are separate
installations. For ALCHEMI, use the repository's
[compatible CUDA environment recipe](../alcf/polaris/mlip/README.md).
Capability discovery reports caller-side installation, not GPU or model
validation. Each worker interpreter must have MatKit and its selected backend
installed; Rootstock also needs an accessible deployment.

Replace the example filenames with your own structures. Each file must contain
one structure. Unified CIF loading requires an unambiguous atom mapping; use
explicit P1 structures when symmetry expansion would change the site count.
Zeo++ requires a fully periodic cell. gRASPA additionally requires finite
`_atom_site_charge` values whose sum matches `--net-charge` (default zero).
Use charges and force-field definitions appropriate for your framework.
The repository's small uncharged test CIF is not a scientific reference or a
gRASPA input.

For unified ALCHEMI, supply geometry-only extended XYZ with the correct cell
and PBC. The current implementation rejects atom arrays, constraints, and
bonds; even an ordinary CIF can carry ASE's `spacegroup_kinds` array and be
rejected.
The examples do not silently remove scientific metadata to bypass validation.

Use a fresh output directory for every invocation. Subprocess execution keeps
engine output in bundle logs. The programs exit 0 for accepted calculations
(or successful preparation), 1 for execution/input-file failures or
unconverged relaxation, and 2 for argument parsing or request/profile
validation errors.
Use `--help` on any program to see its options.

## Zeo++: pore geometry and screening

With `network` on PATH, run all five analyses:

```bash
python examples/zeopp.py framework.cif --outdir runs/pores
```

Select analyses and settings, or process several structures with the same
settings:

```bash
python examples/zeopp.py framework.cif --outdir runs/area-volume \
  --analysis sa --analysis vol --probe-radius 1.2 --channel-radius 1.8 \
  --num-samples 100000 --radii /path/to/custom.rad

python examples/zeopp.py framework_a.cif framework_b.cif \
  --analysis res --analysis sa --outdir runs/pore-screen
```

The default radii are MatKit's bundled UFF radii; high accuracy is enabled.
`res` reports Di/Df/Dif in angstrom, `sa` surface area, `vol` accessible volume,
`psd` a pore-size histogram, and `chan` channel dimensionalities. For sampled
analyses, `probe_radius` must not exceed `channel_radius`. Sample count and
probe choice affect the calculation and should be chosen for your study.
Unified Zeo++ batches run sequentially and retain one bundle per structure.

For an executable outside PATH, save `zeopp-execution.json`:

```json
{
  "executables": {"zeopp": ["/absolute/path/to/network"]},
  "timeout_s": 3600
}
```

Pass `--execution zeopp-execution.json` to the program. A direct API call is:

```python
from matkit.api import ExecutionConfig, PoreRequest, StructureRef, analyze_pores

result = analyze_pores(
    PoreRequest(
        structure=StructureRef(path="framework.cif"),
        analyses=["res", "sa"],
        num_samples=100000,
    ),
    output_dir="runs/pores-api",
    execution=ExecutionConfig(mode="subprocess"),
)
if not result.accepted:
    raise RuntimeError(result.failure)
print(result.payload.results["res"]["Di"])
print(result.payload.results["sa"]["ASA_m2_g"])
```

For the CLI, save `pores.json` beside `framework.cif`:

```json
{
  "operation": "pores",
  "structure": {"path": "framework.cif"},
  "analyses": ["res", "sa", "vol", "psd", "chan"],
  "probe_radius": 1.86,
  "channel_radius": 1.86,
  "num_samples": 100000
}
```

```bash
matkit pores --spec pores.json --outdir runs/pores-cli \
  --execution zeopp-execution.json
matkit inspect runs/pores-cli

# Alternatively, stage without an engine, then execute the same bundle.
matkit prepare --spec pores.json --outdir runs/pores-prepared
matkit execute runs/pores-prepared --execution zeopp-execution.json
```

CLI request paths are relative to their specification file. Python program
input paths are relative to the caller's working directory. Executable paths
in execution profiles should be absolute; commands are argument lists.

## gRASPA: pure-component adsorption and pressure sweeps

Preparation needs no CUDA or gRASPA executable:

```bash
python examples/graspa.py charged_framework.cif \
  --adsorbate CO2 --pressure-pa 100000 --prepare-only \
  --outdir runs/co2-prepared
```

This creates `runs/co2-prepared/00000`. Every pressure point gets an indexed
directory, including a single point. The generated input uses 298 K, a 12.8
angstrom cutoff, 1000 initialization cycles, 1000 equilibration cycles,
10000 production cycles, 5 blocks, and PR-EOS. These demonstration settings
do not establish adequate sampling. The CLI options expose temperature,
cycles, blocks, cutoff, net charge, uptake unit, and a custom template directory.

Save `graspa-execution.json` with your executable:

```json
{
  "executables": {"graspa": ["/absolute/path/to/gRASPA/bin/simulate"]},
  "timeout_s": 3600
}
```

Execute the prepared point, or run a fresh pressure sweep:

```bash
matkit execute runs/co2-prepared/00000 --execution graspa-execution.json
matkit adsorption analyze runs/co2-prepared/00000

python examples/graspa.py charged_framework.cif --adsorbate CO2 \
  --temperature-k 298 --pressure-pa 1000 --pressure-pa 10000 \
  --pressure-pa 100000 --outdir runs/co2-isotherm \
  --execution graspa-execution.json
```

Pressures above correspond to 0.01, 0.1, and 1 bar; **1 bar = 100000 Pa**.
Directories `00000`, `00001`, and `00002` preserve that input order. The
program prints one JSON run record per line, continues after a failed point,
and exits nonzero if any point fails. Each point uses `run_adsorption` because
`run_batch` requires identical pressures and other scientific settings.
For multiple frameworks at one pressure, a list of otherwise identical
`AdsorptionRequest` objects can use `run_batch`.

The same single-point operation through the CLI uses `adsorption.json` beside
your charged CIF:

```json
{
  "operation": "adsorption",
  "structure": {"path": "charged_framework.cif"},
  "adsorbate": "CO2",
  "temperature_K": 298,
  "pressure_Pa": 100000,
  "cutoff_angstrom": 12.8,
  "initialization_cycles": 1000,
  "equilibration_cycles": 1000,
  "production_cycles": 10000,
  "number_of_blocks": 5,
  "fugacity_coefficient": "PR-EOS",
  "net_charge": 0,
  "unit": "mol/kg"
}
```

```bash
matkit adsorption run --spec adsorption.json --outdir runs/co2-cli \
  --execution graspa-execution.json

# Or prepare now and execute later in an allocation with CUDA.
matkit adsorption prepare --spec adsorption.json --outdir runs/co2-staged
matkit execute runs/co2-staged --execution graspa-execution.json
```

Keep prepared bundles intact when copying them to another environment.
MatKit verifies staged inputs before execution; change the request or template
and prepare a new bundle when settings need changing.

Inspect uptake, engine-reported uncertainty, and heat of adsorption:

```python
from matkit.api import inspect_run

result = inspect_run("runs/co2-isotherm/00000")
if not result.accepted:
    raise RuntimeError(result.failure)
print(result.payload.uptake, result.payload.uncertainty, result.payload.unit)
print(result.payload.heat_of_adsorption, result.payload.heat_unit)
```

Uptake is absolute, per framework mass/volume; the default is mol/kg.
Heat is reported in kJ/mol with the engine's sign convention. Accepted output
does not establish equilibration or independent samples.

### CO2/N2 mixture preparation

```bash
python examples/graspa_mixture_setup.py charged_framework.cif \
  --outdir runs/co2-n2-inputs
```

The program directly calls `setup_simulation` with CO2/N2 mole fractions
0.15/0.85, 298 K, total pressure 100000 Pa, and the default template's
component placeholder. MatKit calculates unit-cell replication and copies
the input definitions. This legacy template uses 10000 initialization and
production cycles, zero equilibration cycles, and one block in this example.
Edit the explicit API settings in the program for another mixture.

This directory contains engine inputs, not a unified run bundle. Unified
mixture execution and result parsing are not implemented. The legacy
setup API also does not perform the unified charged-CIF validation. Check
charge/force-field suitability before executing through your engine workflow;
do not pass this directory to `matkit execute` or the single-component parser.

## MLIP: evaluation, relaxation, and batches

The program uses `MLIPMethod` for the checkpoint and one of `MACEAdapter`,
`RootstockAdapter`, or `AlchemiAdapter` for the implementation. Energy mode
requests energy and forces by default; `--property potential_energy` requests
only energy, and repeated `--property` options can include stress when the
model supports it. Energy is in eV, forces in eV/angstrom, and stress in
eV/angstrom^3 using the ASE convention.

Direct MACE defaults to CPU/float64:

```bash
python examples/mlip.py framework.cif --backend ase-mace --checkpoint medium \
  --outdir runs/mace-energy

python examples/mlip.py framework.cif --backend ase-mace --checkpoint medium \
  --driver relax --fmax 0.02 --steps 500 --outdir runs/mace-relax

python examples/mlip.py framework_a.cif framework_b.cif \
  --backend ase-mace --checkpoint medium --outdir runs/mace-batch
```

Relaxation uses fixed-cell FIRE. The default tolerance is 0.01 eV/angstrom
and maximum steps is 1000. An unconverged calculation retains its numerical
results but exits 1.

For CUDA, save `cuda-execution.json`:

```json
{"device": "cuda"}
```

Run one backend per process in its installed environment:

```bash
python examples/mlip.py framework.cif --backend ase-mace --checkpoint medium \
  --dtype float32 --execution cuda-execution.json --outdir runs/mace-cuda

python examples/mlip.py framework.cif --backend rootstock \
  --cluster polaris --checkpoint mace-mp-0-medium \
  --execution cuda-execution.json --outdir runs/rootstock-energy

python examples/mlip.py geometry_a.extxyz geometry_b.extxyz \
  --backend nvalchemi-mace --checkpoint medium --batch-size 16 \
  --execution cuda-execution.json --outdir runs/alchemi-energy

python examples/mlip.py geometry_a.extxyz geometry_b.extxyz \
  --backend nvalchemi-mace --checkpoint medium --batch-size 16 \
  --driver relax --fmax 0.02 --steps 500 \
  --execution cuda-execution.json --outdir runs/alchemi-relax
```

Replace `polaris` with your deployment, or use `--root /path/to/deployment`.
Add `--driver relax` for Rootstock relaxation; multiple inputs form a batch
on any backend. Rootstock precision belongs to its deployment settings, so
the program rejects `--dtype` for Rootstock. Its worker owns CUDA; the caller
does not need a CUDA-enabled PyTorch. ALCHEMI defaults to CUDA/float32 and
supports native chunking; direct MACE and Rootstock batches reuse one
calculator while executing items sequentially. Different model aliases do
not imply identical weights or comparable scientific results.

A direct Python relaxation has no example-specific helpers:

```python
from matkit.api import (
    ExecutionConfig, MLIPMethod, RelaxRequest, StructureRef, relax,
)

result = relax(
    RelaxRequest(
        structure=StructureRef(path="framework.cif"),
        method=MLIPMethod(checkpoint="medium"),
        fmax=0.02,
        steps=500,
    ),
    output_dir="runs/relax-api",
    execution=ExecutionConfig(mode="subprocess", device="cpu"),
)
print(result.accepted)
if result.payload is not None:
    print(result.payload.potential_energy, result.payload.converged)
    print(result.payload.final_structure)  # Relative to runs/relax-api.
```

For the unified CLI, save `energy.json` beside the input:

```json
{
  "operation": "evaluate",
  "structure": {"path": "framework.cif"},
  "method": {"checkpoint": "medium"},
  "adapter": {"type": "ase-mace", "dtype": "float32"},
  "properties": ["potential_energy", "forces"]
}
```

Save a relaxation specification as `relaxation.json`:

```json
{
  "operation": "relax",
  "structure": {"path": "framework.cif"},
  "method": {"checkpoint": "medium"},
  "adapter": {"type": "ase-mace", "dtype": "float32"},
  "fmax": 0.02,
  "steps": 500
}
```

```bash
matkit evaluate --spec energy.json --outdir runs/energy-cli \
  --execution cuda-execution.json
matkit relax --spec relaxation.json --outdir runs/relax-cli \
  --execution cuda-execution.json
```

For Rootstock, change `adapter` to
`{"type": "rootstock", "cluster": "polaris"}` and use its checkpoint ID.
For ALCHEMI, use
`{"type": "nvalchemi-mace", "dtype": "float32", "batch_size": 16}`
and a geometry-only extended XYZ input.

For CLI batches, save a JSON list of complete requests as `batch.json`.
Only the structure fields may differ within a batch. For example:

```json
[
  {
    "operation": "evaluate",
    "structure": {"path": "framework_a.cif"},
    "method": {"checkpoint": "medium"},
    "properties": ["potential_energy", "forces"]
  },
  {
    "operation": "evaluate",
    "structure": {"path": "framework_b.cif"},
    "method": {"checkpoint": "medium"},
    "properties": ["potential_energy", "forces"]
  }
]
```

```bash
matkit batch --spec batch.json --outdir runs/batch-cli
matkit inspect runs/batch-cli
matkit inspect runs/batch-cli/00000
```

## Read results and understand validation

Each unified run retains `request.json`, original/supporting inputs under
`inputs/`, engine files under `work/`, and a `run.json` record. Completed or
failed executions normally also have `result.json`. Worker stdout/stderr logs
are at the bundle root. Zeo++ logs are `work/engine.stdout.log` and
`work/engine.stderr.log`; gRASPA stdout is `work/raspa.log`.

MLIP final geometry is `work/final_structure.extxyz`; keep its adjacent
`.metadata.json` sidecar for atom correspondence and supported metadata.
Charges derived from the original geometry are invalidated after relaxation;
the relaxed output is not automatically a charged gRASPA input.

Batch roots contain `batch_manifest.json` and numbered item bundles. A batch
uses one supervised worker, so its worker logs live at the batch root;
external-engine logs remain in each item's `work/` directory. A batch
may retain successful items alongside failures or unconverged optimizations.
Use `inspect_run` on an item's directory to read its scientific payload;
`matkit inspect` also reads batch manifests. Inspection exit 0 means the
record was readable, even if the calculation failed. `RunResult.accepted`
and `BatchResult.accepted` are computed Python properties; use the run
command's exit code and stored state/checks when consuming CLI JSON.

The examples are checked with synthetic engines and calculator doubles.
No real Zeo++, gRASPA, MACE, Rootstock, or ALCHEMI execution was available
during local validation. For opt-in execution evidence use
[unified_smoke.py](unified_smoke.py) or the
[Polaris MLIP smoke recipe](../alcf/polaris/mlip/README.md).
See the [capability inventory](../docs/capabilities.md) for validation limits.
