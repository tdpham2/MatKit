# Unified scientific operations

The experimental `matkit.api` interface runs without an agent, MOFforge, or
ChemGraph. Python, the operation CLI, and optional MCP tools share validated
requests, scientific results, and persistent run bundles. Existing engine APIs,
CLI commands, defaults, and result layouts remain available.

## Install and select an environment

```bash
pip install .                  # core API, preparation, parsing, CLI
pip install '.[mlip]'           # direct MACE
pip install '.[rootstock]'      # Rootstock client
pip install '.[mcp]'            # optional stdio server (MCP SDK 2)
```

Zeo++ and gRASPA executables must be installed separately. See
[Polaris MLIP environments](../alcf/polaris/mlip/README.md) for the experimental
GPU adapters. Incompatible model packages require separate installations and
interpreters. Core imports and capability discovery do not load models or CUDA.

```bash
matkit capabilities --json
```

Discovery reports caller-side availability, restrictions, and evidence
separately. A remote Rootstock deployment or a configured worker environment can
differ from the caller. Species coverage, stress support, and scientific
applicability depend on the selected model. See the
[capability inventory](capabilities.md).

## Python

```python
from matkit.api import (
    EvaluateRequest, ExecutionConfig, MLIPMethod, StructureRef, evaluate,
)

request = EvaluateRequest(
    structure=StructureRef(path="structure.cif"),
    method=MLIPMethod(checkpoint="medium"),
    properties=["potential_energy", "forces"],
)
result = evaluate(request, output_dir="runs/energy")
print(result.accepted, result.payload)
```

The method identifies the science. `MACEAdapter`, `RootstockAdapter`, and
`AlchemiAdapter` select the implementation; `ExecutionConfig` selects the device,
interpreter, environment overrides, and executable commands. Defaults preserve
the existing adapters: direct MACE uses CPU/float64, ALCHEMI CUDA/float32,
and Rootstock uses its deployment configuration.

`EvaluateRequest` defaults to energy only. Forces and stress must be explicitly
requested, and unavailable requested properties fail. `RelaxRequest` uses
fixed-cell FIRE with `fmax=0.01` eV/angstrom and 1000 maximum steps by default.
ALCHEMI supports only FIRE. Cell optimization and MD remain on legacy APIs.

```python
from matkit.api import RelaxRequest, relax, run_batch

relaxation = RelaxRequest(
    structure=request.structure, method=request.method, fmax=0.02, steps=500,
)
result = relax(relaxation, output_dir="runs/relaxation")
assert result.accepted       # includes requested force convergence

# Requests must share operation, method, adapter, and scientific settings.
# Repeated basenames remain distinct and results preserve input ordering.
batch = run_batch([request, request], output_dir="runs/batch")
```

Batches retain one calculator per request group; ALCHEMI retains native chunking.
Per-item native timings include shared chunk work and are not throughput
benchmarks. Startup is recorded separately. Inputs remain accumulated in memory
in this first release.

## CLI and prepared calculations

Save a specification as `pores.json`:

```json
{
  "operation": "pores",
  "structure": {"path": "structure.cif"},
  "analyses": ["res", "sa"],
  "probe_radius": 1.86,
  "channel_radius": 1.86,
  "num_samples": 100000
}
```

Paths in CLI specifications are relative to the specification file. Model names
remain identifiers unless they resolve to local checkpoint files. Python paths
are relative to the caller's working directory.

For Zeo++ surface area, volume, and PSD, `channel_radius` controls accessibility
and `probe_radius` controls sampling. The probe radius must not exceed the
channel radius; equal radii remain the default. Both APIs pass these values in
Zeo++'s documented channel/probe order. All requested outputs must be present,
complete, and finite. Collection accepts multiline channel output and the
default `.psd_histo` histogram filename, retaining explicitly named legacy
`.psd` files and existing result keys. See the
[Zeo++ command and output examples](https://www.zeoplusplus.org/examples.html).

```bash
matkit pores --spec pores.json --outdir runs/pores
matkit prepare --spec pores.json --outdir runs/prepared
# Copy the entire prepared directory to the execution environment if needed.
matkit execute runs/prepared --execution execution.json
matkit inspect runs/prepared
```

Example execution profile (`execution.json`):

```json
{
  "python": "/path/to/matkit-environment/bin/python",
  "device": "cuda",
  "executables": {
    "zeopp": ["/path/to/network"],
    "graspa": ["/path/to/gRASPA/bin/simulate"]
  },
  "environment": {"OMP_NUM_THREADS": "1"},
  "timeout_s": 3600
}
```

Executable values are argument lists, never shell fragments. Every worker
interpreter must have a compatible MatKit installation and the selected engine
dependencies. Environment overrides travel through the process environment;
their values are not copied into worker configuration artifacts. Selected
numerical runtime settings are recorded in provenance.

`matkit evaluate` and `matkit relax` accept their corresponding specifications.
`matkit batch --spec requests.json --outdir runs/batch` accepts a JSON list of
homogeneous requests. CLI calculations run in a subprocess so engine output
goes to logs. Scientific JSON goes to stdout, diagnostics to stderr. Exit 2
means invalid arguments; exit 1 means a failed calculation or failed required
convergence; exit 0 means accepted requested results. Inspection can exit 0
while reporting a failed calculation because reading the record succeeded.

## Single-component gRASPA CUDA

The unified path requires a fully periodic CIF with finite, atom-mapped
`_atom_site_charge` values summing to the requested net charge. It copies the
charged CIF unchanged. Disordered sites and symmetry expansions without a
provable atom mapping are rejected; provide an explicit P1 structure.

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

These are example settings, not evidence of equilibration or adequate sampling.
The default template and its force-field definitions are staged and hashed.
`template_dir` may select a complete custom template directory. The first
unified parser handles one component; existing mixture setup APIs are unchanged.

```bash
matkit adsorption prepare --spec adsorption.json --outdir runs/adsorption
matkit execute runs/adsorption --execution execution.json
matkit adsorption analyze runs/adsorption
```

Preparation requires neither CUDA nor the gRASPA binary. Execution requires a
zero engine exit code and complete, finite requested results. Sampling quality
remains `unknown`; uncertainty is reported as supplied by the engine, with an
unknown method unless independently established. An engine random seed not
resolved by the adapter remains unknown. This does not promise bitwise replay.

For manual engine launches, keep `work/raspa.log` and record the actual engine
return code in `exit.json` as `{"returncode": 0}` (using the actual code).
`analyze_adsorption` can then collect the prepared bundle. Prefer `matkit execute`
to capture these records reliably. The [Polaris PBS example](../alcf/polaris/unified/run.pbs)
runs inside an existing allocation; MatKit does not submit scheduler jobs.

## Contracts and artifacts

Requests use `schema_name="matkit.request"`, results `"matkit.run"`, and batches
`"matkit.batch"`, each at version 1. Legacy MLIP version-1 files retain their
original meaning. Request schemas are available through
`matkit.api.models.REQUEST_ADAPTER.json_schema()` and result schemas through
`RunResult.model_json_schema()`.

`state` records execution (`prepared`, `running`, `completed`, `failed`, or
`interrupted`). `numerical_validity` and named scientific checks are separate.
An unconverged relaxation can have valid numerical results and `state=completed`
while `accepted` is false. Unknown sampling quality does not become a claim of
equilibrium. Failure information is separate from adsorption uncertainty.

Energy uses `potential_energy` in eV with a model-specific reference; forces
use eV/angstrom; stress uses the ASE Cartesian convention in eV/angstrom³.
The shared interface does not make energies from different methods comparable.
Pore result quantities retain their documented Zeo++ units. Adsorption uptake
is absolute loading on the framework basis, with component and unit recorded.

Each bundle contains:

- `request.json`: staged request using bundle-relative input references.
- `inputs/`: original structure, metadata sidecar, and supporting files.
- `work/`: engine inputs, logs, and output artifacts.
- `run.json`: execution record, settings, provenance, and artifact inventory.
- `result.json`: authoritative committed result after completion/failure.
- `execution.json`, `command.json`, and `exit.json` when applicable.

Input files, sidecars, templates, force fields, and available local model files
are hashed. Unresolved checkpoint identities/versions remain explicitly unknown.
Prepared inputs are checked before execution. Keep the whole bundle together
when moving it; all artifact inventory paths are relative to its root.

Structure sidecars preserve species, coordinates, cell, periodicity, atom IDs,
supported arrays, labels, bonds, and constraints. Unsupported objects and
unmappable inputs fail explicitly. Changed geometries receive parent lineage
and lose inherited derived charges/energies/pore metadata. Recompute charges
before sending a relaxed geometry to charge-dependent adsorption calculations.
Do not move a generated structure without its `.metadata.json` sidecar.

Use a fresh directory for each run/batch. Completed results are committed before
manifest updates; interrupted work remains inspectable. If later teardown or
manifest persistence fails, inspection reports that orchestration failure while
retaining the committed numerical payload. An unreadable manifest does not
prevent recovery of a valid committed result. A hard process kill
can leave a running record or stale lock; automatic resume and restart are not
implemented. Copy a completed result for inspection, and prepare a fresh bundle
for another execution. Engine-specific restart requires additional future work.

## MCP

```bash
matkit-mcp --run-root /scratch/matkit-runs --input-root /data/structures \
  --profiles profiles.json --timeout 60
```

`profiles.json` maps profile names to execution profiles, for example
`{"default": {"executables": {"zeopp": ["/path/to/network"]}}}`. Callers select
a profile name; executable configuration stays with the server. Input paths
must lie inside a configured input root or run root. Use repeated `--input-root`
arguments for model and structure directories.

Tools are `matkit_capabilities`, `matkit_evaluate`, `matkit_relax`, `matkit_pores`,
`matkit_prepare`, `matkit_prepare_adsorption`, and `matkit_inspect`. Select a
catalog with `--tools matkit_capabilities,matkit_pores,matkit_inspect`.

Calls return scalar summaries, scientific checks, failures, and artifact links
such as `matkit://runs/<run_id>/artifacts/<sha256>`. Read those MCP resources to
retrieve structures, arrays, and full result JSON; a worker-local path alone is
not the transfer mechanism. Verify `accepted` and the scientific checks.

Calculations are synchronous and bounded (60 seconds by default, including model
startup but excluding preparation). Timeouts/cancellation terminate workers and
preserve interrupted records. Long calculations use prepared bundles and
CLI/job-script execution. The server has no persistent background queue, HTTP
transport, or scheduler integration in this release. MOFforge and ChemGraph
remain independent; MatKit does not re-export MOFforge's tool catalog.

## Validation and promotion

CPU fixtures test contracts and failures, including subprocesses and real stdio
MCP sessions. Synthetic fixtures are not scientific reference calculations.
Run the opt-in [execution recorder](../examples/unified_smoke.py) in the actual
engine environment. It retains per-case bundles, failures, environment details,
and hashes; it does not assert scientific accuracy or promote capabilities.

```bash
python examples/unified_smoke.py --spec pores.json --spec adsorption.json \
  --execution execution.json --outdir evidence/first-run
```

Support promotion requires reviewed reference cases, failure tests, reproducible
installation, and recorded real execution for the specific capability and
environment. The next milestones are reference/parity benchmarks and safe result
reuse, optional MOFforge/charge integration, the porous-material workflow, wider
simulation support, and ChemGraph agent evaluations.
