# MatKit development roadmap

## Intent and boundaries

MatKit has two equal goals: an independently useful scientific toolkit and an
experimental testing ground for ChemGraph. Keep an explicit stable/experimental
boundary, and promote capabilities using evidence.

MOFforge prepares structures, MatKit evaluates them, and ChemGraph coordinates
workflows. Build on [MOFforge's existing Python and MCP interfaces](https://github.com/tdpham2/mofforge/blob/e49c7ae5c0ee50eb5136234e05013406d96394a7/docs/chemgraph.md).

| Project | Responsibility |
| --- | --- |
| MOFforge | Structure discovery, construction, linker generation, functionalization, defects, desolvation, geometric adsorbate placement, structural checks, and visualization |
| MatKit | Calculators, energy and force evaluation, geometry and cell relaxation, charge prediction, pore calculations, adsorption simulations, scientific results, and benchmarks |
| ChemGraph | Agent planning, tool selection, memory, execution backends, and job tracking across both toolkits |

MOFforge may retain embedding and geometry cleanup used internally during
construction. User-requested scientific relaxation belongs in MatKit. Geometric
adsorbate placement supplies an initial configuration; MatKit determines
adsorption energetics and equilibrium uptake. MOFforge may search existing
database properties, while MatKit computes new properties. Simulation-specific
validation, engine input formatting, and cutoff-dependent cell sizing remain
in MatKit.

- MatKit experiments establish numerical correctness and performance.
- MOFforge integration is optional. MatKit's core remains independent of
  MOFforge and ChemGraph; MOFforge's structure operations do not depend on MatKit.
- Optional MCP adapters expose each project's own functions; workflow skills
  explain their appropriate use. ChemGraph can consume both tool catalogs
  directly without MatKit re-exposing MOFforge's catalog.
- Compatible environments can use Python directly; incompatible ML stacks use
  separate worker/service environments. Separate processes alone do not resolve
  conflicting package requirements.

## Current baseline — 2026-09-05

[PR #15](https://github.com/tdpham2/MatKit/pull/15) merged at `f8ebd1c`, delivering
the first experimental unified execution layer on top of
[PR #14 hardening](pr14-hardening.md). The implementation commits are `aa8d5c8`
(simulation setup/parser fixes) and `e84b29e` (API, execution, CLI, MCP,
documentation, and tests).

| Milestone | Current state | Remaining work |
| --- | --- | --- |
| 1. Reliable foundations | Initial hardening in PR #15; engine-boundary follow-up implemented in the correctness follow-up | Review the follow-up and promote capabilities only after real execution and reference validation |
| 2. Shared contracts and provenance | First-release contracts implemented; acceptance and post-commit interruption follow-up implemented in the correctness follow-up | Review the follow-up; extend contracts for charges and validate cross-project handoffs |
| 3. Reproducibility, resume, and benchmarks | Input hashes, durable results, interruption records, and execution recorder implemented | Reference corpus, real-engine evidence, parity/performance benchmarks, and identity-based resume/reuse |
| 4. MOFforge integration | Planned | Optional dependency, conversions, compatibility, and workflow gates |
| 5. Porous-material workflow | Pore and pure-component adsorption primitives implemented | Charge-result handoff, orchestration, isotherms, and end-to-end validation |
| 6. ChemGraph integration | MatKit tool discovery, calls, and artifact resources implemented/tested locally | Integrate both tool catalogs with ChemGraph, artifact staging, and job tracking |
| 7. Skills and agent evaluations | Planned | Tested workflow recipes, skill loading, and agent evaluation cases |

The first-release scientific scope is direct MACE, Rootstock, and native
ALCHEMI MACE evaluation/fixed-cell relaxation; Zeo++ pore analysis; and
single-component gRASPA CUDA preparation, execution, and parsing. Existing
engine-specific APIs remain available. See the
[unified API guide](../unified-api.md) and
[capability inventory](../capabilities.md) for supported inputs and restrictions.

Local Python 3.12 validation recorded **336 passed, 1 skipped** both from source
and from an installed wheel outside the checkout, including real local MCP
stdio sessions backed by synthetic engines. Build, lint, and formatting checks
passed. These results validate implementation behavior; real model/engine/GPU
execution and scientific reference evidence remain pending. No scientific
adapter has been promoted by these tests.

## Correctness follow-up

The review of PR #15 found engine-boundary and result-acceptance gaps despite
passing CPU tests. The following fixes are implemented in separate correctness branches; they
are not part of the `f8ebd1c` baseline until their PRs are merged:

- [PR #16](https://github.com/tdpham2/MatKit/pull/16): share Zeo++ argument
  construction and output validation across the legacy and unified APIs. Correct unequal-radius ordering, support multiline channel
  output and `.psd_histo`, and reject incomplete or nonfinite requested results.
- [PR #17](https://github.com/tdpham2/MatKit/pull/17): preserve committed
  numerical results while recording later worker timeouts,
  cancellation, teardown failures, and unexpected exit codes. Require requested
  evaluation properties and consistent relaxation convergence when validating
  results; a missing required convergence check cannot qualify for acceptance.
- Normalize bar/Pa pressures in single isotherms. Reject mixed temperatures,
  incompatible uptake/heat units, and duplicate physical pressure points.

Combined Python 3.12 validation of the three changes: **393 passed, 1 skipped**
from an installed wheel outside the checkout. Source lint/format checks and the
sdist/wheel build passed. Focused suites cover Zeo++ (50 passed), worker/API/CLI/MCP
outcomes (70 passed), and isotherm parsing/plotting (84 passed, 1 skipped).

Regression tests exercise documented output shapes, imported result records,
and real supervised processes backed by synthetic calculations. They are not
real-engine evidence. gRASPA's positional parser and PACMOF2's per-output
validation remain priorities for the evidence and charge-handoff work below.

## Next-session entry point

Start from current `main`, read the unified API guide and capability inventory,
and refresh repository instructions. Review and merge the correctness follow-up
before starting the evidence release. Begin milestone 3a with direct MACE and
Zeo++, then authentic gRASPA inputs/output. Build the charge handoff and scripted
workflow next; do not wait for all GPU adapters, full MOFforge integration, or
performance benchmarking. Add identity-based recovery and scaling work after
those gates pass. Preserve existing public behavior and use focused PRs.

Use Python or CLI directly for ordinary simulations. MCP remains an optional
interface to the same backend, with bounded calls and preparation for longer
calculations. Prepared bundles and the
[Polaris PBS example](../../alcf/polaris/unified/run.pbs) support execution inside
existing allocations. Scheduler submission and persistent background jobs are
outside this first release.

## Ordered milestones

### 1. Reliable foundations

Status: initial hardening delivered in PR #15. Additional engine-boundary
corrections are implemented in the follow-up above and await merge. Scientific
support promotion remains conditional on the evidence below.

- Propagate custom cutoffs into unit-cell replication in gRASPA, gRASPA SYCL,
  RASPA2, and pygRASPA setup, including cached batch calculations.
- Reject invalid cutoffs and nonfinite or degenerate cells; set RASPA2 parsing
  success when valid results are obtained and reject invalid numerical output.
- Document support per capability (energy, forces, stress, geometry/cell
  optimization), including environment requirements and evidence.
- Graduation requires documented interfaces, reference fixtures, failure tests,
  reproducible installation, and a recorded real execution.

Acceptance: regression cases reproduce and fix the custom-cutoff mismatch and
false RASPA2 failure. Every advertised capability has a support status.

### 2. Shared contracts and provenance

Status: implemented for the first-release operations, with stricter result
acceptance and post-commit interruption handling in the follow-up above. This is
the common execution contract; DFT, MD, cell optimization, charge prediction, and mixture
results have not been migrated into it.

- Validate versioned requests and reject incompatible model/adapter options.
- Introduce a versioned common result envelope with operation-specific payloads.
- Separate scientific model identity, calculator adapter, and execution location.
- Separate execution state, numerical validity, and required scientific checks.
  An unconverged relaxation retains its numerical result but is not accepted;
  adsorption uncertainty does not establish equilibration or sampling quality.
- Standardize `potential_energy` with compatibility adapters for existing fields;
  preserve legacy defaults through wrappers.
- Record requested/resolved settings, input and model hashes where available,
  versions, precision, units, convergence, and artifact references.
- Define a structure handoff with artifact references, content hashes, parent
  lineage, operation settings, and software versions. Preserve species,
  coordinates, cell, periodicity, and atom correspondence; carry relevant
  labels, bonds, and charge metadata alongside the structure when its file
  format cannot preserve them.
- Track when structural edits invalidate derived charges, energies, and pore
  properties. Recompute affected results before downstream calculations use
  them; do not attach stale results to a modified structure.
- Stage hashed inputs in relocatable bundles and expose artifact contents
  through MCP resources. A server-local path alone is insufficient for remote
  consumers; cross-project client staging remains part of milestone 6.
- Provide one preparation/execution/inspection lifecycle through Python and
  CLI, with optional bounded stdio MCP calls. Execution profiles select the
  interpreter, device, environment, and engine argument lists.
- Reuse calculators in homogeneous batches, retain native ALCHEMI chunking,
  and persist ordered per-item results and failures. Commit results atomically
  before manifest updates and preserve completed numerical payloads when later
  orchestration fails.

Acceptance: representative results validate and round-trip; legacy APIs remain
usable; MatKit's core remains independent of MOFforge and ChemGraph. Structure
handoffs preserve metadata and atom correspondence, report unsupported inputs
explicitly, and prevent reuse of invalidated scientific results.

### 3. Reproducibility, resume, and benchmarks

Status: next implementation milestone. PR #15 supplies portable input hashes,
provenance, incremental batch records, and an
[opt-in execution recorder](../../examples/unified_smoke.py). Recovery of a
committed result is implemented; automatic resume, caching, engine restart,
reference benchmarks, and scientific accuracy claims are not.

Deliver this in three focused steps:

#### 3a. Reference corpus and real-engine evidence

- Add a licensed reference corpus of molecules, crystals, MOFs, varied sizes,
  and intentionally invalid inputs.
- Start with one pinned direct-MACE checkpoint and one pinned Zeo++ build.
  Record the inputs, installation recipe, resolved settings, logs, results,
  expected failures, and comparison criteria for each capability.
- Add authentic charged gRASPA inputs and complete, version-identified engine
  logs. Keep execution compatibility, matching-model numerical parity, and
  accuracy against independent reference data as separate evidence categories.
- Record Rootstock and native ALCHEMI evidence in their own environments when
  available; these adapters do not block useful CPU validation or the first
  scripted workflow. Retain input/model identities and failed runs.
- Check charged CIFs, force-field definitions, units, and uptake/heat parsing
  against reference gRASPA output. Treat sampling adequacy as a separate
  scientific gate; do not infer it from a zero exit code or a finite result.
- Update capability evidence per operation and environment after review.

#### 3b. Numerical parity and performance

- Establish matching-checkpoint numerical parity before measuring performance.
  Use independent reference data for scientific accuracy claims.
- Compare requested energy/forces/stress, relaxation convergence, and sequential
  versus native-batch behavior with documented tolerances and failure cases.
- Measure preparation, input staging, persistence, startup, warm execution,
  throughput, peak memory, and failures separately, recording the environment
  and repeated observations.
- Profile increasing batch sizes before optimizing GPU execution. The current
  implementation copies local checkpoints per item, loads all structures, and
  rereads every item after completions. Use measurements to prioritize bounded
  structure loading, incremental record updates, and deduplicated supporting
  assets while preserving relocatable bundles and interruption recovery.

#### 3c. Identity-based resume and result reuse

- Start explicit resume/reuse with accepted completed MLIP results whose model
  checkpoint content is resolved. Match input/configuration/model identity;
  preserve prior attempts and rerun failures and unconverged optimizations.
  Extend reuse to other scientific operations only after their identity and
  acceptance policies have reference evidence.
- Include structure/metadata, scientific settings, model content, supporting
  inputs, adapter/schema versions, and relevant numerical settings in the reuse
  policy. An unresolved model alias must not silently qualify as an identical
  checkpoint.
- Define stale-lock recovery and batch continuation, preserving already
  committed results. Treat restarting an engine from its checkpoint as a
  separate capability from skipping an accepted completed calculation.

Acceptance: reviewed real-execution and benchmark artifacts; documented parity
tolerances; tests that reject changed resume identities, rerun failed or
unconverged items, and continue interrupted batches without losing results.

### 4. MOFforge structure integration

Status: planned. Neither the integration nor `matkit[mofforge]` exists yet.
This integration does not block the charge handoff or the first scripted MatKit
workflow using an already prepared structure.

- Add an optional `matkit[mofforge]` integration using MOFforge's public Python
  APIs. Keep MatKit's ordinary installation usable independently.
- Start with structural validation and desolvation, then consume construction
  and functionalization outputs. Bridge MOFforge's pymatgen-based structures to
  MatKit's ASE inputs through tested conversions and the milestone 2 handoff.
- Preserve existing MatKit solvent-removal and linker-generation APIs and
  defaults until compatibility tests support migration. MatKit currently uses
  component mass for solvent removal; MOFforge uses component atom counts.
  Do not silently substitute algorithms or defaults.
- Direct new structure-manipulation features to MOFforge. Keep the integration
  experimental until compatibility and workflow checks pass and a real
  execution is recorded.

Acceptance: CPU fixtures cover periodic cells, atom mapping, metadata
preservation, unsupported structures, and desolvation of guests, interpenetrated
frameworks, and retained ions. Legacy behavior remains reproducible; importing
and using MatKit without MOFforge still works. Workflow gates distinguish tool
execution success from structure validity and calculation convergence.

Before adopting HTTP, verify MOFforge's selected-tool startup and track any
correction in MOFforge. At reviewed revision `e49c7ae`, its HTTP entry point
invokes the global server before the selected server. Test that only the
selected catalog is served with the requested host and port before enabling
this transport. See the [reviewed startup implementation](https://github.com/tdpham2/mofforge/blob/e49c7ae5c0ee50eb5136234e05013406d96394a7/src/mofforge/mcp/server.py).

### 5. Flagship porous-material workflow

Status: the first release supplies relaxation, pore analysis, and pure-component
adsorption primitives. Charge prediction remains a legacy API; the shared
charge-result contract and complete scripted workflow are still to be built.

Start with an already prepared structure. MatKit optionally relaxes it, assigns
charges, analyzes pores, prepares/runs single-component adsorption, and exports
an isotherm. Add MOFforge preparation/checking and optional desolvation after its
conversion and compatibility gates pass. Record artifact lineage, scientific
settings, equilibration, and uncertainty methods. Apply structural validation
and derived-result invalidation at the relevant handoffs.

Deliver the charge handoff after the narrow reference-evidence release; it need
not wait for all of milestone 3b/3c or the full milestone 4 integration.

- First add the charge-result handoff: retain the prediction method/version,
  geometry identity, atom mapping, charge units, and total-charge validation.
  Verify each expected PACMOF2 output exists, maps to its input atoms, contains
  finite charges, and satisfies the requested total charge. A successful
  predictor return alone does not establish per-output success. Require
  recomputation after structural changes before adsorption uses it.
- Compose the tested operations into the pipeline, preserving each stage's
  bundle and scientific checks. Aggregate single-component pressure points into
  an isotherm with explicit sampling and uncertainty information.

Acceptance: fixture-driven execution in CPU CI and an opt-in real-engine run
with inspectable artifacts at every stage.

### 6. Narrow ChemGraph integration

Status: MatKit's optional MCP server, selected catalog, bounded worker calls,
artifact resources, and deterministic local integration tests are implemented.
ChemGraph integration and interoperability with MOFforge's catalog remain
planned; local MCP success is not evidence of that integration.

- Start with pore analysis, adsorption preparation, and MLIP execution.
- Expose MatKit and MOFforge directly through their own tools, using the shared
  structure handoff. Keep ownership clear in tool descriptions.
- Use Python for compatible environments and optional MCP services for isolated
  environments; reuse ChemGraph's execution and job tracking.
- Keep large tensors/trajectories in retrievable artifacts.
- Test retrieval and staging between distinct tool roots/environments, including
  hash verification, metadata sidecars, and derived-result invalidation.
- Consolidate duplicated scientific implementations after parity tests pass.

Acceptance: discovery and invocation across both tool catalogs, structured
results, artifact retrieval/staging, and failure/nonconvergence handling tested
with deterministic fixtures.

### 7. Tested skills and agent evaluations

Status: planned. The execution recorder and PBS example are backend recipes;
ChemGraph skill loading and agent evaluations are not implemented.

Add workflow recipes with use conditions, inputs, scientific choices, expected
outputs, and recovery. Teach project ownership, validation gates, artifact
transfer, and invalidation of derived results. Connect recipes to an explicit
ChemGraph skill-loading path. Turn successful scripted workflows into agent
evaluation tasks.

Acceptance: recipe commands execute against tested APIs; evaluations check
numerical results, artifacts, tool selection, and failure handling. Agent
failures become new scientific/integration regression cases.

## Later scientific extensions

Expand the unified API after the initial adapters have reference evidence.
Prioritize new capabilities from validated workflow needs: charge prediction
in milestone 5, then selected DFT, MD, cell optimization, or mixture adsorption
work. Keep operation-specific requests and results; add explicit units,
convergence/sampling checks, environment requirements, fixtures, and real
execution evidence for each extension. Existing legacy interfaces remain
available during migration.

## CI throughout the roadmap

Implemented in PR #15: wheel-based tests outside the source checkout on the
Python 3.10–3.12 CI matrix, bundled-template checks, a separate optional MCP
stdio/cancellation job, and lint/format checks for the new code and tests.
Deterministic fixtures cover contracts, structure metadata, simulation setup,
CLI parity, subprocess failures, interruption recovery, and artifact retrieval.

Continue adding selected optional-dependency and cross-project environments as
milestones require them. Keep ordinary CI fast, deterministic, and CPU-only.
Gate model downloads, real external engines, remote endpoints, and GPUs
explicitly; retain reproducible evidence from opt-in runs. Local synthetic MCP
sessions remain part of ordinary integration testing.
