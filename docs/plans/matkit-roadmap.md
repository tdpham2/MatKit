# MatKit development roadmap

## Intent and boundaries

MatKit has two equal goals: an independently useful scientific toolkit and an
experimental testing ground for ChemGraph. Keep an explicit stable/experimental
boundary, and promote capabilities using evidence.

- MatKit owns scientific preparation, validation, adapters, parsing, and results.
- MatKit experiments establish numerical correctness and performance.
- ChemGraph owns agent planning, memory, tool selection, and distributed jobs.
- Optional MCP adapters expose MatKit functions; workflow skills explain their
  appropriate use. Neither agent framework is a core MatKit dependency.
- Compatible environments can use Python directly; incompatible ML stacks use
  separate worker/service environments. Separate processes alone do not resolve
  conflicting package requirements.

## Prerequisite and next-session entry point

Read [PR #14 hardening](pr14-hardening.md), check its final revision and merge
state, and refresh repository instructions before beginning. Start with
milestone 1. Each milestone is separate work from PR #14 and should use focused
PRs; preserve the user's existing work and public compatibility.

## Ordered milestones

### 1. Reliable foundations

- Propagate custom cutoffs into unit-cell replication in gRASPA, gRASPA SYCL,
  RASPA2, and pygRASPA setup, including cached batch calculations.
- Set RASPA2 parsing success when valid results are obtained.
- Document support per capability (energy, forces, stress, geometry/cell
  optimization), including environment requirements and evidence.
- Graduation requires documented interfaces, reference fixtures, failure tests,
  reproducible installation, and a recorded real execution.

Acceptance: regression cases reproduce and fix the custom-cutoff mismatch and
false RASPA2 failure. Every advertised capability has a support status.

### 2. Shared contracts and provenance

- Introduce a versioned common result envelope with operation-specific payloads.
- Separate scientific model identity, calculator adapter, and execution location.
- Standardize `potential_energy` with compatibility adapters for existing fields;
  preserve legacy defaults through wrappers.
- Record requested/resolved settings, input and model hashes where available,
  versions, precision, units, convergence, and artifact references.

Acceptance: representative results validate and round-trip; legacy APIs remain
usable; MatKit's core remains independent of ChemGraph.

### 3. Reproducibility, resume, and benchmarks

- Add a licensed reference corpus of molecules, crystals, MOFs, varied sizes,
  and intentionally invalid inputs.
- Establish matching-checkpoint numerical parity before measuring performance.
  Use independent reference data for scientific accuracy claims.
- Measure startup, warm execution, throughput, peak memory, and failures
  separately, recording the environment and repeated observations.
- Add explicit resume based on input/configuration/model identity. Reuse only
  compatible completed results; rerun failures and unconverged optimizations.

Acceptance: reproducible benchmark artifacts and tests that reject changed
resume inputs/settings and recover interrupted work without losing results.

### 4. Flagship porous-material workflow

Provide a scripted pipeline: validate structure, optionally desolvate, assign
charges, analyze pores, prepare/run adsorption, and analyze isotherms. Record
artifact lineage, scientific settings, equilibration, and uncertainty methods.

Acceptance: fixture-driven execution in CPU CI and an opt-in real-engine run
with inspectable artifacts at every stage.

### 5. Narrow ChemGraph integration

- Start with pore analysis, adsorption preparation, and MLIP execution.
- Use Python for compatible environments and optional MCP services for isolated
  environments; reuse ChemGraph's execution and job tracking.
- Keep large tensors/trajectories in retrievable artifacts.
- Consolidate duplicated scientific implementations after parity tests pass.

Acceptance: tool discovery, invocation, structured results, artifact retrieval,
and failure/nonconvergence handling tested with deterministic fixtures.

### 6. Tested skills and agent evaluations

Add workflow recipes with use conditions, inputs, scientific choices, expected
outputs, and recovery. Connect them to an explicit ChemGraph skill-loading path.
Turn successful scripted workflows into agent evaluation tasks.

Acceptance: recipe commands execute against tested APIs; evaluations check
numerical results, artifacts, tool selection, and failure handling. Agent
failures become new scientific/integration regression cases.

## CI throughout the roadmap

Test installed wheels and bundled templates, then selected optional-dependency
environments. Keep ordinary CI fast, deterministic, and CPU-only. Gate model
downloads, external engines, live endpoints, and GPUs explicitly.

## Status

Roadmap implementation has not started. PR #14 hardening is complete at
implementation commit `28e764f`; its handoff records 259 passing CPU tests and
one skip. Check the PR's current head and merge state before beginning milestone
1. No GPU capability has been promoted on the basis of this implementation.
