# PR #14 hardening and session handoff

PR: https://github.com/tdpham2/MatKit/pull/14

Starting revision: `7a58171ee92513047517c0fee63eaa4f4e4f8573`.
Branch: `mlip-playground-backends`. This plan covers the new MLIP interface;
the independent growth plan is [matkit-roadmap.md](matkit-roadmap.md).

## Agreed decisions

- Persist completed work immediately; automatic resume is deferred.
- CLI exit 0 requires valid calculations and converged requested optimizations.
  Calculation failure/nonconvergence exits 1; invalid arguments exit 2.
- `success` records numerically valid completion; `converged` separately records
  optimization convergence. Keep usable unconverged results.
- Keep existing public entry points, configuration classes, `energy`, units,
  and fixed-cell optimization scope. Full contract migration belongs later.
- Backends/capabilities without real execution evidence remain experimental.
  CPU tests and prepared GPU recipes do not count as GPU validation.
- Update the existing PR; merging is a separate action.

## Implementation checklist

- [ ] Validate finite configuration, structures, energy, forces, and stress;
  distinguish unavailable stress from calculator failure; use strict JSON.
- [ ] Reject explicitly unsupported CLI/example options; implement strict
  exits and expose unconverged counts.
- [ ] Persist ordered per-item results and manifests atomically during execution;
  retain completed work after interruptions, teardown, and persistence errors.
- [ ] Make mocked GPU tests hermetic; cover adapter boundaries; extend opt-in
  Polaris energy/optimization/batch checks and record their environment.
- [ ] Run the complete CPU suite, lint/format checks, package build, example
  help checks, and shell syntax checks; update the PR description.

## Persistence contract

An initial manifest has ordered pending items and status `running`. A completed
item's result is committed before its manifest entry. Final execution statuses
remain `completed`, `partial`, and `failure`. Catchable orchestration failures
mark the manifest `interrupted` and propagate; an uncatchable termination can
leave `running`. Completed JSON files remain readable in either case.

Manifests expose pending and unconverged counts, per-item convergence, and a
run-level error. Success counts include valid but unconverged results. Reruns
must use a fresh batch directory; existing manifests are not overwritten.

## Acceptance scenarios

- Reject NaN/Inf inputs/outputs and malformed tensors before reporting success.
- Preserve valid converged and unconverged optimization results.
- Forward applicable options and reject explicit unsupported options.
- Preserve ordering, duplicate basenames, calculator reuse, and partial failures.
- Preserve completed records after interruption, write failure, and teardown.
- Mocked tests require neither CUDA nor downloads; real GPU checks are opt-in.

## Verification and next session

Implementation and verification are in progress. No live GPU/cluster execution
is claimed. Final results and implementation revision will be recorded here.

After PR #14 is reviewed, begin roadmap milestone 1: cutoff propagation,
RASPA2 success reporting, and explicit capability support status. Do not begin
resume, new backends, benchmarking, or ChemGraph changes as part of this PR.
