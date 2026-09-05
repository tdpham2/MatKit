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

- [x] Validate finite configuration, structures, energy, forces, and stress;
  distinguish unavailable stress from calculator failure; use strict JSON.
- [x] Reject explicitly unsupported CLI/example options; implement strict
  exits and expose unconverged counts.
- [x] Persist ordered per-item results and manifests atomically during execution;
  retain completed work after interruptions, teardown, and persistence errors.
- [x] Make mocked GPU tests hermetic; cover adapter boundaries; extend opt-in
  Polaris energy/optimization/batch checks and record their environment.
- [x] Run the complete CPU suite, lint/format checks, package build, example
  help checks, and shell syntax checks.

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

Implementation is complete at `28e764f` (four focused implementation commits;
this handoff is a subsequent documentation-only change). PR #14's commit list
is the source of truth for the final published branch head and GitHub checks.

| Commit | Change |
| --- | --- |
| `4346208` | Scientific input/result validation and strict JSON |
| `065d3cc` | Explicit option validation and strict CLI/example exits |
| `c646505` | Incremental atomic persistence and Rootstock worker isolation |
| `28e764f` | Adapter boundary tests, GPU recipe, CI checks, and documentation |

Local verification on Python 3.12:

- `PYTHONPATH=src pytest tests/ -q`: **259 passed, 1 skipped**.
- Ruff lint and format checks pass for `src/`, MLIP tests, CLI tests,
  `examples/mlip_gpu.py`, and `alcf/polaris/mlip/smoke.py`.
- `python -m build --no-isolation --outdir /private/tmp/matkit-pr14-dist`:
  source distribution and wheel built successfully.
- Both GPU Python entry points pass `--help`; both Polaris shell scripts
  pass `bash -n`; `git diff --check` passes.

No real MACE model, Rootstock deployment, ALCHEMI GPU kernel, or cluster job was
run during this implementation. Those capabilities remain experimental. The
opt-in smoke runner records six cases, environment/commit information, logs,
results, and convergence; it continues after individual case failures.

Known limits: no automatic resume/retry, no parity/performance evidence, no
common provenance migration, and no GCMC fixes in this PR. ALCHEMI native
optimization step counts remain unknown (`n_steps=null`); per-item calculation
times for native batches include shared chunk work and are not throughput
measurements. Inputs/results remain accumulated in memory. Abrupt termination
can leave a `running` manifest and a batch lock; use a fresh directory.

After PR #14 is reviewed, begin roadmap milestone 1: cutoff propagation,
RASPA2 success reporting, and explicit capability support status. Do not begin
resume, new backends, benchmarking, or ChemGraph changes as part of this PR.
