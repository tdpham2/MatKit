# Code review follow-ups

Review date: 2026-09-05. Baseline revision: `5cd7a03` on
`docs/calculation-examples-review`.

This file records findings 2-7 from the repository review. They are deferred:
none of the issues below are fixed by the accompanying Zeo++ stale-output
change. Each item should receive a focused implementation and regression test
before it is marked complete.

## 2. gRASPA CLI uses the wrong default output columns

**Severity:** High. **Status:** Deferred.

The gRASPA setup templates and unified `AdsorptionRequest` default to PR-EOS,
but `matkit graspa analyze` calls `get_output_data()` without `eos=True`. The
legacy parser therefore reads the non-EOS column positions by default. In the
review fixture, the same result is interpreted as uptake 12 by the unified API
and uptake 7 by the CLI.

**Impact:** A successful CLI command can report a valid-looking but incorrect
scientific result.

**Recommended remediation:** Persist the chosen fugacity treatment with each
prepared simulation and make analysis consume it. Add an explicit CLI override
for manually produced output, rather than relying on a silent Boolean default.
Test default PR-EOS setup/analyze parity, an explicit non-EOS result, and
missing or contradictory metadata.

## 3. Legacy CLI failures return exit status zero

**Severity:** High. **Status:** Deferred.

Many legacy commands in `src/matkit/cli.py` catch broad exceptions, print an
error to stderr, and return normally. Click consequently exits with status 0.
This affects gRASPA, pygRASPA, RASPA2, plotting, legacy MLIP, PACMOF2, and
Zeo++ command paths.

**Impact:** Shell scripts, schedulers, and CI can treat failed work as
successful and continue with missing or invalid artifacts.

**Recommended remediation:** Convert operational failures to `ClickException`
or an equivalent nonzero exit while retaining Click's exit 2 for invalid
arguments. Add parameterized CLI tests asserting exit 1 and stderr for each
command family, plus exit 0 for successful invocations.

## 4. UMA multiprocessing can deadlock or lose results

**Severity:** High. **Status:** Deferred.

`run_opt_uma_batch()` joins every worker before draining `result_queue`, then
uses `Queue.empty()` to decide whether results remain. A worker can block while
flushing a full queue, preventing `join()` from completing, and
`multiprocessing.Queue.empty()` is not reliable for synchronization.

**Impact:** Large batches can hang indefinitely or produce an incomplete
`results.jsonl` while appearing to have processed all jobs.

**Recommended remediation:** Collect exactly one terminal result per submitted
job while workers are active, with explicit worker-exit and timeout handling;
then join and close queue resources. Synthesize failure records for jobs whose
workers terminate without reporting. Test queue backpressure, abrupt worker
exit, delayed delivery, and complete accounting for every input/model/run-type
combination.

## 5. Batch interruption overwrites preparation failures

**Severity:** Medium. **Status:** Deferred.

`_interrupt_batch()` marks every run with `run.json` but no `result.json` as a
new batch interruption. That includes items which already reached a terminal
preparation failure, so their original stage, exception, and diagnostic are
replaced by the later batch-level error.

**Impact:** Manifests lose the root cause of individual failures and make
recovery decisions less reliable.

**Recommended remediation:** Treat terminal item failures as immutable.
Interruption should update only pending or actively executing items, while the
batch stores its own interruption failure separately. Test a preparation
failure followed by interruption and verify both item and batch diagnostics
survive inspection and restart.

## 6. Sanitized CIF stems can collide

**Severity:** Medium. **Status:** Deferred.

gRASPA and pygRASPA replace dots with underscores when constructing batch
directories and copied CIF names. Distinct inputs such as `a.b.cif` and
`a_b.cif` therefore map to the same output tree and overwrite one another.

**Impact:** Batch manifests can point multiple structures at one simulation,
silently attributing inputs or results to the wrong framework.

**Recommended remediation:** Assign each input a collision-resistant identity,
such as a readable sanitized stem plus a stable short hash of the original
name/path, and reject any remaining duplicate destination before writing.
Preserve the original name in the manifest. Test colliding names in both
gRASPA and pygRASPA single-condition and sweep setup.

## 7. RASPA3 conversion accepts malformed interaction sections

**Severity:** Medium. **Status:** Deferred.

`parse_raspa2_force_field()` slices at most the declared interaction count but
silently skips records with fewer than four fields. It does not verify that the
file contains the declared number of valid interactions before processing the
remaining section.

**Impact:** Conversion can emit an incomplete RASPA3 force field while
reporting success, shifting later lines into the wrong logical section or
omitting atom interactions.

**Recommended remediation:** Require exactly the declared number of interaction
lines and validate every record's name, type, and numeric parameters with a
line-specific error. Test truncated sections, malformed records, invalid
numbers, zero interactions, and a valid file whose parsed count exactly matches
the declaration.
