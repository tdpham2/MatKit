# MLIP playground on Polaris

This directory installs and smoke-tests the three agent-free MLIP paths in
MatKit:

- direct MACE through ASE;
- a cluster-managed MACE checkpoint through Rootstock;
- native batched MACE through NVIDIA ALCHEMI Toolkit.

The adapters and these recipes are **experimental**. Preparing or testing the
scripts with doubles does not verify a live installation. The scripts collect
integration evidence when explicitly run on Polaris; they are not performance
benchmarks and do not establish model parity or scientific accuracy.

## Prerequisites

Rootstock is deployed on Polaris, but ALCF users need access to its shared
installation. Follow the current Polaris instructions in the
[Matter Model Almanac](https://garden-ai.github.io/almanac/clusters/) before
running the Rootstock smoke test.

Run the installer from the MatKit checkout on a Polaris login node:

```bash
export MATKIT_MLIP_ENV=/lus/eagle/projects/PROJECT/USER/envs/matkit-mlip
bash alcf/polaris/mlip/install.sh
```

The installer creates an isolated Python 3.12 environment, installs the CUDA
12 and MACE extras for ALCHEMI, installs Rootstock, and installs this checkout
in editable mode. Override `MATKIT_MLIP_ENV`; the default is `.venv` in the
repository.

This is a MatKit-specific environment. ALCHEMI 0.2 supports Python 3.11–3.13;
its MACE extra pins `mace-torch==0.3.15`. ChemGraph currently requires MACE
`>=0.3.16`, so installing its full dependency stack here is incompatible.
Different environments are required to support those versions simultaneously.
Rootstock uses a separate deployment-managed model environment. Its caller
does not require a local CUDA-enabled PyTorch installation.

## Smoke test

Edit the `#PBS -A` project in `smoke.pbs`, then submit it while the checkout is
your working directory:

```bash
qsub -v MATKIT_MLIP_ENV="$MATKIT_MLIP_ENV" \
  alcf/polaris/mlip/smoke.pbs
```

The defaults use the small periodic structure in `tests/data`, direct and
ALCHEMI checkpoint alias `medium`, and Rootstock checkpoint
`mace-mp-0-medium`. Override paths or checkpoint names when submitting:

```bash
qsub -v MATKIT_MLIP_ENV="$MATKIT_MLIP_ENV",\
MATKIT_SMOKE_INPUT=/path/to/input.cif,\
MACE_CHECKPOINT=/path/to/model.pt,\
ROOTSTOCK_CHECKPOINT=mace-mp-0-medium \
  alcf/polaris/mlip/smoke.pbs
```

Results are written under `projects/mlip_smoke_$PBS_JOBID` by default. Set
`MATKIT_SMOKE_OUTPUT` to choose a **new** persistent directory. Reusing an
existing output directory is refused.

The PBS script invokes `smoke.py`, which runs six separate Python processes:
energy and fixed-cell optimization for each backend. ALCHEMI evaluates a
two-structure native batch for both drivers. Inputs are the original structure
and a reproducibly perturbed copy. Override `MATKIT_SMOKE_STEPS` (default 1000)
or `MATKIT_SMOKE_FMAX` (default 0.01 eV/angstrom) when submitting.

Every case retains stdout, stderr, calculation results, and its exit code.
`smoke_report.json` records the caller's package versions, GPU information,
Rootstock deployment resolution, input hash, settings, commands, observed
convergence, and validation outcome. It is updated after each case; one failure
does not prevent other cases from running. A case passes only if its CLI exits
zero, results are finite and correctly shaped, the cell is unchanged within
serialization precision, and requested optimizations converge. The overall
runner exits nonzero if any case fails.

Keep the full output directory with any capability-validation record, including
the MatKit commit used. Promote only the backend/capability/environment actually
verified by that record. Caller package versions do not identify all packages
inside a Rootstock worker; retain the deployment information and obtain worker
versions when assessing reproducibility or parity.

For compute-node downloads, the PBS script exports the ALCF HTTP proxy. Model
weights should be allowed to finish downloading before treating later timings
as performance measurements.
