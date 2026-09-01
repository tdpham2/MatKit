# MLIP playground on Polaris

This directory installs and smoke-tests the three agent-free MLIP paths in
MatKit:

- direct MACE through ASE;
- a cluster-managed MACE checkpoint through Rootstock;
- native batched MACE through NVIDIA ALCHEMI Toolkit.

These scripts validate that the installations work. They are not performance
benchmarks and do not collect repeated timing or parity statistics.

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
`MATKIT_SMOKE_OUTPUT` to choose another persistent directory.

For compute-node downloads, the PBS script exports the ALCF HTTP proxy. Model
weights should be allowed to finish downloading before treating later timings
as performance measurements.
