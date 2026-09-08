# MLIP throughput benchmarks on Polaris

These scripts measure **performance** of MatKit's MLIP backends — how many
structures per second a backend sustains, how timing scales with structure size,
and how NVIDIA ALCHEMI native batching compares to sequential ASE MACE.

This is distinct from `../smoke.py`, which is a correctness/integration test and
explicitly **not** a performance benchmark. Run the smoke test first to confirm
the backend works on the node, then run these to characterize throughput.

All numbers are hardware-, checkpoint-, and version-specific. Every run writes a
`bench_meta.json` capturing the GPU model, package versions, and arguments —
keep it with any results you report.

## Prerequisites

Install the MLIP environment and pass the smoke test as described in
[`../README.md`](../README.md):

```bash
export MATKIT_MLIP_ENV=/lus/eagle/projects/<PROJECT>/<USER>/envs/matkit-mlip
bash alcf/polaris/mlip/install.sh
qsub -v MATKIT_MLIP_ENV="$MATKIT_MLIP_ENV" alcf/polaris/mlip/smoke.pbs
```

Edit the `#PBS -A PROJECT` line in the `.pbs` scripts before submitting.

## Single-GPU sweeps

```bash
qsub -v MATKIT_MLIP_ENV="$MATKIT_MLIP_ENV" \
  alcf/polaris/mlip/benchmark/bench.pbs
```

Runs two sweeps for both `nvalchemi-mace` and `ase-mace`:

- **batch_size** — a fixed set of `--n-structures` copies, ALCHEMI batch size
  swept over `--batch-sizes`. ASE MACE is sequential (batch size 1) and serves as
  the baseline. Throughput should rise with batch size, then plateau when the GPU
  saturates. Peak GPU memory is sampled from `nvidia-smi`.
- **structure_size** — one structure per supercell factor (`--size-factors`),
  measuring per-structure time vs atom count.

Override defaults at submit time:

```bash
qsub -v MATKIT_MLIP_ENV="$MATKIT_MLIP_ENV",\
MATKIT_BENCH_INPUT=/path/to/structure.cif,\
MACE_CHECKPOINT=medium,\
MATKIT_BENCH_DRIVER=opt,\
MATKIT_BENCH_NSTRUCT=128 \
  alcf/polaris/mlip/benchmark/bench.pbs
```

Run `bench.py --help` for the full flag list (checkpoint, dtype, driver, steps,
batch sizes, size factors).

## Multi-GPU node throughput

Polaris nodes have 4x A100. The runner uses one GPU per process, so node-level
throughput is measured by launching one process per GPU over a round-robin shard
of the inputs:

```bash
qsub -v MATKIT_MLIP_ENV="$MATKIT_MLIP_ENV" \
  alcf/polaris/mlip/benchmark/bench_multigpu.pbs
```

`node_result.json` reports aggregate `node_throughput_structs_per_s`. Compare it
to the single-GPU throughput at the same batch size — ideal scaling is ~4x.

## Reading results

`bench.py` writes `bench_results.jsonl`, one JSON object per configuration:

| field | meaning |
|-------|---------|
| `sweep` | `batch_size` or `structure_size` |
| `backend` | `nvalchemi-mace` or `ase-mace` |
| `batch_size`, `n_structures`, `n_atoms` | configuration |
| `setup_time_s` | model load time (from the manifest) |
| `wall_time_s` | batch execution wall time (from the manifest) |
| `elapsed_s` | end-to-end `run_mlip_batch` time |
| `throughput_structs_per_s` | `succeeded / elapsed_s` |
| `calc_time_s_mean` / `_max` | per-item calculation time |
| `peak_gpu_mem_mib` | max used GPU memory sampled during the run |

Quick summary with `jq`:

```bash
jq -r 'select(.sweep=="batch_size") |
  [.backend, .batch_size, .throughput_structs_per_s, .peak_gpu_mem_mib] | @tsv' \
  bench_results.jsonl
```

## Caveats

- Let model weights finish downloading before trusting timings. The PBS scripts
  do a throwaway warm-up run first; the ALCF HTTP proxy is exported for
  compute-node downloads.
- `setup_time_s` (model load) is reported separately from per-item time; when
  comparing backends, look at steady-state throughput, not the first item.
- These measure MatKit's execution path, not raw kernel performance, and do not
  establish model parity or scientific accuracy.
