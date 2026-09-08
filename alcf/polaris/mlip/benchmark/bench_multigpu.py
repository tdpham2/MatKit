#!/usr/bin/env python3
"""Node-level throughput: one MLIP process per GPU over a sharded input set.

The MatKit runner uses a single GPU per process (``_execute_inputs`` loads one
model on ``config.device``). A Polaris node has 4x A100, so node throughput is
measured by launching one ``run_mlip_batch`` process per GPU, each pinned with
``CUDA_VISIBLE_DEVICES`` to a disjoint shard of the inputs, then aggregating the
per-process manifests.

Each worker runs ``examples`` -style batching via a small inline driver so no
runner changes are needed. Structures are round-robin sharded across GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ase.io import read, write


_WORKER = r"""
import json, sys
from pathlib import Path
from matkit.mlip import (
    ASEMACEConfig, NVAlchemiMACEConfig, MLIPCalculationConfig, run_mlip_batch,
)
cfg = json.loads(sys.argv[1])
inputs = cfg["inputs"]
if cfg["backend"] == "nvalchemi-mace":
    backend = NVAlchemiMACEConfig(
        checkpoint=cfg["checkpoint"], device="cuda", dtype=cfg["dtype"])
else:
    backend = ASEMACEConfig(
        checkpoint=cfg["checkpoint"], device="cuda", dtype=cfg["dtype"])
calc = MLIPCalculationConfig(driver=cfg["driver"], steps=cfg["steps"])
summary = run_mlip_batch(
    inputs, backend, calculation=calc,
    output_dir=cfg["output_dir"], batch_size=cfg["batch_size"])
print(json.dumps({
    "succeeded": summary["succeeded"],
    "failed": summary["failed"],
    "wall_time_s": summary["wall_time_s"],
    "manifest_file": summary["manifest_file"],
}))
"""


def _prepare_inputs(base_atoms, count, work_dir):
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = work_dir / f"struct_{index:04d}.extxyz"
        write(path, base_atoms)
        paths.append(str(path))
    return paths


def _shard(paths, n_gpus):
    """Round-robin split so each GPU gets a comparable load."""
    shards = [[] for _ in range(n_gpus)]
    for index, path in enumerate(paths):
        shards[index % n_gpus].append(path)
    return shards


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-gpus", type=int, default=4)
    parser.add_argument(
        "--backend",
        default="nvalchemi-mace",
        choices=["nvalchemi-mace", "ase-mace"],
    )
    parser.add_argument("--checkpoint", default="medium")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--driver", default="energy", choices=["energy", "opt"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--n-structures", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.n_gpus < 1 or args.n_structures < 1:
        raise SystemExit("--n-gpus and --n-structures must be positive")
    out = args.output_dir.resolve()
    if out.exists():
        raise SystemExit("--output-dir must be a new directory")
    out.mkdir(parents=True)

    base_atoms = read(args.input)
    paths = _prepare_inputs(base_atoms, args.n_structures, out / "inputs")
    shards = _shard(paths, args.n_gpus)

    procs = []
    started = time.perf_counter()
    for gpu_index, shard in enumerate(shards):
        if not shard:
            continue
        cfg = {
            "inputs": shard,
            "backend": args.backend,
            "checkpoint": args.checkpoint,
            "dtype": args.dtype,
            "driver": args.driver,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "output_dir": str(out / f"gpu_{gpu_index}"),
        }
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_index))
        proc = subprocess.Popen(
            [sys.executable, "-c", _WORKER, json.dumps(cfg)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        procs.append((gpu_index, proc))

    workers = []
    total_succeeded = 0
    total_failed = 0
    for gpu_index, proc in procs:
        stdout, stderr = proc.communicate()
        record = {"gpu": gpu_index, "return_code": proc.returncode}
        try:
            record.update(json.loads(stdout.strip().splitlines()[-1]))
        except (ValueError, IndexError):
            record["error"] = stderr.strip()[-2000:]
        total_succeeded += record.get("succeeded", 0)
        total_failed += record.get("failed", 0)
        workers.append(record)
        (out / f"gpu_{gpu_index}_worker.log").write_text(stderr)

    wall = time.perf_counter() - started
    node_result = {
        "n_gpus": args.n_gpus,
        "backend": args.backend,
        "batch_size": args.batch_size,
        "n_structures": args.n_structures,
        "wall_time_s": wall,
        "node_throughput_structs_per_s": (
            total_succeeded / wall if wall > 0 else 0.0
        ),
        "total_succeeded": total_succeeded,
        "total_failed": total_failed,
        "workers": workers,
    }
    (out / "node_result.json").write_text(json.dumps(node_result, indent=2))
    print(json.dumps(node_result, indent=2))
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
