#!/usr/bin/env python3
"""Opt-in live MLIP validation; each backend/driver uses a separate process."""

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from ase.io import read, write


def _probe(command):
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=30
        )
        return {
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def _record_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _validate_outputs(case_dir, structures, driver, fmax, batched):
    if batched:
        manifest = json.loads((case_dir / "batch_manifest.json").read_text())
        assert (
            manifest["pending"]
            == manifest["failed"]
            == manifest["unconverged"]
            == 0
        )
        assert [item["index"] for item in manifest["items"]] == list(
            range(len(structures))
        )
        paths = [Path(item["result_file"]) for item in manifest["items"]]
    else:
        paths = [case_dir / "result.json"]
    summaries = []
    for path, original in zip(paths, structures):
        result = json.loads(path.read_text())
        assert result["success"], result["error"]
        assert math.isfinite(result["energy"])
        assert result["energy_unit"] == "eV"
        assert result["force_unit"] == "eV/angstrom"
        forces = np.asarray(result["forces"])
        assert forces.shape == (len(original), 3) and np.isfinite(forces).all()
        final = result["final_structure"]
        assert final["atomic_numbers"] == original.numbers.tolist()
        assert final["pbc"] == original.pbc.tolist()
        assert np.allclose(
            final["cell"], original.cell.array, rtol=1e-6, atol=1e-6
        )
        positions = np.asarray(final["positions"])
        assert (
            positions.shape == (len(original), 3)
            and np.isfinite(positions).all()
        )
        if result["stress"] is not None:
            stress = np.asarray(result["stress"])
            assert stress.shape == (3, 3) and np.isfinite(stress).all()
        if driver == "opt":
            assert result["converged"], "optimization did not converge"
            assert np.linalg.norm(forces, axis=1).max() <= fmax + 1e-8
        else:
            assert np.allclose(positions, original.positions, rtol=0, atol=1e-6)
        summaries.append(
            {
                "result_file": str(path),
                "energy": result["energy"],
                "converged": result["converged"],
                "n_steps": result["n_steps"],
                "max_force": float(np.linalg.norm(forces, axis=1).max()),
            }
        )
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", default="medium")
    parser.add_argument("--rootstock-checkpoint", default="mace-mp-0-medium")
    parser.add_argument("--cluster", default="polaris")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--fmax", type=float, default=0.01)
    args = parser.parse_args(argv)
    if args.steps < 1 or not math.isfinite(args.fmax) or args.fmax <= 0:
        parser.error("steps and fmax must be positive; fmax must be finite")
    out = args.output_dir.resolve()
    if out.exists():
        parser.error("--output-dir must be a new directory")
    original = read(args.input)
    perturbed = original.copy()
    perturbed.rattle(stdev=0.01, seed=7)
    inputs_dir = out / "inputs"
    inputs_dir.mkdir(parents=True)
    input_paths = [
        inputs_dir / "original.extxyz",
        inputs_dir / "perturbed.extxyz",
    ]
    structures = [original, perturbed]
    for path, atoms in zip(input_paths, structures):
        write(path, atoms)
    packages = {}
    for name in (
        "matkit",
        "ase",
        "mace-torch",
        "rootstock",
        "nvalchemi-toolkit",
        "torch",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "matkit_revision": _probe(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parents[3]),
                "rev-parse",
                "HEAD",
            ]
        ),
        "packages": packages,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": _probe(["nvidia-smi"]),
        "rootstock_deployment": _probe(
            ["rootstock", "resolve", "--cluster", args.cluster, "--json"]
        ),
        "input_file": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "settings": {
            "checkpoint": args.checkpoint,
            "rootstock_checkpoint": args.rootstock_checkpoint,
            "cluster": args.cluster,
            "steps": args.steps,
            "fmax": args.fmax,
        },
        "cases": [],
        "note": (
            "Integration evidence only; this does not establish parity, "
            "performance, or scientific accuracy."
        ),
    }
    report_path = out / "smoke_report.json"
    _record_report(report_path, report)
    for backend in ("ase-mace", "rootstock", "nvalchemi-mace"):
        for driver in ("energy", "opt"):
            batched = backend == "nvalchemi-mace"
            case_dir = out / backend / driver
            case_dir.mkdir(parents=True)
            command = [
                sys.executable,
                "-m",
                "matkit.cli",
                "mlip",
                "run-batch" if batched else "run",
                "--backend",
                backend,
                "--driver",
                driver,
                "--device",
                "cuda",
            ]
            selected = list(range(2)) if batched else [int(driver == "opt")]
            for index in selected:
                command.extend(["--input", str(input_paths[index])])
            command.extend(
                [
                    "--outdir" if batched else "--output",
                    str(case_dir if batched else case_dir / "result.json"),
                ]
            )
            if backend == "rootstock":
                command.extend(
                    [
                        "--checkpoint",
                        args.rootstock_checkpoint,
                        "--cluster",
                        args.cluster,
                        "--timeout",
                        "1200",
                    ]
                )
            else:
                command.extend(
                    ["--checkpoint", args.checkpoint, "--dtype", "float32"]
                )
            if batched:
                command.extend(["--batch-size", "2"])
            if driver == "opt":
                command.extend(
                    ["--steps", str(args.steps), "--fmax", str(args.fmax)]
                )
            case = {
                "backend": backend,
                "driver": driver,
                "command": command,
                "status": "running",
            }
            report["cases"].append(case)
            _record_report(report_path, report)
            try:
                with (
                    (case_dir / "stdout.log").open("w") as stdout,
                    (case_dir / "stderr.log").open("w") as stderr,
                ):
                    process = subprocess.run(
                        command, stdout=stdout, stderr=stderr
                    )
                case["return_code"] = process.returncode
                case["results"] = _validate_outputs(
                    case_dir,
                    [structures[i] for i in selected],
                    driver,
                    args.fmax,
                    batched,
                )
                assert process.returncode == 0, (
                    f"CLI exited {process.returncode}"
                )
                case["status"] = "passed"
            except Exception as exc:
                case.update(
                    status="failed", error=str(exc) or type(exc).__name__
                )
            _record_report(report_path, report)
            print(f"{backend} {driver}: {case['status']}")
    print(f"Evidence report: {report_path}")
    return int(any(case["status"] != "passed" for case in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
