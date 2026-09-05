#!/usr/bin/env python3
"""Record opt-in execution evidence for explicitly supplied specifications."""

import argparse
import json
from pathlib import Path

from matkit.api import ExecutionConfig, run
from matkit.api.bundles import atomic_json, environment_versions
from matkit.api.structures import sha256
from matkit.operation_cli import resolve_request_paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", action="append", required=True, type=Path)
    parser.add_argument("--execution", type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.outdir.resolve()
    if root.exists():
        parser.error("Use a fresh evidence directory")
    profile = json.loads(args.execution.read_text()) if args.execution else {}
    execution = ExecutionConfig.model_validate(
        {**profile, "mode": "subprocess"}
    )
    root.mkdir(parents=True)
    report = {
        "kind": "execution evidence; not an accuracy benchmark",
        "environment": environment_versions(),
        "cases": [],
    }
    for index, spec in enumerate(args.spec):
        case = {
            "spec": str(spec.resolve()),
            "bundle": f"{index:05d}",
            "accepted": False,
        }
        try:
            case["spec_sha256"] = sha256(spec)
            request = resolve_request_paths(
                json.loads(spec.read_text()), spec.resolve().parent
            )
            result = run(
                request, output_dir=root / case["bundle"], execution=execution
            )
            case.update(
                accepted=result.accepted,
                state=result.state,
                failure=result.failure.model_dump() if result.failure else None,
            )
        except Exception as exc:
            case.update(state="failed", failure=str(exc))
        report["cases"].append(case)
        atomic_json(root / "execution_report.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if all(case["accepted"] for case in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
