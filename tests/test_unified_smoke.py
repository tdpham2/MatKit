"""Check execution evidence recording with a synthetic engine."""

import importlib.util
import json
from pathlib import Path
import sys


def test_evidence_recorder_continues_after_failure(sample_cif, tmp_path):
    script = Path(__file__).parents[1] / "examples" / "unified_smoke.py"
    spec = importlib.util.spec_from_file_location("unified_smoke", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps({"operation": "pores", "structure": {"path": sample_cif}})
    )
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    profile = tmp_path / "execution.json"
    engine = Path(__file__).parent / "fixtures" / "fake_engine.py"
    profile.write_text(
        json.dumps(
            {"executables": {"zeopp": [sys.executable, str(engine), "zeopp"]}}
        )
    )
    root = tmp_path / "evidence"
    assert (
        module.main(
            [
                "--spec",
                str(bad),
                "--spec",
                str(good),
                "--execution",
                str(profile),
                "--outdir",
                str(root),
            ]
        )
        == 1
    )
    report = json.loads((root / "execution_report.json").read_text())
    assert [case["accepted"] for case in report["cases"]] == [False, True]
    assert len(report["cases"][1]["spec_sha256"]) == 64
