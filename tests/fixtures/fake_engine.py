"""Synthetic subprocess fixtures (MIT); no scientific accuracy claims."""

from pathlib import Path
import sys
import time

mode = sys.argv[1]
if "--sleep" in sys.argv:
    Path("started").write_text("started")
    time.sleep(30)
if "--fail" in sys.argv:
    print("intentional engine failure", file=sys.stderr)
    raise SystemExit(7)
if mode == "zeopp":
    fixtures = Path(__file__).parents[1] / "data" / "zeopp"
    for analysis in ("res", "sa", "vol", "psd", "chan"):
        if f"-{analysis}" in sys.argv and not (
            "--partial" in sys.argv and analysis == "sa"
        ):
            Path(f"structure.{analysis}").write_bytes(
                (fixtures / f"test_structure.{analysis}").read_bytes()
            )
elif mode == "graspa":
    for i in range(14):
        print(f"Overall: Average: {25 if i == 0 else i + 1}, +/- 0.1")
    if "--partial" not in sys.argv:
        print("Work time 2.0")
else:
    raise SystemExit("unknown fixture engine")
