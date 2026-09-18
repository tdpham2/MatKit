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
            suffix = "psd_histo" if analysis == "psd" else analysis
            Path(f"structure.{suffix}").write_bytes(
                (fixtures / f"test_structure.{analysis}").read_bytes()
            )
elif mode == "graspa":
    fixture = (
        Path(__file__).parents[1] / "data" / "graspa" / "single_component.txt"
    )
    for line in fixture.read_text().splitlines():
        if "--partial" in sys.argv and line.startswith("Work time"):
            continue
        print(line)
else:
    raise SystemExit("unknown fixture engine")
