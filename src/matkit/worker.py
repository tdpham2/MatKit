"""Internal subprocess entry point; communicate results through run bundles."""

import argparse
import os
from pathlib import Path
import signal

from matkit.api import ExecutionConfig
from matkit.api.runtime import _execute_batch_claimed, _execute_claimed


def _interrupt(signum, frame):
    raise KeyboardInterrupt(f"Worker received signal {signum}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args(argv)
    root = args.bundle.resolve()
    if os.environ.get("MATKIT_WORKER_PROCESS") != "1" or (
        root / ".matkit.lock"
    ).read_text() != str(os.getppid()):
        parser.error(
            "Workers must be launched by MatKit's supervising executor"
        )
    signal.signal(signal.SIGTERM, _interrupt)
    execution = ExecutionConfig.model_validate_json(
        (root / "execution.json").read_text()
    )
    if args.batch:
        result = _execute_batch_claimed(root, execution)
        return 0 if result.accepted else 1
    result = _execute_claimed(root, execution)
    return 0 if result.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
