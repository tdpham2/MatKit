"""Exercise real supervision after a synthetic calculation has committed."""

from pathlib import Path
import sys
import time

from matkit.api import ExecutionConfig
from matkit.api.runtime import _execute_batch_claimed, _execute_claimed

root = Path(sys.argv[1])
behavior = sys.argv[2]
config = ExecutionConfig.model_validate_json(
    (root / "execution.json").read_text()
)
execute = _execute_batch_claimed if "--batch" in sys.argv else _execute_claimed
result = execute(root, config)
assert result.accepted
(root / "worker.finished").write_text("scientific result committed")
if behavior == "hang":
    time.sleep(60)
elif behavior == "teardown_error":
    raise RuntimeError("fixture teardown failed after commit")
elif behavior == "bad_exit":
    raise SystemExit(7)
