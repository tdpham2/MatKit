#!/bin/bash -l

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MLIP_ENV="${MATKIT_MLIP_ENV:-${REPO_ROOT}/.venv}"

module use /soft/modulefiles
module load conda/2025-09-25

if [[ ! -x "${MLIP_ENV}/bin/python" ]]; then
    python -m venv "${MLIP_ENV}"
fi

source "${MLIP_ENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.nvidia.com \
    'nvalchemi-toolkit[cu12,mace]>=0.2,<0.3'

python -m pip install -e \
    "${REPO_ROOT}[mlip,rootstock,nvalchemi_mace]"

python - <<'PY'
from importlib.metadata import version

for package in (
    "matkit",
    "ase",
    "mace-torch",
    "rootstock",
    "nvalchemi-toolkit",
    "torch",
):
    print(f"{package}=={version(package)}")
PY

echo
echo "Environment installed at ${MLIP_ENV}"
echo "Check Rootstock access with: rootstock resolve --cluster polaris --json"
echo "Submit alcf/polaris/mlip/smoke.pbs from the MatKit checkout next."
