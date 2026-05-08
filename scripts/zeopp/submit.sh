#!/bin/bash -l
# ============================================================
# PBS job script: Zeo++ pore analysis via matkit
# ============================================================
#PBS -l walltime=02:00:00
#PBS -l select=1:system=crux
#PBS -l filesystems=home:eagle
#PBS -q workq-route
#PBS -A IQC
#PBS -N zeopp

cd $PBS_O_WORKDIR

# ----- User Configuration ----- #
CIF_DIR="${CIF_DIR:-.}"
OUTPUT_DIR="${OUTPUT_DIR:-./zeopp_results}"
NETWORK_BIN="${NETWORK_BIN:-$HOME/soft/zeo++-0.3/network}"
NUM_SAMPLES="${NUM_SAMPLES:-100000}"
PROBE_RADIUS="${PROBE_RADIUS:-1.86}"
CHAN_RADIUS="${CHAN_RADIUS:-1.86}"
MAX_WORKERS="${MAX_WORKERS:-32}"
# ------------------------------- #

matkit zeopp run-batch \
    --cif-dir "$CIF_DIR" \
    --outdir "$OUTPUT_DIR" \
    --analysis res \
    --analysis sa \
    --network-path "$NETWORK_BIN" \
    --probe-radius "$PROBE_RADIUS" \
    --chan-radius "$CHAN_RADIUS" \
    --num-samples "$NUM_SAMPLES" \
    --max-workers "$MAX_WORKERS"

echo "Finished with exit code $?"
