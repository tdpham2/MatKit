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
NETWORK_BIN="${NETWORK_BIN:-$HOME/soft/zeo++-0.3/network}"
NUM_SAMPLES="${NUM_SAMPLES:-100000}"
PROBE_RADIUS="${PROBE_RADIUS:-1.86}"
CHAN_RADIUS="${CHAN_RADIUS:-1.86}"
# ------------------------------- #

for cif in "${CIF_DIR}"/*.cif; do
    [ -f "$cif" ] || continue
    name=$(basename "$cif" .cif)
    echo "Running Zeo++ on $name ..."

    matkit zeopp run \
        --cif "$cif" \
        --analysis res \
        --analysis sa \
        --network-path "$NETWORK_BIN" \
        --probe-radius "$PROBE_RADIUS" \
        --chan-radius "$CHAN_RADIUS" \
        --num-samples "$NUM_SAMPLES" \
        --outdir "${CIF_DIR}/${name}_zeopp" &
done

wait
echo "All Zeo++ jobs finished with exit code $?"
