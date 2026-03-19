#!/bin/bash

# ----- User Configuration ----- #
INPUT_DIR="${INPUT_DIR:-/lus/flare/projects/IQC/thang/SC_Hari/workflow/test/cifs}"
OUTPUT_DIR="${OUTPUT_DIR:-/lus/flare/projects/IQC/thang/SC_Hari/workflow/test/}"
ADSORBATE="${ADSORBATE:-H2}"
TEMPERATURE="${TEMPERATURE:-77.0}"
PRESSURE="${PRESSURE:-10000000}"
N_CYCLE="${N_CYCLE:-10000}"
GRASPA_BIN="${GRASPA_BIN:-gRASPA/graspa-sycl/bin/sycl.out}"
# ------------------------------- #

# Load modules and activate environment
module load frameworks
source /lus/flare/projects/IQC/thang/SC_Hari/pacmof2/pacmof2_venv/bin/activate

# Run the workflow
python workflow.py \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --adsorbate "${ADSORBATE}" \
    --temperature "${TEMPERATURE}" \
    --pressure "${PRESSURE}" \
    --n_cycle "${N_CYCLE}" \
    --graspa_bin "${GRASPA_BIN}" \

echo "Workflow finished with exit code $?"
