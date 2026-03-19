#!/bin/bash
# ============================================================
# Install PACMOF2 on ALCF Aurora
# Predicting partial atomic charges in MOFs (DFT-level accuracy)
# ============================================================

module load frameworks
python3 -m venv pacmof2_venv --system-site-packages
source pacmof2_venv/bin/activate

# Clone the repository
git clone https://github.com/snurr-group/pacmof2.git
cd pacmof2

# Download pre-trained models from HuggingFace
wget -P pacmof2/models/ https://huggingface.co/tdphamm/PACMOF2/resolve/main/PACMOF2_neutral.gz
wget -P pacmof2/models/ https://huggingface.co/tdphamm/PACMOF2/resolve/main/PACMOF2_ionic.gz

# Install PACMOF2 and its dependencies
# (ase, pymatgen, scikit-learn==1.3.2, tqdm)
pip install -e .
echo "Done. PACMOF2 installed successfully."
echo "Usage: pacmof2 --help"
