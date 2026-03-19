#!/bin/bash
# ============================================================
# Install PACMOF2 on ALCF Aurora
# Predicting partial atomic charges in MOFs (DFT-level accuracy)
# ============================================================

module load frameworks
python3 -m venv sim_venv --system-site-packages
source sim_venv/bin/activate

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

# Install matkit
cd ..
pip install git+https://github.com/tdpham2/MatKit.git

# Install parsl
pip install parsl

# Instal gRASPA-SYCL
git clone https://github.com/alvarovm/gRASPA
cd gRASPA/graspa-sycl
cmake -DUSE_SYCL=1 . -Bbuild
cd build && make
echo "Done. gRASPA-SYCL installed successfully."

