# PACMOF2 Installation on Aurora

PACMOF2 predicts partial atomic charges in Metal-Organic Frameworks with DFT-level accuracy.

Repository: https://github.com/snurr-group/pacmof2

## Installation

1. Load the frameworks module and create a virtual environment:
```bash
module load frameworks
python3 -m venv pacmof2_venv --system-site-packages
source pacmof2_venv/bin/activate
```

2. Clone the repository:
```bash
git clone https://github.com/snurr-group/pacmof2.git
cd pacmof2
```

3. Download the pre-trained models from HuggingFace:
```bash
wget -P pacmof2/models/ https://huggingface.co/tdphamm/PACMOF2/resolve/main/PACMOF2_neutral.gz
wget -P pacmof2/models/ https://huggingface.co/tdphamm/PACMOF2/resolve/main/PACMOF2_ionic.gz
```

4. Install PACMOF2 and its dependencies (ase, pymatgen, scikit-learn==1.3.2, tqdm):
```bash
pip install -e .
```

## Usage

```bash
pacmof2 --help
```

Alternatively, you can use the provided `install.sh` script to run all steps automatically:
```bash
bash install.sh
```
