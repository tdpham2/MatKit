# MatKit 🧪

**MatKit** is a modular Python toolkit for building, managing, and validating simulation input files across a wide range of molecular and materials modeling software. It currently supports input generation for:

- 🟦 [RASPA2]=
- 🟨 RASPA3
- 🧬 gRASPA
- ⚛️ DFT tools (e.g., VASP, Quantum ESPRESSO)
- 💧 Molecular Dynamics tools (e.g., LAMMPS, GROMACS)

> MatKit is designed to accelerate research by unifying simulation setup workflows under a clean, reusable Python interface.

---

## 🚀 Features

- Modular structure for supporting multiple simulation engines
- Template-based input builders (Jinja2)
- CIF/XYZ/POSCAR file handling
- Force field management and validation
- HPC submission script generation (coming soon)
- CLI support (planned)
- Easy integration into existing workflows and Jupyter notebooks

---

## 📦 Installation

MatKit uses [PEP 621](https://peps.python.org/pep-0621/) and can be installed with `pip`:

```bash
# Editable install for development
git clone https://github.com/yourusername/matkit.git
cd matkit
pip install -e .
