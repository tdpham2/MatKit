# MOFid Installation on Aurora

MOFid is available at https://github.com/snurr-group/mofid

Steps:
1. Install anaconda/miniconda and create a new environment
```bash
mkdir env
conda create -p env/mofid_env python=3.10
```
2. Clone the MOFid repo:

```bash
git clone https://github.com/snurr-group/mofid
```
3. Install dependencies (gcc, g++, openjdk) (I only tested with 11.4.0 but should work with GCC 8 to 12 according to the MOFId repo)
```bash
conda install -c conda-forge gcc=11.4.0
conda install -c conda-forge gxx=11.4.0
conda install conda-forge::openjdk
```

4. Use the provided Makefile. Update the paths for MY_CC and MY_CXX variables.

5. Compile MOFid. See https://snurr-group.github.io/mofid/compiling/ for instructions.