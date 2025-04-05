from ase.io import read, write
import sys
from ase.build import sort

atoms = read(sys.argv[1])
atoms_sorted = sort(atoms)

print(atoms_sorted)
write("POSCAR", atoms_sorted)
