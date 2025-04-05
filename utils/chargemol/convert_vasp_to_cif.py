from ase.io import read, write

atoms = read('CONTCAR')

write('test.cif', atoms)
