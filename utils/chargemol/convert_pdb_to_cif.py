from ase.io import read as ase_read
from ase.io import write as ase_write
import sys
fname = sys.argv[1]

data =  []
all_elements = []
all_positions = []

start_indices = []
end_indices = []

with open(fname, 'r') as f:
    for line in f:
        data.append(line.strip())

start_indices = [index for index, line in enumerate(data) if 'MODEL' in line]
end_indices = [index for index, line in enumerate(data) if line == 'ENDMDL']
labels = list(range(len(start_indices)))
for l, s, e in zip(labels, start_indices, end_indices):
    framework = ase_read('Framework_0_final_1_1_1_P1.cif')

    elements = []
    positions = []
    for i in range(s, e):
        if 'CRYST1' in data[i]:
            cell_params = data[i].split()[1:]
            cell_params = [float(i) for i in cell_params]

        elif 'ATOM' in data[i]:
            coords = data[i].split()
            elements.append(coords[2])
            pos = [float(i) for i in coords[4:7]]
            positions.append(pos)

    with open('temp.xyz', 'w') as f:
        f.write("{}\n\n".format(len(elements)))
        for i, j in zip(elements, positions):
            f.write("{} {} {} {}\n".format(i, j[0], j[1], j[2]))

    atoms = ase_read('temp.xyz')
    atoms.set_cell(cell_params)
    framework.extend(atoms)

    ase_write('{}.cif'.format(l), framework)
"""
end_index = data.index('ENDMDL')

for index, line in enumerate(data):
    if 'CRYST1' in line:
        start_index = index
        cell_params = line.split()[1:]
        cell_params = [float(i) for i in cell_params]

for k in range(start_index + 1, end_index):
    line = data[k].split()
    elements.append(line[2])
    positions.append(line[4:7])

with open('temp.xyz', 'w') as f:
    f.write("{}\n\n".format(len(elements)))
    for i, j in zip(elements, positions):
        f.write("{} {} {} {}\n".format(i, j[0], j[1], j[2]))

atoms = ase_read('temp.xyz')
atoms.set_cell(cell_params)
#ase_write('temp.cif', system)

framework.append(atoms)

print(framework)
"""
