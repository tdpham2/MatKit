from ase.io import read as ase_read
from ase.io import write as ase_write
import sys
import subprocess
import glob
import numpy as np
import os

output = 'minimization' # 'all'
#output = 'all'
output_fname = glob.glob('Output/System_0/*')

if os.path.isfile('temp.dat'):
    subprocess.run('rm temp.dat', shell=True)
if len(output_fname) > 1:
    print('Found too many output file. Double check!')
else:
    output_fname = output_fname[0]
    subprocess.run("grep 'Final energy' {} | awk '{{print $5}}' > temp.dat".format(output_fname), shell=True)
    energies = []

    with open('temp.dat', 'r') as f:
        for line in f:
            energies.append(float(line.strip()))
    index = np.argmin(energies)
    print('Run_{} has lowest energy'.format(index))
    movie_path = os.path.join('Movies', 'System_0', 'Run_' + str(index))
    movie = glob.glob(movie_path + '/' + '*allcomponents.pdb')[0]
    cif_path = os.path.join('Movies', 'System_0', 'Framework_0_initial_1_1_1_P1.cif')

    data =  []
    all_elements = []
    all_positions = []

    start_indices = []
    end_indices = []

    if os.path.isfile(movie) and os.path.isfile(cif_path):
        with open(movie, 'r') as f:
            for line in f:
                data.append(line.strip())

        start_indices = [index for index, line in enumerate(data) if 'MODEL' in line]
        end_indices = [index for index, line in enumerate(data) if line == 'ENDMDL']
        labels = list(range(len(start_indices)))

        if output == 'minimization':
            framework = ase_read(cif_path)
            elements = []
            positions = []
            for i in range(start_indices[-1], end_indices[-1]):
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

            ase_write('optimized.cif', framework)
        elif output == 'all':
            for l, s, e in zip(labels, start_indices, end_indices):
                framework = ase_read(cif_path)

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

