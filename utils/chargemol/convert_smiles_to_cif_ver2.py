"""
Convert SMILES string to CIF file with bonding information. Please check bond details before proceeding, because all bonds are assigned as single bond
"""
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase import neighborlist
import subprocess
import re
import numpy as np
import networkx as nx
#  Check if a line is a coordinate of an atoms
PT = ['H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar',
          'K' , 'Ca', 'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
          'Rb', 'Sr', 'Y' , 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I' , 'Xe',
          'Cs', 'Ba', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr',
          'Ra', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Ac', 'Th',
          'Pa', 'U' , 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'FG', 'X' ]

def nn(string):
    return re.sub('[^a-zA-Z]','', string)

def nl(string):
    return re.sub('[^0-9]','', string)

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def iscoord(line):
    if nn(line[0]) in PT and line[1] in PT and False not in map(isfloat,line[2:5]):
        return True
    else:
        return False

def isbond(line):
    if nn(line[0]) in PT and nn(line[1]) in PT and '.' in line[3] and len(line) == 5:
        return True
    else:
        return False
# Update the current cif by removing nearest H bonds and changing atom name to X
def update_cif_with_connection_site(cif_in, connection_sites, cif_out):
    with open(cif_in, 'r') as f:
        data = f.readlines()
        data = list(filter(None, (i.strip() for i in data))) 
    X_sites = ['X'+str(i[1:]) for i in connection_sites]
    rem_index = []
    H_site = []
# 1st loop to convert connection site to remove X-H bond
    for site in connection_sites:
        for k, line in enumerate(data):
            line = line.split()
            if isbond(line) and site in line:
                if 'H' in line[0]:
                    print(line)
                    rem_index.append(k)
                    H_site.append(line[0])
                if 'H' in line[1]:
                    print(line)
                    rem_index.append(k)
                    H_site.append(line[1])
# 2nd loop to remove H coords
    for k, line in enumerate(data):
        line = line.split()
        for i in H_site:
            if iscoord(line) and i in line:
                rem_index.append(k)
    for index in sorted(rem_index, reverse=True):
        del data[index]

    with open(cif_out, 'w') as f:
        for line in data:
            f.write(line+'\n')

    for x, c in zip(X_sites, connection_sites):
        subprocess.call("sed -i 's/{}/{}/g' {}".format(c, x, cif_out), shell=True)

# Convert single bond to aromatic bond
def update_aromatic_bond(cif_in, cif_out):

    nodes = []
    edges = []

    with open(cif_in, 'r') as f:
        data = f.readlines()
        data = list(filter(None, (i.strip() for i in data)))
        for line in data:
            line = line.split()
            if iscoord(line):
                nodes.append(line[0])
            if isbond(line):
                edge = (line[0], line[1])
                edges.append(edge)
    # Create nx graph 
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    # Find cylic subgraph
    basis_cycles = nx.cycle_basis(G)
    new_file = []
    for index, line_ in enumerate(data):
        line = line_.split()
        if isbond(line):
            for cycle in basis_cycles:
                if line[0] in cycle and line[1] in cycle and line[-1] == 'S':
                   new_line = '    '.join([line[0], line[1], line[2], line[3], 'A'])
                else:
                   new_line = line_
            new_file.append(new_line)
        else:
            new_file.append(line_)
    print(new_file)
    with open(cif_out,'w') as f:
        for line in new_file:
            f.write(line+'\n')
### Global Variable ###
convert = True
update = True
fix_aromatic = True
 
if convert == True:
    smiles = ['c2ccc(Oc1ccccc1)cc2']
    fout = ['test_{}'.format(k) for k in range(len(smiles))]

    for index, smile in enumerate(smiles):
        print(smile)
        #subprocess.call("obabel -:'{}' -O test.xyz --gen3d --uff 10000".format(smile), shell=True)
        subprocess.call("obabel -:'{}' -O test.xyz --gen3d".format(smile), shell=True)
        structure = ase_read('test.xyz')
        a, b, c = 40, 40, 40
        structure.set_cell([a, b, c])
        structure.set_pbc(True)
        structure.center()
        ase_write('temp.cif', structure)

        cutOff = neighborlist.natural_cutoffs(structure)
        nl = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=False)
        nl.update(structure)

        # Keeping track of atom label and index
        names = []
        labels = []
        new_struct = ase_read('temp.cif')
        cutOff = neighborlist.natural_cutoffs(new_struct)
        nl = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=False)
        nl.update(new_struct)

        angs = new_struct.cell.angles()
        lengths = new_struct.cell.lengths()

        ### Text for CIF
        cell_text = "_cell_length_a                    "+str(lengths[0])+"\n_cell_length_b                    "+str(lengths[1])+"\n_cell_length_c                    "+str(lengths[2])+"\n_cell_angle_alpha                 "+str(angs[0])+"\n_cell_angle_beta                 "+str(angs[1])+"\n_cell_angle_gamma                 "+str(angs[2])+"\n"
        top = "_audit_creation_date              2019-12-23\n_audit_creation_method            'tobacco_3.0'\n_symmetry_space_group_name_H-M    'P1'\n_symmetry_Int_Tables_number       1\n_symmetry_cell_setting            triclinic\nloop_\n_symmetry_equiv_pos_as_xyz\n  x,y,z\n"
        med = "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_U_iso_or_equiv\n_atom_site_adp_type\n_atom_site_occupancy\n_atom_site_charge\n"
        bot = "loop_\n_geom_bond_atom_site_label_1\n_geom_bond_atom_site_label_2\n_geom_bond_distance\n_geom_bond_site_symmetry_2\n_ccdc_geom_bond_type\n"
        end_loop = '   0.00000  Uiso   1.00      0.00000\n'
        pos_text = ''
        bot_text = ''        
        pos = new_struct.get_scaled_positions()
        for k, atom in enumerate(new_struct):
            pos_text += atom.symbol+str(k+1)+'        '+atom.symbol+'   '+str(np.round(pos[k][0], 5))+'     '+str(
            np.round(pos[k][1], 5))+'     '+str(np.round(pos[k][2], 5))+end_loop

            parent_indices, offsets = nl.get_neighbors(k)
            for idx in parent_indices:
                d = new_struct.get_distance(k, idx, mic=True)
                bot_text += atom.symbol + \
                    str(k+1)+'   '+new_struct[idx].symbol + \
                    str(idx+1)+'   '+str(d)+'    .     S\n'

        ### Write CIF
        with open('L'+str(index+1)+'.cif', 'w') as w:
            w.write('data_'+'L'+str(index+1)+'\n'+top +
                    cell_text+med+pos_text+bot+bot_text)

        subprocess.call('rm temp.cif', shell=True)

    if update == True:
        connection_sites  = ['C3', 'C1']                      
        update_cif_with_connection_site('L1.cif', connection_sites, 'temp.cif')

    if fix_aromatic == True:
        update_aromatic_bond('temp.cif', 'L1_final.cif')
