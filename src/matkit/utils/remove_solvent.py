from ase.io import read as ase_read
from ase.io import write as ase_write
from ase import neighborlist
import networkx as nx
from ase.build import sort
import numpy as np


def remove_solvent(path_to_cif, output_path, mass_ratio=0.8, skin=0.3):
    """ Remove solvent, ions from a MOF using either chemical formula or ASE neighborlist

    Parameters:
        cif: str, absolute path to cif file
        output_path: str, path to write output cif
    """

    atoms = ase_read(path_to_cif)
    cutOff = neighborlist.natural_cutoffs(atoms)
    neighborList = neighborlist.NeighborList(cutOff, 
                                             self_interaction=False, 
                                             bothways=True, 
                                             skin=skin) 
    neighborList.update(atoms)
    G = nx.Graph()

    for k in range(len(atoms)):
        tup = (k, {"element": f"{atoms.get_chemical_symbols()[k]}", 
                   "pos": atoms.get_positions()[k]})
        G.add_nodes_from([tup])

    for k in range(len(atoms)):
        for i in neighborList.get_neighbors(k)[0]:
            G.add_edge(k, i)

    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    massG = []
    solvent_indice = []
    for index, g in enumerate(Gcc):
        g = list(g)
        fragment = atoms[g]
        fragment = sort(fragment)
        massG.append(sum(atoms[g].get_masses())) # Mass of each disconnected subgraph

    max_index = np.argmax(massG)
    for index, mass in enumerate(massG):
        if index == max_index:
            continue
        else:
            if mass/massG[max_index] < mass_ratio:
                for n in Gcc[index]:
                    solvent_indice.append(n)
    ions_index = sorted(solvent_indice, reverse=True)
    del atoms[ions_index]
    ase_write(f'{output_path}', atoms)


