import subprocess
import numpy as np
import networkx as nx
from ase.io import read as ase_read, write as ase_write
from ase import neighborlist

# Periodic table list
PT = {
    e: True
    for e in [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "FG",
        "X",
    ]
}


def isfloat(s):
    """Check if a string can be converted to a float."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_coord(line):
    """Check if a line represents atomic coordinates in CIF format."""
    return (
        line[0] in PT
        and line[1] in PT
        and all(isfloat(x) for x in line[2:5])
        and len(line) == 6
    )


def is_bond(line):
    """Check if a line represents a chemical bond in CIF format."""
    return line[0] in PT and line[1] in PT and "." in line[3] and len(line) == 5


def update_cif_with_connection_site(
    input_cif: str, connection_sites: list[str], output_cif: str
) -> None:
    """Process CIF file to prepare connection sites for linking.

    Args:
        input_cif: Path to input CIF file
        connection_sites: List of atom labels to be converted to connection sites
        output_cif: Path to output CIF file
    """
    # Read and clean input file
    with open(input_cif, encoding="utf-8") as f:
        cif_lines = [line.strip() for line in f if line.strip()]

    # Prepare connection site labels
    connection_labels = {site: f"X{site[1:]}" for site in connection_sites}
    # Track atoms and bonds to remove
    atoms_to_remove = set()
    lines_to_remove = set()
    # Find hydrogen atoms connected to connection sites
    for site in connection_sites:
        for line_idx, line in enumerate(cif_lines):
            atoms = line.split()
            if not is_bond(atoms) or site.strip() not in atoms:
                continue
            # Check both atoms in the bond
            for atom_idx in (0, 1):
                if atoms[atom_idx].startswith("H"):
                    atoms_to_remove.add(atoms[atom_idx])
                    lines_to_remove.add(line_idx)
    # Remove lines containing hydrogen atoms
    for line_idx, line in enumerate(cif_lines):
        atoms = line.split()
        if is_coord(atoms) and any(atom in atoms_to_remove for atom in atoms):
            lines_to_remove.add(line_idx)  # Filter out removed lines
    processed_lines = [
        line for idx, line in enumerate(cif_lines) if idx not in lines_to_remove
    ]
    with open(output_cif, "w", encoding="utf-8") as f:
        f.write("\n".join(processed_lines) + "\n")
    # Replace connection sites with X atoms
    for old_label, new_label in connection_labels.items():
        subprocess.run(
            ["sed", "-i", "-e", f"s|{old_label}|{new_label}|g", output_cif],
            check=True,
        )


def update_aromatic_bond(cif_in, cif_out):
    """Update CIF file by converting single bonds in aromatic rings to aromatic bonds."""
    with open(cif_in, encoding="utf-8") as f:
        data = [line.strip() for line in f if line.strip()]

    nodes, edges = [], []
    for line in data:
        tokens = line.split()
        if is_coord(tokens):
            nodes.append(tokens[0])
        elif is_bond(tokens):
            edges.append((tokens[0], tokens[1]))

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    cycles = nx.cycle_basis(G)

    new_lines = []
    for line in data:
        tokens = line.split()
        if is_bond(tokens) and tokens[-1] == "S":
            for cycle in cycles:
                if tokens[0] in cycle and tokens[1] in cycle:
                    tokens[-1] = "A"
                    break
            new_lines.append("    ".join(tokens))
        else:
            new_lines.append(line)

    with open(cif_out, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")


def smiles_to_cif(smiles, output_prefix="L1"):
    """Convert SMILES string to CIF format with bonding information."""
    xyz_file = "temp.xyz"
    cif_file = "temp.cif"
    final_cif = f"{output_prefix}.cif"

    subprocess.call(f"obabel -:'{smiles}' -O {xyz_file} --gen3d", shell=True)
    struct = ase_read(xyz_file)
    struct.set_cell([40, 40, 40])
    struct.set_pbc(True)
    struct.center()
    ase_write(cif_file, struct)

    cutoffs = neighborlist.natural_cutoffs(struct)
    nl = neighborlist.NeighborList(
        cutoffs, self_interaction=False, bothways=False
    )
    nl.update(struct)

    lengths = struct.cell.lengths()
    angles = struct.cell.angles()

    pos_block = ""
    bond_block = ""
    scaled_pos = struct.get_scaled_positions()
    for i, atom in enumerate(struct):
        label = f"{atom.symbol}{i + 1}"
        x, y, z = np.round(scaled_pos[i], 5)
        pos_block += f"{label:<10}{atom.symbol:<6}{x:<10}{y:<10}{z:<10}0.00000  Uiso   1.00      0.00000\n"
        indices, _ = nl.get_neighbors(i)
        for j in indices:
            d = struct.get_distance(i, j, mic=True)
            bond_block += f"{label:<6}{struct[j].symbol}{j + 1:<6}{round(d, 4):<8}.     S\n"

    with open(final_cif, "w", encoding="utf-8") as f:
        f.write(
            f"data_{output_prefix}\n"
            "_audit_creation_date              2025-06-02\n"
            "_audit_creation_method            'tobacco_3.0'\n"
            "_symmetry_space_group_name_H-M    'P1'\n"
            "_symmetry_Int_Tables_number       1\n"
            "_symmetry_cell_setting            triclinic\n"
            "loop_\n"
            "_symmetry_equiv_pos_as_xyz\n"
            "  x,y,z\n"
            f"_cell_length_a                    {lengths[0]}\n"
            f"_cell_length_b                    {lengths[1]}\n"
            f"_cell_length_c                    {lengths[2]}\n"
            f"_cell_angle_alpha                 {angles[0]}\n"
            f"_cell_angle_beta                 {angles[1]}\n"
            f"_cell_angle_gamma                 {angles[2]}\n"
            "loop_\n"
            "_atom_site_label\n"
            "_atom_site_type_symbol\n"
            "_atom_site_fract_x\n"
            "_atom_site_fract_y\n"
            "_atom_site_fract_z\n"
            "_atom_site_U_iso_or_equiv\n"
            "_atom_site_adp_type\n"
            "_atom_site_occupancy\n"
            "_atom_site_charge\n"
            f"{pos_block}"
            "loop_\n"
            "_geom_bond_atom_site_label_1\n"
            "_geom_bond_atom_site_label_2\n"
            "_geom_bond_distance\n"
            "_geom_bond_site_symmetry_2\n"
            "_ccdc_geom_bond_type\n"
            f"{bond_block}"
        )

    subprocess.call(f"rm {xyz_file} {cif_file}", shell=True)
    return final_cif


def create_linker(
    smiles: str,
    connection_sites: list[str],
    output_cif: str = "final_output.cif",
):
    """
    Create a linker CIF from a SMILES string and a list of connection sites.
    """
    initial_cif = smiles_to_cif(smiles)
    update_cif_with_connection_site(initial_cif, connection_sites, "temp.cif")
    update_aromatic_bond("temp.cif", output_cif)

    # Cleanup intermediate files if needed, or keep them.
    # For now, we leave them as the original script did?
    # The original script does `rm temp.cif` implicitely by overwriting or maybe not.
    # Actually, original script leaves `temp.cif`.
    # We will just print success.
    print(f"Successfully created linker: {output_cif}")


# === Run the Workflow ===
if __name__ == "__main__":
    # Example usage when running directly
    # smiles = "[nH]1nnnc1"
    smiles = "Nc1nc[nH]n1"
    connection_sites = ["N3 ", "N5 ", "N6 "]
    create_linker(smiles, connection_sites)
