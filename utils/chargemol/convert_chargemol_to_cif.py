from ase.io import read as ase_read
from ase.io import write as ase_write
import subprocess

atoms = []
x = []
y = []
z = []
charges = []
with open('DDEC6_even_tempered_net_atomic_charges.xyz', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if i == 0:
            natoms = int(line)
        elif i == 1:
            data = line.split()
            a1 = float(data[10])
            a2 = float(data[11])
            a3 = float(data[12])
            b1 = float(data[15])
            b2 = float(data[16])
            b3 = float(data[17])
            c1 = float(data[20])
            c2 = float(data[21])
            c3 = float(data[22])

        elif i <= natoms + 1:
            data = line.split()
            atoms.append(data[0])
            x.append(data[1])
            y.append(data[2])
            z.append(data[3])
            charges.append(float(data[4]))


with open('temp.xyz', 'w') as f:
    f.write('{}\n'.format(natoms))
    f.write("Lattice=\"{} {} {} {} {} {} {} {} {}\" Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n".format(a1, a2, a3, b1, b2, b3, c1, c2, c3))

    for d, x1, y1, z1 in zip(atoms, x, y, z):
        f.write("{} {} {} {}\n".format(d, x1, y1, z1))
def write_cif(fileobj, images, charges, format='default'):
    """ Description
    This is a clone of the ASE's write_cif funtion from the ase.io.cif module. It is modified so as to write the '_atom_site_charge' also
    while writing the CIF file.

    :type fileobj: string/file handle
    :param fileobj:path string or file handle to the output CIF

    :type images: ASE atoms object
    :param images: the atoms object you want to write to CIF format.

    :type format: string
    :param format: Some option found within the original function. Refer to ASE's documentation for more info.

    :raises:

    :rtype: None. Just writs the file.
    """

    def write_enc(fileobj, s):
        """Write string in latin-1 encoding."""
        fileobj.write(s.encode("latin-1"))

    from ase.utils import basestring
    from ase.parallel import paropen
    # from ase.io import cif

    """Write *images* to CIF file."""
    if isinstance(fileobj, basestring):
        fileobj = paropen(fileobj, 'wb')

    if hasattr(images, 'get_positions'):
        images = [images]

    for i, atoms in enumerate(images):
        write_enc(fileobj, 'data_image%d\n' % i)

        a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()

        if format == 'mp':

            comp_name = atoms.get_chemical_formula(mode='reduce')
            sf = split_chem_form(comp_name)
            formula_sum = ''
            ii = 0
            while ii < len(sf):
                formula_sum = formula_sum + ' ' + sf[ii] + sf[ii + 1]
                ii = ii + 2

            formula_sum = str(formula_sum)
            write_enc(fileobj, '_chemical_formula_structural       %s\n' %
                      atoms.get_chemical_formula(mode='reduce'))
            write_enc(fileobj, '_chemical_formula_sum      "%s"\n' %
                      formula_sum)

        # Do this only if there's three non-zero lattice vectors
        if atoms.number_of_lattice_vectors == 3:
            write_enc(fileobj, '_cell_length_a       %g\n' % a)
            write_enc(fileobj, '_cell_length_b       %g\n' % b)
            write_enc(fileobj, '_cell_length_c       %g\n' % c)
            write_enc(fileobj, '_cell_angle_alpha    %g\n' % alpha)
            write_enc(fileobj, '_cell_angle_beta     %g\n' % beta)
            write_enc(fileobj, '_cell_angle_gamma    %g\n' % gamma)
            write_enc(fileobj, '\n')

            write_enc(fileobj, '_symmetry_space_group_name_H-M    %s\n' %
                      '"P 1"')
            write_enc(fileobj, '_symmetry_int_tables_number       %d\n' % 1)
            write_enc(fileobj, '\n')

            write_enc(fileobj, 'loop_\n')
            write_enc(fileobj, '  _symmetry_equiv_pos_as_xyz\n')
            write_enc(fileobj, "  'x, y, z'\n")
            write_enc(fileobj, '\n')

        write_enc(fileobj, 'loop_\n')

        # Is it a periodic system?
        coord_type = 'fract' if atoms.pbc.all() else 'Cartn'

        if format == 'mp':
            write_enc(fileobj, '  _atom_site_type_symbol\n')
            write_enc(fileobj, '  _atom_site_label\n')
            write_enc(fileobj, '  _atom_site_symmetry_multiplicity\n')
            write_enc(fileobj, '  _atom_site_{0}_x\n'.format(coord_type))
            write_enc(fileobj, '  _atom_site_{0}_y\n'.format(coord_type))
            write_enc(fileobj, '  _atom_site_{0}_z\n'.format(coord_type))
            write_enc(fileobj, '  _atom_site_occupancy\n')
        else:
            write_enc(fileobj, '  _atom_site_label\n')
            write_enc(fileobj, '  _atom_site_occupancy\n')
            write_enc(fileobj, '  _atom_site_{0}_x\n'.format(coord_type))
            write_enc(fileobj, '  _atom_site_{0}_y\n'.format(coord_type))
            write_enc(fileobj, '  _atom_site_{0}_z\n'.format(coord_type))
            write_enc(fileobj, '  _atom_site_thermal_displace_type\n')
            write_enc(fileobj, '  _atom_site_B_iso_or_equiv\n')
            write_enc(fileobj, '  _atom_site_type_symbol\n')
            write_enc(fileobj, '  _atom_site_charge\n')

        if coord_type == 'fract':
            coords = atoms.get_scaled_positions().tolist()
        else:
            coords = atoms.get_positions().tolist()
        symbols = atoms.get_chemical_symbols()
        occupancies = [1 for i in range(len(symbols))]

        # try to fetch occupancies // rely on the tag - occupancy mapping
        try:
            occ_info = atoms.info['occupancy']

            for i, tag in enumerate(atoms.get_tags()):
                occupancies[i] = occ_info[tag][symbols[i]]
                # extend the positions array in case of mixed occupancy
                for sym, occ in occ_info[tag].items():
                    if sym != symbols[i]:
                        symbols.append(sym)
                        coords.append(coords[i])
                        occupancies.append(occ)
        except KeyError:
            pass

        no = {}

        for symbol, pos, occ, charge in zip(symbols, coords, occupancies, charges):
            if symbol in no:
                no[symbol] += 1
            else:
                no[symbol] = 1
            if format == 'mp':
                write_enc(fileobj,
                          '  %-2s  %4s  %4s  %7.5f  %7.5f  %7.5f  %6.1f %6.1f\n' %
                          (symbol, symbol + str(no[symbol]), 1,
                           pos[0], pos[1], pos[2], occ, charge))
            else:
                write_enc(fileobj,
                          '  %-8s %6.4f %7.5f  %7.5f  %7.5f  %4s  %6.3f  %s  %6.6f\n'
                          % ('%s%d' % (symbol, no[symbol]),
                             occ,
                             pos[0],
                             pos[1],
                             pos[2],
                             'Biso',
                             1.0,
                             symbol, charge))
    return None

mol = ase_read('temp.xyz')
mol.set_initial_charges(charges=charges)

write_cif('test.cif', mol, charges)
