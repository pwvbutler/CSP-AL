from collections import defaultdict


def generate_datafile_str(comment, lattice_vectors, atoms, energy, charge):
    """
    generate input.data file content
    
    param:
        comment - str
            the comment line
        lattice_vectors - numpy (3, 3) array
            the direct lattice vectors
        atoms - [<dict>]
            contains dict for each atom including label, element, position, forces
        energy - float
            energy of system
        charge - float
            charge of system
    
    return:
        file_content - str
            the contents of the file as a single continuous string
    """
    content = ['begin']
    content.append(
        f'comment   {comment}'
    )
    for vector in lattice_vectors:
        content.append(
            "lattice   {:.16E} {:.16E} {:.16E}".format(*vector)
        )
    for atom_data in atoms:
        content.append(
            # x, y, z, elem, c, n, fx, fy, fz (in direct/cartesian)
            "atom   {:.16E} {:.16E} {:.16E} {} {} {} {:.16E} {:.16E} {:.16E}".format(
                float(atom_data[1]), # x
                float(atom_data[2]), # y
                float(atom_data[3]), # z
                str(atom_data[0]), # element
                0.0,
                0.0,
                float(atom_data[4]), # fx
                float(atom_data[5]), # fy
                float(atom_data[6]), # fz
            )
        )
    content.append(
        f"energy   {energy:.16E}"
    )
    content.append(
        f"charge   {charge:.16E}"
    )
    content.append('end\n')
    
    return "\n".join(content)


def parse_datafile_content(file_content):
    """
    parse input.data file content
    
    param:
        file_content - str
            the contents of the file as a single continuous string
    
    return:
        comment - str
            the comment line
        lattice_vectors - numpy (3, 3) array
            the direct lattice vectors
        atoms - [<dict>]
            contains dict for each atom including label, element, position, forces
        energy - float
            energy of system
        charge - float
            charge of system
    """
    structure_data = []
    entries = file_content.split('end')
    
    for entry in entries:
        contents = entry.split("\n")
        
        comment = None
        lattice_vectors = []
        atoms = []
        energy = None
        charge = None
        element_count = defaultdict(int)

        for line in contents:
            line_content = line.split()
            if len(line_content) < 1:
                continue

            if line_content[0] == 'comment' and len(line_content) > 1:
                comment = ' '.join(line_content[1:])
            elif line_content[0] == 'lattice':
                lattice_vectors.append(
                    [float(x) for x in line_content[1:]]
                )
            elif line_content[0] == 'atom':
                atom_coords = np.array([ float(x) for x in line_content[1:4] ]) # cartesian
                element = line_content[4] # element symbol
                forces = [ float(x) for x in line_content[7:] ] # # not currently used
                element_count[element] += 1
                label = element + str(element_count[element])
                atoms.append(
                    {
                        'label': label,
                        'element': element,
                        'position': atom_coords,
                        'forces': forces,
                    }
                )
            elif line_content[0] == 'energy':
                energy = float(line_content[1])
            elif line_content[0] == 'charge':
                charge = float(line_content[1])
            else:
                continue
        
        if atoms:
            structure_data.append(
                (comment, np.array(lattice_vectors), atoms, energy, charge)
            )
    
    return structure_data