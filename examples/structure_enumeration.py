from ase.build import fcc111, molecule, add_adsorbate
from caxpert.src.tasks.gen_str import generate_structures, select_covs
from ase.db import connect
from ase.constraints import FixAtoms
import os
from ase.io import write

prim_structure = fcc111('Ni',size=(1,1,4), vacuum=13)
prim_structure.set_tags([0 for i in range(len(prim_structure))])
# Create an adsorbate structure
co = molecule('CO', vacuum=13, tags=[2,2])
h = molecule('H', vacuum=13, tags=[2])
# Create a tuple of the adsorbate and the index of the binding atom
adsorbate_list = [(co, 1), (h, 0)]
add_adsorbate(prim_structure, co, 1.8, position='fcc', offset=(0, 0), mol_index=1)
# Create a list of the indices of the center atoms of the adsorbates
ads_center_atom_ids = [a.index for a in prim_structure if a.symbol == 'C']
# Set the cell size
cell_size = 10
# Generate the structures
generate_structures(prim_structure, adsorbate_list, ads_center_atom_ids, cell_size)

select_covs('init_structures.db', {'co':(0, 1), 'h':(0.1, 1)}, 10)
with connect('dft_structures.db') as db:
    for row in db.select():
        atoms = row.toatoms()
        fix_layer = atoms[1].z 
        constraints = FixAtoms([a.index for a in atoms if abs(a.z - fix_layer) < 0.1 or a.z < fix_layer])
        atoms.set_constraint(constraints)
        for a in atoms:
            if a.symbol == 'Ni':
                a.magmom = 10.8
        db.write(atoms, id=row.id, key_value_pairs=row.key_value_pairs)

with connect('dft_structures.db') as db:
    for row in db.select():
        atoms = row.toatoms()
        dir_ = f'dft_relax/{row.original_id}'
        os.makedirs(dir_, exist_ok=True)
        write(f'{dir_}/init.traj', atoms)
