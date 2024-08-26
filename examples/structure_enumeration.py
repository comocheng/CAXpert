from ase.build import fcc111, molecule, add_adsorbate
from ase.io.trajectory import Trajectory
from ase.db import connect
import os
import numpy as np
from ase.constraints import FixAtoms
from caxpert.src.tasks.gen_str import generate_structures, select_covs, make_trajs, get_slabs_from_db

prim_structure = fcc111('Ni',size=(1,1,4), vacuum=13)
fix_layer = prim_structure[1].position[2]
prim_structure.set_tags([0 for i in range(len(prim_structure))])
prim_structure[3].tag = 1
prim_structure.set_constraint(FixAtoms([a.index for a in prim_structure if a.z <= fix_layer]))
for a in prim_structure:
    if a.symbol == 'Ni':
        a.magmom = 10.8

# Create an adsorbate structure
co = molecule('CO', vacuum=13, tags=[2,2])
h = molecule('H', vacuum=13, tags=[2])
# Create a tuple of the adsorbate and the index of the binding atom
adsorbate_list = [(co, 1), (h, 0)]
add_adsorbate(prim_structure, co, 1.8, position='fcc', offset=(0, 0), mol_index=1)
# Create a list of the indices of the center atoms of the adsorbates
ads_center_atom_ids = [a.index for a in prim_structure if a.symbol == 'C']
# Set the cell size, this is how many unit cells to use to make the super cell
cell_size = 10
# Generate the structures
generate_structures(prim_structure, adsorbate_list, ads_center_atom_ids, cell_size)

# select the structures with CO and H
co_h_ids = select_covs('init_structures.db', {'co':(0.3, 1), 'h':(0.1, 1)}, 10, total_atom_num_constraint=24)
# select the structures with only H
h_only_ids = select_covs('init_structures.db', {'co':(0, 0), 'h':(0.1, 1)}, 10, total_atom_num_constraint=24, output_db='dft_structures_h_only.db')

make_trajs(co_h_ids, 'dft_structures.db',  'dft_relax')
make_trajs(h_only_ids, 'dft_structures_h_only.db',  'dft_relax_h_only')

get_slabs_from_db('init_structures.db')

slabs_db = 'slabs.db'
with connect(slabs_db) as db:
    for i in os.listdir('slabs'):
        relaxed_atoms = Trajectory(f'slabs/{i}/relax.traj')[-1]
        forces = relaxed_atoms.get_forces()
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force < 0.05:
            db.write(relaxed_atoms)
        else:
            print(f'{i} is not converged')
