import random, os, logging
import pandas as pd
from ase.io.trajectory import Trajectory
from ase.db import connect
from ase.atom import Atom
from ase.atoms import Atoms
from ase.build.surface import add_adsorbate
from ase.constraints import FixAtoms
from ase.io import write
from icet.tools import enumerate_structures
from ..utils.error import AdsorbatesNotTaggedError, TooManyAdsorbatesError, NoStructureMatchQueryError, SurfaceNotTaggedError, BulkTagError 
from ..utils.utils import elements_place_holder, is_generator_empty

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_structures(prim_structure, adsorbates, ads_center_atom_ids, cell_size, db_path='init_structures.db', elements_place_holder=elements_place_holder, fixed_layers=None):
    """
    This function enumerates structures using the Cluster Expansion Tool (ICET).
    prim_structure: ase.atom.Atoms or ase.atom.Atom or str
        The primitive structure to extend, can either be ase Atoms object, ase Atom object, or a path to a trajectory file.
    adsorbates: tuple, (ase.atom.Atoms, int)
        The adsorbates in the prim_str. 
        The tuple consists of the adsorbate structure and the index of the atoms binding to the surface
    ads_center_atoms: list
        The indices of center atoms of the adsorbates in the prim_str.
    cell_size: int
        The cell size to enumerate the structures.
    db_path: str
        The path to the ASE database to store the structures
    elements_place_holder: list
        The list of elements to use as place holders for adsorbates in the enumeration.
    """
    if len(adsorbates) > len(elements_place_holder):
        raise TooManyAdsorbatesError('Toom many adsorbates to enumerate, make it less than the elements_place_holder (default is 6).')
    if isinstance(prim_structure, Atoms) or isinstance(prim_structure, Atom):
        pass
    elif isinstance(prim_structure, str):
        prim_structure = Trajectory(prim_structure)[-1]
    else:  
        raise ValueError('The primitive structure should be an ASE Atoms object or a path to a trajectory file.')
    species = []
    if not any([tag == 2 for tag in prim_structure.get_tags()]):
        raise AdsorbatesNotTaggedError('The adsorbate atoms in the primitive structure should be tagged as 2.')
    if not any([tag == 1 for tag in prim_structure.get_tags()]):
        raise SurfaceNotTaggedError('The surface atoms in the primitive structure should be tagged as 1.')
    if not any([tag == 0 for tag in prim_structure.get_tags()]):
        raise BulkTagError('The bulk atoms in the primitive structure should be tagged as 0.')
    if any([tag not in [0,1,2] for tag in prim_structure.get_tags()]):
        raise ValueError('The tags should be 0, 1, or 2. 0 for bulk, 1 for surface, and 2 for adsorbates.')
    surface_z_coords = set([atom.position[2] for atom in prim_structure if atom.tag == 1])
    # only works with FixAtoms constraint
    constraints = prim_structure.constraints[0]
    if constraints:
        fixed_layers = [round(prim_structure[i].z, 2) for i in constraints.index]
    info = prim_structure.info.get('adsorbate_info', {})
    if 'top layer atom index' in info:
        top_layer_atom_index = info['top layer atom index']
        surface_z = prim_structure[info['top layer atom index']].position[2]
    else:
        surface_z = max([atom.position[2] for atom in prim_structure if atom.tag !=2])
        top_layer_atom_index = [atom.index for atom in prim_structure if atom.position[2] == surface_z][0]
    
    for i in sorted(range(len(prim_structure)), reverse=True):
        atom = prim_structure[i]
        if atom.tag == 2:
            if i not in ads_center_atom_ids:
                del prim_structure[i]
            else:
                prim_structure[i].symbol = 'O'
    ads_identities = dict()
    # set all the adsorbates to be tagged as 2
    for ads in adsorbates:
        for i in range(len(ads[0])):
            ads[0][i].tag = 2

    for i in range(len(adsorbates)):
        ads_identities[elements_place_holder[i]] = adsorbates[i]

    for atom in prim_structure:
        if atom.tag == 2:
            pool = ['X']
            for i in range(len(adsorbates)):
                pool.append(elements_place_holder[i])
            species.append(pool)
        else:
            species.append([atom.symbol])
    generated_structures = enumerate_structures(prim_structure, range(1, cell_size), species)
    with connect(db_path) as db:
        cov = dict()
        for struct in generated_structures:
            top_layer_atom_num = 0
            for ads in ads_identities.values():
                cov[ads[0].get_chemical_formula().lower()] = 0
            struct_to_db = struct.copy()
            struct_to_db.info['adsorbate_info'] = {'top layer atom index':top_layer_atom_index}
            # replace the place holder atoms with the adsorbates
            for i in sorted(range(len(struct)),reverse=True):
                atom = struct[i]
                if atom.symbol == 'X':
                    del struct_to_db[i]
                elif atom.symbol in ads_identities.keys():
                    del struct_to_db[i]
                    add_adsorbate(struct_to_db, ads_identities[atom.symbol][0], atom.position[2]-surface_z, position=atom.position[:2], mol_index=ads_identities[atom.symbol][1])
                    cov[ads_identities[atom.symbol][0].get_chemical_formula().lower()] += 1
                else:
                    if atom.position[2] in surface_z_coords:
                        top_layer_atom_num += 1
                        struct_to_db[i].tag = 1
                    else:
                        struct_to_db[i].tag = 0
            for ads in cov.keys():
                cov[ads] = round(cov[ads]/top_layer_atom_num, 3)

            if fixed_layers:
                constraint = FixAtoms([a.index for a in struct_to_db if round(a.z, 2) in fixed_layers])
                struct_to_db.set_constraint(constraint)
            else:
                logging.warning('No fixed layers are provided.')
            db.write(struct_to_db, top_layer_atom_index=top_layer_atom_index, **cov)
    logging.info('The structures have been generated and stored in the database.')

def select_covs(db_path, ads_ranges, structure_num, total_atom_num_constraint=False, output_db='dft_structures.db'):
    """
    This function randomly selects a structure from each coverage group and 
    stores the structure's index in a csv file.
    db_path: str 
        The directory to the ASE database where the structures are stored.
    ads_ranges: dict, {str: tuple}
        A dictionary of adsorbates (key) and their coverage ranges (value).
        The order of the adsorbates will determine the which adsorbate's coverage will be 
        proritized to query the database.
    structure_num: int
        The number of structures to randomly select from the filtered coverage combinations.
    total_atom_num_constraint: int
        The maximum number of atoms in the unit cell, if set, the structures with more atoms will be filtered out.
    output_db: str 
        The output db to store the randomly selected structures.

    Returns:
        dict: a dictionary contain selected structures
        The dictionary contains the following keys:
            - "coverage" (float): The strcuture's index with the corresponding coverage.
    """
    with connect(db_path) as db:
        query = []
        for ads in ads_ranges.keys():
            query.append(f'{ads.lower()}>={ads_ranges[ads][0]}')
            query.append(f'{ads.lower()}<={ads_ranges[ads][1]}')
        query = ','.join(query)
        filtered_structs = db.select(query)
        if is_generator_empty(filtered_structs):
            raise NoStructureMatchQueryError('No structure matches to your query in the database.')
        filtered_structures = []
        for struct in filtered_structs:
            if total_atom_num_constraint:
                if len(struct.toatoms()) > total_atom_num_constraint:
                    continue
                filtered_structures.append(struct)
            else:
                filtered_structures.append(struct)
    random_sample_size = min(structure_num, len(filtered_structures))
    if random_sample_size != structure_num:
        logging.warning(f'The number of structures to randomly select is greater than the number of structures matches to your query. Randomly selecting {random_samples} structures.')
    random_samples = random.sample(range(len(filtered_structures)), random_sample_size)
    samples_pool = []
    sample_ids = []
    for i in random_samples:
        samples_pool.append(filtered_structures[i])
    with connect(output_db) as dbout:
        for struct in samples_pool:
            logging.info(f'{struct.key_value_pairs}, the structure id is{struct.id}')
            sample_ids.append(struct.id)
            dbout.write(struct, key_value_pairs=struct.key_value_pairs, original_id=struct.id,round=1)
    logging.info(f'{len(samples_pool)} structures have been randomly selected and stored in the database.')
    if not output_db:
        return samples_pool
    return sample_ids

def make_trajs(struct_ids, fix_layer, target_db='dft_structures.db', dest_dir='dft_relax', tolerance=0.1):
    with connect(target_db) as db:
        for row in db.select():
            atoms = row.toatoms()
            constraints = FixAtoms([a.index for a in atoms if abs(a.z - fix_layer) < tolerance or a.z < fix_layer])
            atoms.set_constraint(constraints)
            for a in atoms:
                if a.symbol == 'Ni':
                    a.magmom = 10.8
            db.write(atoms, id=row.id, key_value_pairs=row.key_value_pairs)

    with connect(target_db) as db:
        for i in struct_ids:
            row = db.get(original_id=i)
            atoms = row.toatoms()
            dir_ = f'{dest_dir}/{row.original_id}'
            os.makedirs(dir_, exist_ok=True)
            write(f'{dir_}/init.traj', atoms)

def get_slabs_from_db(db_path, fix_layer, dest_path='slabs', tolerance=0.1):
    """
    This function reads the structures from the database and writes only the unique slabs to a directory,
    the slabs will be relaxed by DFT to calculate the adsorption energies of adsorbates.
    db_path: str
        The path to the ASE database where the structures are stored.
    dest_path: str
        The path to the directory to store the slabs.
    """
    slabs = dict()
    with connect(db_path) as db:
        for row in db.select():
            atoms = row.toatoms()
            slabs[atoms.cell.cellpar().tobytes()] = row.id
        
        for v in slabs.values():
            row = db.get(id=v)
            atoms = row.toatoms()
            for i in sorted(range(len(atoms)), reverse=True):
                if atoms[i].tag == 2:
                    del atoms[i]
            constraints = FixAtoms([a.index for a in atoms if abs(a.z - fix_layer) < tolerance or a.z < fix_layer])
            atoms.set_constraint(constraints)
            for a in atoms:
                if a.symbol == 'Ni':
                    a.magmom = 10.8
            os.makedirs(f'{dest_path}/{row.id}', exist_ok=True)
            write(f'{dest_path}/{row.id}/init.traj', atoms)
    logging.info(f'The slabs have been written to path {dest_path}.')
