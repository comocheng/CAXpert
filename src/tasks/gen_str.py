import random, os, logging
import pandas as pd
from ase.io.trajectory import Trajectory
from ase.db import connect
from ase.atom import Atom
from ase.atoms import Atoms
from ase.build.surface import add_adsorbate
from icet.tools import enumerate_structures
from ..utils.error import AdsorbatesNotTaggedError, TooManyAdsorbatesError, NoStructureMatchQueryError
from ..utils.utils import elements_place_holder, is_generator_empty

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_structures(prim_structure, adsorbates, ads_center_atom_ids, cell_size, surf_layer_tol=0.1, db_path='init_structures.db', elements_place_holder=elements_place_holder):
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
    surf_layer_tol: float
        The tolerance for identify if an atom is in the top layer of the surface.
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
                    if atom.position[2] < surface_z + surf_layer_tol and atom.position[2] > surface_z - surf_layer_tol:
                        top_layer_atom_num += 1
            for ads in cov.keys():
                cov[ads] = round(cov[ads]/top_layer_atom_num, 3)
            db.write(struct_to_db, top_layer_atom_index=top_layer_atom_index, **cov)
    logging.info('The structures have been generated and stored in the database.')

def select_covs(db_path, ads_ranges, structure_num, output_db='dft_structures.db'):
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
            filtered_structures.append(struct)
    random_sample_size = min(structure_num, len(filtered_structures))
    if random_sample_size != structure_num:
        logging.warning(f'The number of structures to randomly select is greater than the number of structures matches to your query. Randomly selecting {random_samples} structures.')
    random_samples = random.sample(range(len(filtered_structures)), random_sample_size)
    samples_pool = []
    for i in random_samples:
        samples_pool.append(filtered_structures[i])
    with connect(output_db) as dbout:
        for struct in samples_pool:
            print(struct.key_value_pairs, struct.id)
            dbout.write(struct, key_value_pairs=struct.key_value_pairs, original_id=struct.id,round=1)
    logging.info(f'{len(samples_pool)} structures have been randomly selected and stored in the database.')
    if not output_db:
        return samples_pool
