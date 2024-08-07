import random, os, logging
import pandas as pd
from ase.io.trajectory import Trajectory
from ase.db import connect
from ase.atom import Atom
from ase.atoms import Atoms
from ase.build.surface import add_adsorbate
from icet.tools import enumerate_structures
from ..utils.error import AdsorbatesNotTaggedError, TooManyAdsorbatesError
from ..utils.utils import elements_place_holder

def generate_structures(prim_structure, adsorbates, ads_center_atom_ids, cell_size, db_path='init_structures.db', elements_place_holder=elements_place_holder):
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
    surface_z = max([atom.position[2] for atom in prim_structure if atom.tag !=2])
    top_layer_atom_index = [atom.index for atom in prim_structure if atom.position[2] == surface_z][0]
    for i in sorted(range(len(prim_structure)), reverse=True):
        atom = prim_structure[i]
        if atom.tag == 2:
            if i not in ads_center_atom_ids:
                del prim_structure[i]
            else:
                # new_atom = Atom('O', position=atom.position, tag=2)
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
        for struct in generated_structures:
            struct_to_db = struct.copy()
            struct_to_db.info['adsorbate_info'] = {'top layer atom index':top_layer_atom_index}
            for i in sorted(range(len(struct)),reverse=True):
                atom = struct[i]
                if atom.symbol == 'X':
                    del struct_to_db[i]
                elif atom.symbol in ads_identities.keys():
                    del struct_to_db[i]
                    add_adsorbate(struct_to_db, ads_identities[atom.symbol][0], atom.position[2]-surface_z, position=atom.position[:2], mol_index=ads_identities[atom.symbol][1])
            for i in range(len(struct_to_db)):
                print(struct_to_db[i])
            db.write(struct_to_db)
    logging.info('The structures have been generated and stored in the database.')

def cov_draw(db_path, output_csv):
    """
    This function randomly selects a structure from each coverage group and 
    stores the structure's index in a csv file.

    Args:
        db_path (str): the directory to the ASE database where the structures are stored.
        output_csv (str): the output csv file to store the ids of randomly selected structures.

    Returns:
        dict: a dictionary contain selected structures
        The dictionary contains the following keys:
            - "coverage" (float): The strcuture's index with the corresponding coverage.
    """
    ce_sys = {}
    db = connect(db_path)
    for index in range(len(db)):
        traj = db.get(sid=index).toatoms()
        site_num = sum(1 for atom in traj if atom.symbol == 'Ni') / 4
        ads_num = sum(1 for atom in traj if atom.symbol == 'C')
        cov = ads_num / site_num
        try:
            ce_sys[cov].append(int(index))
        except KeyError:
            ce_sys[cov] = [int(index)]

    random_samples = dict()
    covs = sorted([i for i in ce_sys.keys()])
    for k in covs:
        if k != 0:
            random_samples[k] = random.sample(ce_sys[k],1)
    
    df = pd.DataFrame(random_samples)
    if os.path.exists(output_csv):
        dfout = pd.read_csv(output_csv)
        df_new = dfout.append(df, ignore_index=True)
        df_new.to_csv(output_csv, index=False)
    else:
        df.to_csv(output_csv, index=False)
    return random_samples
