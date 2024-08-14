import os, logging
from ase.db import connect
from ase.io.trajectory import Trajectory
from fairchem.data.oc.utils import DetectTrajAnomaly
from ase.neighborlist import NeighborList
from ase.data import covalent_radii
import networkx as nx
from collections import Counter
from ase.calculators.singlepoint import SinglePointCalculator

def check_problematic_structs(traj_path):
    trajs = Trajectory(traj_path)
    unique_tags = set(trajs[0].get_tags())
    for t in unique_tags:
        if t > 2 or t < 0: 
            raise ValueError(f'The tag {t} is not valid, the bulk atoms should be tagged as 0, the surface atoms should be tagged as 1, and the adsorbates should be taggged as 2')
    detector = DetectTrajAnomaly(trajs[0], trajs[1], trajs[0].get_tags())
    anom = (
            detector.is_adsorbate_dissociated()
            or detector.is_adsorbate_desorbed()
            or detector.has_surface_changed()
            or detector.is_adsorbate_intercalated()
        )
    return anom
    
class MakeTrainingDB:
    """
    A class to create a training database for the machine learning model.
    """
    def __init__(self, file_list, slab_db, ads_db, db_name='training_data/ml_train.db'):
        self.file_list = file_list
        self.db_name = db_name
        if not os.path.exists(slab_db):
            raise FileNotFoundError(f'{slab_db} does not exist.')
        self.slab_db = slab_db
        if not os.path.exists(ads_db):
            raise FileNotFoundError(f'{ads_db} does not exist.')
        self.ads_db = ads_db
    
    def count_adsorbates(self, atoms):
        """
        Count the different adsorbates in the structure.
        atoms: ase.Atoms
            The structure to count the adsorbates.
        """
        radiis = [covalent_radii[i] for i in atoms.get_atomic_numbers()]
        nl = NeighborList(radiis, self_interaction=False, bothways=True)
        nl.update(atoms)
        adsorbates = [atom for atom in atoms if atom.tag==2]
        G = nx.Graph()
        for a in adsorbates:
            G.add_node(a.index)
        
        for a in adsorbates:
            neighbor_list = nl.get_neighbors(a.index)[0]
            neighbor_list = [j for j in neighbor_list if atoms[j].tag != 1]
            for j in neighbor_list:
                G.add_edge(a.index, j)

        components = list(nx.connected_components(G))
        component_symbols = []
        for node in components:
            component_symbols.append([atoms[i].symbol for i in node])
        
        adsorbate_counts = dict()
        gas_refs = []
        with connect(self.ads_db) as db:
            for i in db.select():
                atms = i.toatoms()
                adsorbate_counts[atms.get_chemical_formula()] = 0
                gas_refs.append(atms)
        
        for i in component_symbols:
            for j in gas_refs:
                if Counter(i) == Counter(j.get_chemical_symbols()):
                    adsorbate_counts[j.get_chemical_formula()] += 1
        return adsorbate_counts

    def create_ase_database(self):
        slabs = dict()
        gas_refs = dict()
        with connect(self.slab_db) as db:
            for i in db.select():
                atoms = i.toatoms()
                slabs[atoms.cell.cellpar().tobytes()] = atoms
        
        with connect(self.ads_db) as db:
            for i in db.select():
                atoms = i.toatoms()
                gas_refs[atoms.get_chemical_formula()] = atoms

        db_dir = os.path.dirname(self.db_name)
        
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        with connect(self.db_name) as db:
            for file in self.file_list:
                if check_problematic_structs(file):
                    logging.warning(f'{file} has problematic structures, skip it.')
                    continue
                trajs = Trajectory(file)
                ads_counts = self.count_adsorbates(trajs[0])
                gas_ref_e = sum([gas_refs[i].get_potential_energy()*j for i, j in ads_counts.items()])
                if gas_ref_e == 0:
                    logging.warning(f'The gas reference energy is 0, skip it.')
                    continue
                slab_e = slabs[trajs[0].cell.cellpar().tobytes()].get_potential_energy()
                for traj in trajs:
                    binding_energy = traj.get_potential_energy() - slab_e - gas_ref_e
                    calc = SinglePointCalculator(traj, energy=binding_energy, forces=traj.get_forces())
                    traj.set_calculator(calc)
                    db.write(traj)
        logging.info(f'The training database is created at {self.db_name}')
