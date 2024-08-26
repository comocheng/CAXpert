# This script only work for fcc111 currently
# it only 
import os, logging
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from ase.db import connect
from caxpert.src.utils.utils import timeit
from caxpert.src.utils.error import StructuresNotValidatedError
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

class CalculateEnergy:
    """
    A class to perform DFT calculations using various ASE calculators.

    This class encapsulates the setup and execution of DFT calculations on atomic systems
    using different calculators provided by ASE.

    Attributes
    ----------
    init_traj: str
        Path to the file with the initial trajectory, the last structure is read from the init_traj.

    calculator: ase.calculators.calculator.Calculator
        An ASE calculator instance configured for the DFT calculation.
    
    restart: bool
        A flag to indicate if the calculation is a restart from a previous calculation,
        if True, the last structure from the init_traj is read and the new calculations are appended to the trajectory.
    """
    def __init__(self, init_traj, calculator, restart=False, fmax=0.03):
        self.init_traj = init_traj
        self.calculator = calculator
        self.restart = restart
        self.fmax = fmax
    
    @classmethod
    def init_for_db(cls, db_path, calculator, restart=False, fmax=0.03):
        """
        Initialize the class from a database entry.

        db_path: str 
            A path to the ASE database.
        calculator: ase.calculators.calculator.Calculator
            An ASE calculator instance configured for the DFT calculation.
        fmax: float
            The maximum force threshold for the relaxation.
        """
        return cls(db_path, calculator, restart, fmax)

    @timeit
    def calculate_energy(self):
        parent_dir = os.path.dirname(self.init_traj)
        traj_file = os.path.join(parent_dir, 'relax.traj')
        logfile = os.path.join(parent_dir, 'ase.log')
        if self.restart:
            adslab = Trajectory(traj_file)[-1]
            if np.max(np.linalg.norm(adslab.get_forces(), axis=1))<=self.fmax:
                logging.warning(f'Structure is relaxed under the force threshold, terminate it.')
                return 0
            traj_file = Trajectory(self.init_traj, 'a', adslab)
        else:
            adslab = Trajectory(self.init_traj)[0]
        with open(logfile, 'a') as f:
            f.write(f"Start DFT calculation:\n")
        if not adslab.constraints:
            logging.warning('The structure has no constraints, please make sure you do not need it!')
        adslab.calc = self.calculator
        opt_slab = BFGS(adslab, logfile=logfile, trajectory=traj_file)
        opt_slab.run(fmax=self.fmax)
        print('Done!')
    
    @timeit
    def calculate_energy_from_db(self, struct_id, output_path):
        """
        This function calculates the energy using DFT from a database verified by DFT after ML inference

        db_path: str 
            A path to the ASE database.
        struct_id: int 
            The structure id to calculate.
        output_path: str
            the output path to write Atoms.
        calculator: ase.calculators.calculator.Calculator
            An ASE calculator instance configured for the DFT calculation.
        fmax: float
            The maximum force threshold for the relaxation.
        """
        if '.db' not in self.init_traj:
            raise ValueError('The database path is not provided')
        with connect(self.init_traj) as db:
            adslab = db.get(struct_id).toatoms()
        logfile = os.path.join(os.path.dirname(output_path), 'ase.log')
        f_max = np.max(np.linalg.norm(adslab.get_forces(), axis=1))
        if f_max >= self.fmax:
            if not adslab.constraints:
                logging.warning('The structure has no constraints, please make sure you do not need it!')
            adslab.calc = self.calculator
            with open(logfile, 'a') as f:
                f.write(f"Start DFT calculation with the bottom 2 layers fixed:\n")
            opt_slab = BFGS(adslab, logfile=logfile, trajectory=output_path)
            opt_slab.run(fmax=self.fmax)
        else:
            print(f'Structure {struct_id} is relaxed under the force threshold, skip it.')

def ml_val(strut_ids, db_path, calculator,output_db, restart=False):
    """
    This function performs DFT single point calculations on the ML predicted structures in the database.
    strut_ids: list
        A list of structure ids in the initial structure database to calculate.
    db_path: str
        The path to the initial structure database.
    calculator: ase.calculators.calculator.Calculator
        An ASE calculator instance configured for the DFT calculation.
    output_db: str
        The path to the output database.
    restart: bool
        A flag to indicate if the calculation is a restart from a previous calculation, if set to True, 
        the function will read the output database and carry on to calculate the structures that are not validated.
        If set to True, the struct_ids variable will be ignored, so users can leave it as an empty list.
    """
    
    if restart:
        if not os.path.exists(output_db):
            raise FileNotFoundError(f'The output database {output_db} is not found, please make sure it exists.')
        structs = []
        restart_ids = []
        with connect(output_db) as db:
            for row in db.select():
                if not row.toatoms().get_chemical_formula():
                    structs.append(row.original_id)
        with connect(db_path) as db:
            for i in restart_ids:
                row = db.get(id=i)
                adslab = row.toatoms()
                adslab.calc = calculator
                structs.append((adslab, row.id, row.key_value_pairs))
    else:
        if os.path.exists(output_db):
            with connect(output_db) as db:
                for row in db.select():
                    if not row.toatoms().get_chemical_formula():
                        raise StructuresNotValidatedError('Some structures in the output database are not validated, run this function with restart mode.')

        structs = []
        with connect(db_path) as db , connect(output_db) as db_out:
            for i in strut_ids:
                row = db.get(id=i)
                adslab = row.toatoms()
                try:
                    db_out.get(original_id=row.id)
                    logging.warning(f'Structure {row.id} is already calculated, skip it.')
                except KeyError:
                    db_out.reserve(original_id=row.id) # reserve the id to save the indices randomly sampled            
                    adslab.calc = calculator
                    structs.append((adslab, row.id, row.key_value_pairs))

    for s, original_id, kvp in structs:
        # # temperary solution for Ni magmom, will be deleted in the future
        # for a in s:
        #     if a.symbol == 'Ni':
        #         a.magmom = 10.8
        energy = s.get_potential_energy()
        forces = s.get_forces()
        with connect(output_db) as db:
            calc = SinglePointCalculator(s, energy=energy, forces=forces)
            s.set_calculator(calc)
            index = db.get(original_id=original_id).id
            # db.write(s, original_id=original_id, key_value_pairs=kvp)
            db.update(index, s, data=kvp)
