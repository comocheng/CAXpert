# This script only work for fcc111 currently
# it only 
import os, logging
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from ase.db import connect
from caxpert.src.utils.utils import timeit
import numpy as np

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
