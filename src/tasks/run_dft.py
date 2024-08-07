# This script only work for fcc111 currently
# it only 
import os, logging
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from ase.db import connect
from caxpert.src.utils.utils import timeit, check_reconstruction
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
        if f_max > self.fmax and not check_reconstruction(adslab):
            if not adslab.constraints:
                logging.warning('The structure has no constraints, please make sure you do not need it!')
            adslab.calc = self.calculator
            with open(logfile, 'a') as f:
                f.write(f"Start DFT calculation with the bottom 2 layers fixed:\n")
            opt_slab = BFGS(adslab, logfile=logfile, trajectory=output_path)
            opt_slab.run(fmax=self.fmax)
        else:
            print(f'Structure {struct_id} is relaxed under the force threshold, skip it.')
        
    # @timeit
    # def optimize_slab(self):
    #     logfile = 'ase_lb.log'
    #     # relax the slab first
    #     slab = Trajectory('slab.traj')[-1]
    #     slab_relax_traj = 'slab_relax.traj'

    #     with open(logfile, 'a') as f:
    #         f.write(f"Start DFT calculation:\n")
    #     if not slab.constraints:
    #         logging.warning('The structure has no constraints, please make sure you do not need it!')
    #     print(slab.constraints)
    #     slab = self.set_up_calculator(slab)
    #     opt_slab = BFGS(slab, logfile=logfile, trajectory=slab_relax_traj)
    #     opt_slab.run(fmax=self.calc_settings['fmax'])
    #     print('Done!')


# if __name__ == "__main__":
#     dft_cmd = sys.argv[1].lower()
#     # qe_setting_p = os.path.join(os.path.dirname(os.getcwd()), '../qe_settings.yaml')
#     qe_setting_p = '/global/cfs/cdirs/m4126/xuchao/ce_models/co_ni/qe_settings.yaml'
#     if dft_cmd == 'dft-ml':
#         db_path = sys.argv[2]
#         struct_id = sys.argv[3]
#         output_path = sys.argv[4]
#         CalculateEnergy(qe_setting_p).calculate_energy_from_db(db_path, struct_id, output_path)
#     elif dft_cmd == 'dft':
#         CalculateEnergy(qe_setting_p).calculate_energy()
#     elif dft_cmd == 'slab':
#         CalculateEnergy(qe_setting_p).optimize_slab()
