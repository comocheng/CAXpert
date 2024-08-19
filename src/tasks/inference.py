import time, os
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from functools import wraps
from ase.optimize import BFGS
from ase.db import connect
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
import numpy as np
from ase.io.trajectory import Trajectory
from caxpert.src.utils.utils import timeit


def ml_validate(checkpoint_path, database_path, trainer='equiformerv2_forces', fig_path='parity_plot.png'):
    """
    Validate the ML model using the test set.
    checkpoint_path: str
        The path to the checkpoint file.
    database_path: str
        The path to the test database.
    trainer: str
        The trainer to pass to the OCPCalculator.
    fig_path: str
        The path to save the parity plot.
    """
    calc = OCPCalculator(checkpoint_path=checkpoint_path, trainer=trainer)
    db = connect(database_path)
    trajs = []
    for row in db.select():
        trajs.append(row.toatoms())
    traj_e_dfts = [traj.get_potential_energy() for traj in trajs]
    fmax_e_dfts = [np.max(np.linalg.norm(traj.get_forces(), axis=1)) for traj in trajs]
    traj_e_ocps = []
    fmax_e_ocps = []
    for traj in trajs:
        traj.calc = calc
        traj_e_ocps.append(traj.get_potential_energy())
        fmax_e_ocps.append(np.max(np.linalg.norm(traj.get_forces(), axis=1)))
    plt.figure(figsize=(6, 6))
    plt.scatter(traj_e_dfts, traj_e_ocps, color='b', marker='o', label='ML predictions')
    plt.plot([min(traj_e_dfts), max(traj_e_dfts)], [min(traj_e_ocps), max(traj_e_ocps)], color='r', linestyle='--')
    plt.xlabel('DFT')
    plt.ylabel('ML predictions')
    plt.title('Parity Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_path)
    return mean_squared_error(traj_e_dfts, traj_e_ocps, squared=True), mean_squared_error(fmax_e_dfts, fmax_e_ocps, squared=True)

@timeit
def ml_relax_db(input_db, checkpoint_path, start_id, output_path='', interval=1000, log_file='-', fmax=0.03, steps=300, trainer='equiformerv2_forces'):
    """
    Relax the structures in the database using the ML model.
    This function is designed to be used with SLURM job arrays.
    The structure database can be split into intervals and each interval can be relaxed in parallel.
    input_db: str
        The path to the database with the structures to relax.
    checkpoint_path: str
        The path to the checkpoint file.
    start_id: int
        The ID of the first structure to relax.
    output_path: str
        The path to save the relaxed structures.
    interval: int
        The number of structures to relax in each job.
    log_file: str
        The path to log the relax history.
    fmax: float
        The maximum force for the relaxation.
    steps: int
        The number of steps for the relaxation.
    trainer: str
        The trainer to pass to the OCPCalculator.
    """
    start_id = int(start_id)
    stop_id = start_id + interval
    output_traj = os.path.join(output_path, f'ml_inf_{start_id}_to_{stop_id}.traj')
    if os.path.exists(output_traj):
        start_id = int(start_id) + len(Trajectory(output_traj)) - 1 

    query = f'id>={start_id},id<{stop_id}'
    with connect(input_db) as db:
        for row in db.select(query):
            adslab = row.toatoms()
            with Trajectory(output_traj, 'a') as traj:
                calc = OCPCalculator(checkpoint_path=checkpoint_path, trainer=trainer)   
                adslab.calc = calc
                opt_slab = BFGS(adslab, logfile=log_file)
                opt_slab.run(fmax=fmax, steps=steps)
                traj.write(adslab)
    print('Done!')
