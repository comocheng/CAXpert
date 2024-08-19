import time, os
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from functools import wraps
from ase.optimize import BFGS
from ase.db import connect
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds to complete.")
        return result
    return wrapper

@timeit
# use lock to ensure the database is not accessed by multiple processes 
def ml_relax(adslab, checkpoint_path, db_path, lock=None, data_to_db=None, traj_file=None, log_file='-', fmax=0.03, steps=300, trainer='equiformerv2_forces'):
    calc = OCPCalculator(checkpoint_path=checkpoint_path, trainer=trainer)   
    adslab.calc = calc
    opt_slab = BFGS(adslab)
    opt_slab.run(fmax=fmax, steps=steps, logfile=log_file, trajectory=traj_file)
    if lock is not None:
        lock.acquire()
    # try statement to ensure the lock is released
    try:
        with connect(db_path) as db:
            id = db.reserve(adslab)
            if id is not None:
                if data_to_db is not None:
                    data_to_db(adslab, id=id, key_value_pairs=data_to_db)
                db.write(adslab, id=id)
    finally:
        if lock is not None:
            lock.release()
    print('Done!')

def ml_validate(checkpoint_path, database_path, trainer='equiformerv2_forces', fig_path='parity_plot.png'):
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

# def ml_relax_db(struct_id, input_db, checkpoint_path, db_path, traj_file=None, log_file='-', fmax=0.03, steps=300, trainer='equiformerv2_forces'):
#     with connect(input_db) as db:
#         row = db.get(struct_id)
#         adslab = row.toatoms()
#         data_to_db = row.key_value_pairs
#     calc = OCPCalculator(checkpoint_path=checkpoint_path, trainer=trainer)   
#     adslab.calc = calc
#     opt_slab = BFGS(adslab)
#     with connect(db_path) as db:
#         sid = db.reserve(id=struct_id)
#         if sid is not None:
#             if 2 in adslab.get_tags():
#                 opt_slab.run(fmax=fmax, steps=steps, logfile=log_file, trajectory=traj_file)
#                 db.write(adslab, id=id, key_value_pairs=data_to_db)
#             else:
#                 calc = SinglePointCalculator(adslab, energy=0, forces=adslab.get_forces())
#                 adslab.set_calculator(calc)
#                 db.write(adslab, id=id, key_value_pairs=data_to_db)
#     print('Done!')

def ml_relax_db(input_db, checkpoint_path, start_id, output_path='', interval=1000, traj_file=None, log_file='-', fmax=0.03, steps=300, trainer='equiformerv2_forces'):    
    output_traj = os.path.join(output_path, f'ml_inf_{start_id}_to_{stop_id}.traj')
    start_id = int(start_id)
    stop_id = start_id + interval
    if os.path.exists(output_traj):
        start_id = int(start_id) + len(Trajectory(output_traj)) - 1 

    query = f'id>={start_id},id<{stop_id}'
    with connect(input_db) as db:
        for row in db.select(query):
            original_id = row.id
            adslab = row.toatoms()
            data_to_db = row.key_value_pairs
            with Trajectory(output_traj, 'a') as traj:
                calc = OCPCalculator(checkpoint_path=checkpoint_path, trainer=trainer)   
                adslab.calc = calc
                opt_slab = BFGS(adslab, logfile=log_file, trajectory=traj_file)
                opt_slab.run(fmax=fmax, steps=steps)
                # output_db.write(adslab, id=original_id, key_value_pairs=data_to_db)
                traj.write(adslab)
    print('Done!')
