import multiprocessing, time
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from functools import wraps
from ase.optimize import BFGS
from ase.db import connect

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
def ml_relax(adslab, checkpoint_path, db_path, data_to_db=None, traj_file=None, log_file='-', fmax=0.03, steps=300, trainer='equiformerv2_forces'):
    calc = OCPCalculator(checkpoint_path=checkpoint_path, trainer=trainer)   
    adslab.calc = calc
    opt_slab = BFGS(adslab)
    opt_slab.run(fmax=fmax, steps=steps, logfile=log_file, trajectory=traj_file)
    with connect(db_path) as db:
        id = db.reserve(adslab)
        if id is not None:
            if data_to_db is not None:
                data_to_db(adslab, id=id, key_value_pairs=data_to_db)
            db.write(adslab, id=id)
    print('Done!')
