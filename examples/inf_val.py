from caxpert.src.tasks.inference import ml_validate, ml_relax, ml_relax_db
import multiprocessing, os
from ase.db import connect
from functools import partial

# checkpoint_path = 'ft/checkpoints/2024-08-15-13-01-04-co_h_ni_cov/best_checkpoint.pt'
# rmse_e, rmse_f = ml_validate(checkpoint_path, 'training_data/datasets/test.db', trainer='equiformerv2_forces', fig_path='ft/parity_plot.png')
start_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
ml_relax_db('init_structures.db', checkpoint_path='ft/checkpoints/2024-08-15-13-01-04-co_h_ni_cov/best_checkpoint.pt', output_path='ft', start_id=start_id, fmax=0.01, steps=300, trainer='equiformerv2_forces')

# This code use the database to run the inference in an array of SLURM jobs
# data = []
# strcuture_db = 'init_structures.db'
# struct_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
# ml_relax_db(struct_id, 'init_structures.db', checkpoint_path='ft/checkpoints/2024-08-15-13-01-04-co_h_ni_cov/best_checkpoint.pt', db_path='ft/ml_inf.db', fmax=0.01, steps=300, trainer='equiformerv2_forces')


# This code use multiprocessing to run the inference, but did not work well
# if rmse_e<0.01 and rmse_f < 0.01:
#     lock = multiprocessing.Manager().Lock()
#     p = partial(ml_relax, checkpoint_path=checkpoint_path, db_path='ft/ml_inf.db', lock=lock, traj_file=None, log_file='-', fmax=0.01, steps=300, trainer='equiformerv2_forces')
#     data = []
#     strcuture_db = 'init_structures.db'
#     with connect(strcuture_db) as db:
#         for row in db.select():
#             adslab = row.toatoms()
#             data.append((adslab, row.key_value_pairs))
#     cpus = multiprocessing.cpu_count()
#     with multiprocessing.Pool(cpus) as pool:
#         pool.map(p, data)
# else:
#     raise ValueError(f'The model is not accurate enough to make inference. The RMSE of energy is {rmse_e}, and the RMSE of the max force is {rmse_f}')
