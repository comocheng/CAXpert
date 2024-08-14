from caxpert.src.tasks.make_db import MakeTrainingDB
import os
from ase.db import connect
import fairchem.core.common.tutorial_utils as utils

def get_traj_paths(parent_p):
    return [f'{parent_p}/{i}/relax.traj' for i in os.listdir(parent_p)]
co_h_traj_paths = get_traj_paths('dft_relax')
h_traj_paths = get_traj_paths('dft_relax_h_only')
co_h_traj_paths.extend(h_traj_paths)
MakeTrainingDB(co_h_traj_paths, 'slabs.db', 'gas_ref.db').create_ase_database()
os.makedirs('training_data/datasets', exist_ok=True)
utils.train_test_val_split('training_data/ml_train.db', (0.8, 0.1, 0.1),('training_data/datasets/train.db','training_data/datasets/val.db', 'training_data/datasets/test.db'))
