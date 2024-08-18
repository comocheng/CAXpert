from caxpert.src.tasks.make_db import MakeTrainingDB
import subprocess
import fairchem.core.common.tutorial_utils as utils
from caxpert.src.utils.utils import add_fw

def get_traj_paths(parent_p):
    return [f'{parent_p}/{i}/relax.traj' for i in os.listdir(parent_p)]
co_h_traj_paths = get_traj_paths('dft_relax')
h_traj_paths = get_traj_paths('dft_relax_h_only')
co_h_traj_paths.extend(h_traj_paths)
MakeTrainingDB(co_h_traj_paths, 'slabs.db', 'gas_ref.db').create_ase_database()
os.makedirs('training_data/datasets', exist_ok=True)
utils.train_test_val_split('training_data/ml_train.db', (0.8, 0.1, 0.1),('training_data/datasets/train.db','training_data/datasets/val.db', 'training_data/datasets/test.db'))

date_today = '2024-08-15'
command = f'time python /global/cfs/cdirs/m4126/xuchao/fairchem/main.py --mode train --config-yml ft/config.yml --checkpoint /global/cfs/cdirs/m4126/xuchao/ce_models/co_ni/ocp_train/fine_tuning/checkpoints/2024-06-22-17-21-20-co_ni_cov/best_checkpoint.pt --run-dir ft --identifier co_h_ni_cov --amp > ft/train.txt 2>&1'
add_fw([command],'/global/cfs/cdirs/m4126/xuchao/qm_calcs/methanation_ni/fws/my_launchpad.yaml', date_today)
queue = '/global/cfs/cdirs/m4126/xuchao/caxpert/examples/fw_configs/my_qadapter_gpu.yaml'
worker = '/global/cfs/cdirs/m4126/xuchao/caxpert/examples/fw_configs/my_fworker.yaml'
cmd = ["qlaunch", "-q", queue, '-w', worker, 'singleshot']
subprocess.run(cmd, check=True)
