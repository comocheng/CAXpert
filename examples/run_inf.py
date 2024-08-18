import subprocess
from caxpert.src.utils.utils import add_fw

date_today = '2024-08-18'
command = f'python inf_val.py'
add_fw([command],'/global/cfs/cdirs/m4126/xuchao/qm_calcs/methanation_ni/fws/my_launchpad.yaml', date_today)
queue = '/global/cfs/cdirs/m4126/xuchao/caxpert/examples/fw_configs/array_qadapter.yaml'
worker = '/global/cfs/cdirs/m4126/xuchao/caxpert/examples/fw_configs/my_fworker.yaml'
cmd = ["qlaunch", "-q", queue, '-w', worker, 'singleshot']
subprocess.run(cmd, check=True)
