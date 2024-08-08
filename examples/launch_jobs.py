from caxpert.src.utils.utils import add_fw
import os, subprocess

date_today = '2024-08-08'
job_num = 0
commands = []
for i in os.listdir('dft_relax'):
    if '.ipynb' in i:
        continue
    job_num += 1
    command = f'python /global/cfs/cdirs/m4126/xuchao/caxpert/examples/start_dfts.py  /global/cfs/cdirs/m4126/xuchao/caxpert/examples/dft_relax/{i}/init.traj'
    commands.append(command)
add_fw(commands,'/global/cfs/cdirs/m4126/xuchao/qm_calcs/methanation_ni/fws/my_launchpad.yaml', date_today)
queue = '/global/cfs/cdirs/m4126/xuchao/caxpert/examples/fw_configs/my_qadapter.yaml'
worker = '/global/cfs/cdirs/m4126/xuchao/caxpert/examples/fw_configs/my_fworker.yaml'
cmd = ["qlaunch", "-q", queue, '-w', worker, 'rapidfire',"--nlaunches", str(job_num)]
subprocess.run(cmd, check=True)
