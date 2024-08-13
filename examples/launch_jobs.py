from caxpert.src.utils.utils import add_fw
import os, subprocess

def add_firetasks(date_today, traj_paths):
    job_num = 0
    commands = []
    for d in traj_paths:
        for i in os.listdir(d):
            if '.ipynb' in i:
                continue
            job_num += 1
            command = f'python /global/cfs/cdirs/m4126/xuchao/caxpert/examples/start_dfts.py  /global/cfs/cdirs/m4126/xuchao/caxpert/examples/{d}/{i}/init.traj 0.05'
            commands.append(command)
    add_fw(commands,'/global/cfs/cdirs/m4126/xuchao/qm_calcs/methanation_ni/fws/my_launchpad.yaml', date_today)
    return job_num

date_today = '2024-08-12'
traj_paths = ['slabs', 'dft_relax', 'dft_relax_h_only']
job_num = add_firetasks(date_today, traj_paths)
queue = '/global/cfs/cdirs/m4126/xuchao/caxpert/examples/fw_configs/my_qadapter_gpu.yaml'
worker = '/global/cfs/cdirs/m4126/xuchao/caxpert/examples/fw_configs/my_fworker.yaml'
cmd = ["qlaunch", "-q", queue, '-w', worker, 'rapidfire',"--nlaunches", str(job_num)]
subprocess.run(cmd, check=True)
