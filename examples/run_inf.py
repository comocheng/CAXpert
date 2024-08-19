import subprocess
from caxpert.src.utils.utils import add_fw
import yaml

def get_job_num(yml_path):
    with open(yml_path, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    array_str = data['array']
    if "-" in array_str:
        job_num = int(array_str.split('-')[1]) - int(array_str.split('-')[0]) + 1
    elif ":" in array_str and "," in array_str:
        job_num = 0
        start = int(array_str.split(",")[0])
        stop = int(array_str.split(",")[1].split(":")[0]) + 1
        interval = int(array_str.split(",")[1].split(":")[1])
        for i in range(start, stop, interval):
            job_num += 1
    elif "," in array_str and ":" not in array_str:
        job_num = len(array_str.split(","))
    return job_num

date_today = '2024-08-19'
command = f'python inf_val.py'
job_number = get_job_num('/global/cfs/cdirs/m4126/xuchao/caxpert/examples/fw_configs/array_qadapter.yaml')
add_fw([command]*job_number,'/global/cfs/cdirs/m4126/xuchao/qm_calcs/methanation_ni/fws/my_launchpad.yaml', date_today)
queue = '/global/cfs/cdirs/m4126/xuchao/caxpert/examples/fw_configs/array_qadapter.yaml'
worker = '/global/cfs/cdirs/m4126/xuchao/caxpert/examples/fw_configs/my_fworker.yaml'
cmd = ["qlaunch", "-q", queue, '-w', worker, 'singleshot']
subprocess.run(cmd, check=True)
