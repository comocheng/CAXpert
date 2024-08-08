import time, yaml
import numpy as np
from fireworks import Firework, ScriptTask, LaunchPad

elements_place_holder = ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn']

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result
    return wrapper

def check_reconstruction(atoms, tolerance=0.1):
    """
    This function checks if the surface is reconstructed
    atoms: ase.Atoms 
        The structure to check
    tolerance: float 
        The tolerance to consider atoms in the top layer (default: 0.1 Å).
    """
    z_coords = [atom.z for atom in atoms if atom.tag != 2]
    if not z_coords:
        return False
    top_layer_z = np.max(z_coords)
    top_ids = []
    for i in atoms:
        if i.z == top_layer_z:
            top_ids.append(i.index)
    if not any([atoms[i].z - top_layer_z > tolerance for i in top_ids]):
        return False
    else:
        return True

def is_generator_empty(generator):
    """
    Check if a generator is empty.
    generator: The generator to check.
    """
    try:
        first_item = next(generator)
        return False
    except StopIteration:
        return True

def add_fw(commands, lpad_config, reset_date):
    """
    Add a firework to the launchpad.
    command: list
        The commands to run the scripts.
    lpad_config: str
        The path to the launchpad configuration yaml file.
    reset_date: str
        The date to reset the launchpad. Formatted in "YEAR-MM-DD", example: '2024-08-04'
    """
    with open(lpad_config) as f:
        config = yaml.safe_load(f)
    launchpad = LaunchPad(host=config['host'], port=config['port'], name=config['name'], username=config['username'], password=config['password'])
    launchpad.reset(reset_date, require_password=True)
    for command in commands:
        firetask = ScriptTask.from_str(command)
        firework = Firework(firetask)   
        launchpad.add_wf(firework)
