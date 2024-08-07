import time
import numpy as np

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
    Args:
        atoms (ase.Atoms): the structure to check
        tolerance (float): The tolerance to consider atoms in the top layer (default: 0.1 Å).
    """
    z_coords = atoms.positions[:, 2]
    top_layer_z = np.max(z_coords)
    top_ids = []
    for i in atoms:
        if i.z == top_layer_z:
            top_ids.append(i.index)
    if not any([atoms[i].z - top_layer_z > tolerance for i in top_ids]):
        return False
    else:
        return True

elements_place_holder = ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn']
