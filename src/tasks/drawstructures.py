import random
import pandas as pd
from ase.io.trajectory import Trajectory
from ase.db import connect
import os

def cov_draw(db_path, output_csv):
    """
    This function randomly selects a structure from each coverage group and 
    stores the structure's index in a csv file.

    Args:
        db_path (str): the directory to the ASE database where the structures are stored.
        output_csv (str): the output csv file to store the ids of randomly selected structures.

    Returns:
        dict: a dictionary contain selected structures
        The dictionary contains the following keys:
            - "coverage" (float): The strcuture's index with the corresponding coverage.
    """
    ce_sys = {}
    db = connect(db_path)
    for index in range(len(db)):
        traj = db.get(sid=index).toatoms()
        site_num = sum(1 for atom in traj if atom.symbol == 'Ni') / 4
        ads_num = sum(1 for atom in traj if atom.symbol == 'C')
        cov = ads_num / site_num
        try:
            ce_sys[cov].append(int(index))
        except KeyError:
            ce_sys[cov] = [int(index)]

    random_samples = dict()
    covs = sorted([i for i in ce_sys.keys()])
    for k in covs:
        if k != 0:
            random_samples[k] = random.sample(ce_sys[k],1)
    
    df = pd.DataFrame(random_samples)
    if os.path.exists(output_csv):
        dfout = pd.read_csv(output_csv)
        df_new = dfout.append(df, ignore_index=True)
        df_new.to_csv(output_csv, index=False)
    else:
        df.to_csv(output_csv, index=False)
    return random_samples
