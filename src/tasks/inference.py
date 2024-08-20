import os
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from functools import wraps
from ase.optimize import BFGS
from ase.db import connect
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
import numpy as np
from ase.io.trajectory import Trajectory
from caxpert.src.utils.utils import timeit
import plotly.express as px
import pandas as pd


def ml_validate(checkpoint_path, database_path, trainer='equiformerv2_forces', fig_path='parity_plot.png'):
    """
    Validate the ML model using the test set.
    checkpoint_path: str
        The path to the checkpoint file.
    database_path: str
        The path to the test database.
    trainer: str
        The trainer to pass to the OCPCalculator.
    fig_path: str
        The path to save the parity plot.
    """
    calc = OCPCalculator(checkpoint_path=checkpoint_path, trainer=trainer)
    db = connect(database_path)
    trajs = []
    for row in db.select():
        trajs.append(row.toatoms())
    traj_e_dfts = [traj.get_potential_energy() for traj in trajs]
    fmax_e_dfts = [np.max(np.linalg.norm(traj.get_forces(), axis=1)) for traj in trajs]
    traj_e_ocps = []
    fmax_e_ocps = []
    for traj in trajs:
        traj.calc = calc
        traj_e_ocps.append(traj.get_potential_energy())
        fmax_e_ocps.append(np.max(np.linalg.norm(traj.get_forces(), axis=1)))
    plt.figure(figsize=(6, 6))
    plt.scatter(traj_e_dfts, traj_e_ocps, color='b', marker='o', label='ML predictions')
    plt.plot([min(traj_e_dfts), max(traj_e_dfts)], [min(traj_e_ocps), max(traj_e_ocps)], color='r', linestyle='--')
    plt.xlabel('DFT')
    plt.ylabel('ML predictions')
    plt.title('Parity Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_path)
    return mean_squared_error(traj_e_dfts, traj_e_ocps, squared=True), mean_squared_error(fmax_e_dfts, fmax_e_ocps, squared=True)

@timeit
def ml_relax_db(input_db, checkpoint_path, start_id, output_path='', interval=1000, log_file='-', fmax=0.03, steps=300, trainer='equiformerv2_forces'):
    """
    Relax the structures in the database using the ML model.
    This function is designed to be used with SLURM job arrays.
    The structure database can be split into intervals and each interval can be relaxed in parallel.
    input_db: str
        The path to the database with the structures to relax.
    checkpoint_path: str
        The path to the checkpoint file.
    start_id: int
        The ID of the first structure to relax.
    output_path: str
        The path to save the relaxed structures.
    interval: int
        The number of structures to relax in each job.
    log_file: str
        The path to log the relax history.
    fmax: float
        The maximum force for the relaxation.
    steps: int
        The number of steps for the relaxation.
    trainer: str
        The trainer to pass to the OCPCalculator.
    """
    start_id = int(start_id)
    stop_id = start_id + interval
    output_traj = os.path.join(output_path, f'ml_inf_{start_id}_to_{stop_id}.traj')
    if os.path.exists(output_traj):
        start_id = int(start_id) + len(Trajectory(output_traj)) - 1 

    query = f'id>={start_id},id<{stop_id}'
    with connect(input_db) as db:
        for row in db.select(query):
            adslab = row.toatoms()
            with Trajectory(output_traj, 'a') as traj:
                calc = OCPCalculator(checkpoint_path=checkpoint_path, trainer=trainer)   
                adslab.calc = calc
                opt_slab = BFGS(adslab, logfile=log_file)
                opt_slab.run(fmax=fmax, steps=steps)
                traj.write(adslab)
    print('Done!')

def mk_inf_db(input_db, trajs_path, output_db):
    """
    This function writes the ML relaxed structures to a database.
    It reads the extra key_value_pairs from the original input_db
    and writes them to the output_db.
    input_db: str
        The path to the original database.
    trajs_path: str or list
        The path to the directory with the relaxed structures written in .traj format.
        if a str is passed, the files must be written as 'ml_inf_{START ID}_to_{STOP ID}.traj'.
        if a list is passed, the files must be in the order to match the structures' order in the input_db.
    output_db: str
        The path to the output database.
    """
    if type(trajs_path) == list:
        trajs = trajs_path
    else:
        traj_ps = [i for i in os.listdir(trajs_path) if 'ml_inf' in i]
        traj_ps = sorted(traj_ps, key=lambda x:int(x.split('_')[2]))
        trajs = [os.path.join(trajs_path, t) for t in traj_ps]
    if not os.path.exists(input_db):
        raise FileNotFoundError(f'{input_db} does not exist!')
    str_id = 0
    with connect(input_db) as db, connect(output_db) as odb:
        for t in trajs:
            atoms = Trajectory(t)
            for a in atoms:
                str_id += 1
                key_value_pairs = db.get(id=str_id).key_value_pairs
                odb.write(a, key_value_pairs=key_value_pairs)
    print('Done!')

class MLInfDataProcess:
    def __init__(self, input_db, adsorbate_names, metal_atom, unit_cell_metal_atom_num):
        """
        This class is designed to process the data for ML inference.
        input_db: str
            The path to the database.
        adsorbate_names: list
            The names of the adsorbates.
        metal_atom: str
            The name of the metal atom.
        unit_cell_metal_atom_num: int
            The number of metal atoms in the unit cell.
        """
        self.input_db = input_db
        self.adsorbate_names = adsorbate_names
        self.metal_atom = metal_atom
        self.unit_cell_metal_atom_num = unit_cell_metal_atom_num
    def plot_energy(self, output_fig=None):
        """
        Plot the energy of the structures in the database. This function is not designed to work with alloys.
        output_fig: str
            The path to save the plot.
        """
        if len(self.adsorbate_names) > 2:
            raise ValueError('System with adsorbate number more than 2 is not supported now!')
        energies = []
        coverages = []
        with connect(self.input_db) as db:
            for row in db.select():
                energy = row.toatoms().get_potential_energy()
                covs = [row.key_value_pairs[n] for n in self.adsorbate_names]
                sites = row.toatoms().get_chemical_symbols().count(self.metal_atom)/self.unit_cell_metal_atom_num
                energies.append(energy/sites)
                coverages.append(covs)
        if len(coverages[0]) == 1:
            coverages = [i for i in coverages]
            plt.scatter(coverages, energies)
        elif len(coverages[0]) == 2:
            x = np.array([i[0] for i in coverages])
            y = np.array([i[1] for i in coverages])
            data = zip(x,y,np.array(energies))
            df = pd.DataFrame(data, columns=[self.adsorbate_names[0], self.adsorbate_names[1], 'Adsorption Energy(eV)'])
            fig = px.scatter_3d(df, x=self.adsorbate_names[0], y=self.adsorbate_names[1], z='Adsorption Energy(eV)')
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                scene=dict(
                aspectratio=dict(x=2, y=2, z=1)  # Make x and y axes appear longer
            ),
        )
            fig.update_traces(marker=dict(size=2), selector=dict(mode='markers'))
            if output_fig is not None:
                fig.write_html(output_fig)
            else:
                fig.show()
    def get_convex_hull(self):
        structs = dict()
        with connect(self.input_db) as db:
            for row in db.select():
                covs = tuple([row.key_value_pairs[n] for n in self.adsorbate_names])
                sites = row.toatoms().get_chemical_symbols().count(self.metal_atom)/self.unit_cell_metal_atom_num
                energy = row.toatoms().get_potential_energy() / sites
                if not covs in structs:
                    structs[covs] = [energy]
                else:
                    structs[covs].append(energy)
        hulls = dict()
        for k in structs.keys():
            hulls[k] = min(structs[k])
        return hulls
    # def get_structures_to_validate(self, gs_ids, dft_list_csv):
    #     old_gs_ids = pd.read_csv(dft_list_csv).iloc[:,:].to_numpy()
    #     dft_1d = old_gs_ids.reshape(old_gs_ids.shape[0] * old_gs_ids.shape[1])
    #     mask = ~np.isnan(dft_1d)
    #     old_gs_ids = dft_1d[mask]
    #     next_round = []
    #     for i in gs_ids:
    #         if i not in list(old_gs_ids) and i not in [0, 1]:
    #             next_round.append(i)
    #     return next_round
