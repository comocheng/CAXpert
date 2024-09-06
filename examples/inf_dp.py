from caxpert.src.tasks.inference import mk_inf_db, MLInfDataProcess
from ase.calculators.espresso import Espresso
from caxpert.src.tasks.run_dft import ml_val
import os
from caxpert.src.tasks.gen_str import ml_val_db_to_trajs

mk_inf_db('init_structures.db','ft/ml_inf', 'ft/ml_inf.db')
mp = MLInfDataProcess('ft/ml_inf.db', ['co', 'h'], 'Ni', 4)
mp.plot_energy(output_fig='ft/ml_inf.html')
chs = mp.get_convex_hull()

# write your own code to choose the structures you want to use for validation and retrain the model
h_only = mp.get_structures_to_validate([[0,0], [0.1,1]],10, cov_must_have=[(float(0), float(1))])
co_only = mp.get_structures_to_validate([[0.1,1], [0,0]],10, cov_must_have=[(float(1), float(0))])
h_co = mp.get_structures_to_validate([[0.3,1], [0.1,1]],10)
ids_to_val = []
for strs in [h_only, co_only, h_co]:
# for strs in [h_co]:
    for v in strs.values():
        ids_to_val.append(v[1])

calculated_ids = [int(i) for i in os.listdir('dft_relax')]
calculated_ids.extend([int(i) for i in os.listdir('dft_relax_h_only')])
ids_to_val = [i for i in ids_to_val if i not in calculated_ids]

espresso_settings = {
    'control': {
        'verbosity': 'high',
        'calculation': 'scf',
        'pseudo_dir': '/global/homes/x/xuchao/espresso/pseudo',
        'disk_io': 'none'
    },
    'system': {
        'input_dft': 'RPBE',
        'occupations': 'smearing',
        'smearing': 'mv',
        'degauss': 0.01,
        'ecutwfc': 40,
        'nspin': 2,
    },
    'electrons': {
        'electron_maxstep': 100,
        'mixing_mode': 'local-TF',
        'mixing_beta': 0.2,
        'diagonalization': 'cg',
    },
}
command = "srun pw.x -npool 1 -ndiag 1 -input espresso.pwi > espresso.pwo"
pseudopotentials = {
                        'Ni': 'Ni_ONCV_PBE-1.2.upf',
                        'C': 'C_ONCV_PBE-1.2.upf',
                        'O': 'O_ONCV_PBE-1.2.upf',
                        'H': 'H_ONCV_PBE-1.2.upf',
                        }
kpts=(5, 5, 1)

qe_calc = Espresso(
                command=command,
                pseudopotentials=pseudopotentials,
                tstress=True,
                tprnfor=True,
                kpts=kpts,
                input_data=espresso_settings,
                disk_io='none',
                )

ml_val(ids_to_val, 'ft/ml_inf.db', qe_calc, 'ft/ml_inf_dft_val.db')
# ml_val([], 'ft/ml_inf.db', qe_calc, 'ft/ml_inf_dft_val.db', restart=True)

ml_val_db_to_trajs('ft/ml_inf_dft_val.db')
