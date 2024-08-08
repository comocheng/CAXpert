import os, sys
from ase.calculators.espresso import Espresso
from caxpert.src.tasks.run_dft import CalculateEnergy

# Set up the Calculator with Quantum Espresso
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
        'electron_maxstep': 200,
        'mixing_mode': 'local-TF',
        'mixing_beta': 0.5,
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

def run_dft(file_path, fmax=0.03):
    dir_ = os.path.dirname(file_path)
    qe_calc = Espresso(
                command=command,
                pseudopotentials=pseudopotentials,
                tstress=True,
                tprnfor=True,
                kpts=kpts,
                input_data=espresso_settings,
                disk_io='none',
                directory=dir_
                )
    CalculateEnergy(file_path, qe_calc, restart=False, fmax=fmax).calculate_energy()

if __name__ == '__main__':
    if len(sys.argv)==2:
        f = sys.argv[1]
        run_dft(f)
    if len(sys.argv)>2:
        f = sys.argv[1]
        fmax = float(sys.argv[2])
        run_dft(f, fmax)
    else:
        print('Usage: python run_dft.py file_path [fmax]')
        sys.exit(1)
