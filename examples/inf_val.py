from caxpert.src.tasks.inference import ml_validate, ml_relax_db
import os


checkpoint_path = 'ft/checkpoints/2024-08-15-13-01-04-co_h_ni_cov/best_checkpoint.pt'
rmse_e, rmse_f = ml_validate(checkpoint_path, 'training_data/datasets/test.db', trainer='equiformerv2_forces', fig_path='ft/parity_plot.png')
start_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
ml_relax_db('init_structures.db', checkpoint_path='ft/checkpoints/2024-08-15-13-01-04-co_h_ni_cov/best_checkpoint.pt', output_path='ft', start_id=start_id, fmax=0.01, steps=300, trainer='equiformerv2_forces')
