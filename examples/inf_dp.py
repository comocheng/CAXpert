from caxpert.src.tasks.inference import mk_inf_db, plot_energy

# mk_inf_db('init_structures.db','ft/ml_inf', 'ft/ml_inf.db')
plot_energy('ft/ml_inf.db', ['co', 'h'], 'Ni', 4, output_fig='ft/ml_inf.html')
