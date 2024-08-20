from caxpert.src.tasks.inference import mk_inf_db, MLInfDataProcess

# mk_inf_db('init_structures.db','ft/ml_inf', 'ft/ml_inf.db')
mp = MLInfDataProcess('ft/ml_inf.db', ['co', 'h'], 'Ni', 4)
mp.plot_energy(output_fig='ft/ml_inf.html')
