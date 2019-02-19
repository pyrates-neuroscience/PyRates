# PyRates internal imports
from pyrates.utility.cluster_compute import *

# np.set_printoptions(precision=2)

#####################################
# Create ClusterGridSearch instance #
#####################################
nodes = [
        'animals',
        'spanien',
        'carpenters',
        'osttimor'
        ]

compute_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/CGS_nextgen_NMM_example"

cgs = ClusterGridSearch(nodes, compute_dir=compute_dir)

############################
# Global config parameters #
############################
config_file = f'{compute_dir}/Config/simple_nextgen_NMM.json'

circuit_template = "pyrates.examples.simple_nextgen_NMM.Net5"

dt = 1e-4
T = 2.0
inp = 2. + np.random.randn(int(T/dt), 1) * 1.0

param_map = {'J_e': {'var': [('Op_e.0', 'J_e'), ('Op_i.0', 'J_e')],
                     'nodes': ['PC.0', 'IIN.0']},
             'J_i': {'var': [('Op_e.0', 'J_i'), ('Op_i.0', 'J_i')],
                     'nodes': ['PC.0', 'IIN.0']}
             }

inputs = {("PC", "Op_e.0", "inp"): inp.tolist()}
outputs = {"r": ("PC", "Op_e.0", "r")}

create_cgs_config(fp=config_file, circuit_template=circuit_template,
                  param_map=param_map, dt=dt, simulation_time=T, inputs=inputs,
                  outputs=outputs)

#########################
# Run ClusterGridSearch #
#########################
worker_env = "/data/u_salomon_software/anaconda3/envs/PyRates/bin/python"
worker_file = "/data/hu_salomon/PycharmProjects/PyRates/pyrates/utility/cluster_worker.py"

# Parameter grid
################
param_grid = {'J_e': np.arange(1., 11., 1.), 'J_i': np.arange(1., 11., 1.)}

print("Starting cluster grid search!")
t0 = t.time()
res_dir, grid_file = cgs.run(config_file=config_file,
                             param_grid=param_grid,
                             permute=True,
                             chunk_size="dist_equal_add_mod",
                             #chunk_sie=50,
                             worker_env=worker_env,
                             worker_file=worker_file)
print(f'Cluster grid search elapsed time: {t.time()-t0:.3f} seconds')

##################
# GATHER RESULTS #
##################
# filter_ = {'J_e': np.arange(8., 12., 2.), 'J_i': np.arange(2., 8., 2.)}
# filter_grid = linearize_grid(filter_, permute=False)
#
result = gather_cgs_results(res_dir, num_header_params=3)

# results.to_csv("/nobackup/spanien1/salomon/ClusterGridSearch/CGS_nextgen_NMM_example/Results/test.csv", index=True)

# ###############
# # GRID SEARCH #
# ###############
#
# print("Starting grid search!")
# start_gs = time.time()
#
# results = grid_search(circuit_template=circuit_template,
#                       param_grid=param_grid, param_map=param_map,
#                       inputs=inputs, outputs=outputs,
#                       dt=dt, simulation_time=T, permute_grid=True)
#
# # Create result file
# results.to_csv(f'{compute_dir}/grid_search_nextgen_nmm_result.csv', index=True)
#
# elapsed_gs = time.time() - start_gs
# print("Cluster grid search elapsed time: {0:.3f} seconds".format(elapsed_gs))


########
# EVAL #
########


