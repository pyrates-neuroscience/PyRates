from pyrates.utility.cluster_grid_search import *
# from pyrates.utility import plot_timeseries, grid_search


############################
# Global config parameters #
############################
config_name = "simple_nextgen_NMM_1"
config_file = f'/nobackup/spanien1/salomon/ClusterGridSearch/{config_name}.json'

circuit_template = "pyrates.examples.simple_nextgen_NMM.Net5"

dt = 1e-4
T = 2.0
inp = (2. + np.random.randn(int(T/dt), 1) * 1.0).tolist()

param_map = {'J_e': {'var': [('Op_e.0', 'J_e'), ('Op_i.0', 'J_e')],
                     'nodes': ['PC.0', 'IIN.0']},
             'J_i': {'var': [('Op_e.0', 'J_i'), ('Op_i.0', 'J_i')],
                     'nodes': ['PC.0', 'IIN.0']}
             }

inputs = {("PC", "Op_e.0", "inp"): inp}
outputs = {"r": ("PC", "Op_e.0", "r")}

# create_cgs_config(fp=config_file, circuit_template=circuit_template,
#                   param_map=param_map, dt=dt, simulation_time=T, inputs=inputs,
#                   outputs=outputs)

##################
# Parameter grid #
##################
param_grid = {'J_e': np.arange(8., 18., 2.), 'J_i': np.arange(2., 10., 2.)}
# param_grid = linearize_grid(param_grid, permute=True)
# param_grid = "/data/hu_salomon/Documents/ClusterGridSearch/CGS_TestDir/Grids/CGSTestGrid.csv"

#########################
# Cluster configuration #
#########################
host_config = {
    'hostnames': [
        'animals',
        # 'spanien',
        # 'carpenters',
        'osttimor'
        ],
    'host_env_cpu': "/data/u_salomon_software/anaconda3/envs/PyRates/bin/python",
    'host_env_gpu': "",
    'host_file': "/data/hu_salomon/PycharmProjects/PyRates/pyrates/utility/cluster_worker.py",
    'host_dir': ""
}

#######################
# CLUSTER GRID SEARCH #
#######################

# Optional: Directory to use as compute directory for current CGS instance.
# If none is specified, default compute directory is created at the config_file's location
compute_dir = f'/nobackup/spanien1/salomon/ClusterGridSearch/{config_name}'

print("Starting cluster grid search!")
start_cgs = time.time()

# Create ClusterGridSearch instance
cgs = ClusterGridSearch(config_file, compute_dir=compute_dir)

# Create compute cluster
clients = cgs.create_cluster(host_config)

# Compute grid inside the cluster. Can be called multiple times with different grids after a cluster is created
res_dir, grid_file = cgs.compute_grid(param_grid, num_params="dist_equal_add_mod", permute=True)
# cgs.compute_grid(param_grid, num_params="dist_equal_add_mod", permute=True)
# cgs.compute_grid(param_grid, num_params=10, permute=True)

# results = gather_cgs_results(res_dir, params=3)

elapsed_cgs = time.time() - start_cgs
print("Cluster grid search elapsed time: {0:.3f} seconds".format(elapsed_cgs))


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






