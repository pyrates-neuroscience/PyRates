# External imports
import matplotlib.pyplot as plt

# PyRates internal imports
from pyrates.utility import plot_timeseries
from pyrates.cluster_compute.cluster_compute import *

# np.set_printoptions(precision=2)

#####################################
# Create ClusterGridSearch instance #
#####################################
nodes = [
        # 'animals',
        'spanien',
        'carpenters',
        'osttimor',
        ]

compute_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/CGS_nextgen_NMM_example_2"

# Create compute directory and connect to nodes
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
worker_env = "/data/u_salomon_software/anaconda3/envs/PyRates/bin/python3"
worker_file = "/data/hu_salomon/PycharmProjects/PyRates/pyrates/cluster_compute/cluster_workers/gridsearch_worker.py"

# Parameter grid
################
params = {'J_e': np.arange(1., 11., 1.), 'J_i': np.arange(1., 11., 1.)}

print("Starting cluster grid search!")
t0 = t.time()
res_dir, grid_file = cgs.run(thread_kwargs={
                                "worker_env": worker_env,
                                "worker_file": worker_file
                                 },
                             config_file=config_file,
                             chunk_size="dist_equal_add_mod",
                             # chunk_size=50,
                             param_grid=params,
                             permute=True)

print(f'Cluster grid search elapsed time: {t.time()-t0:.3f} seconds')

##################
# GATHER RESULTS #
##################

results = gather_cgs_results(res_dir)


########
# EVAL #
########
# print(results.loc[:, (params['J_e'][0], params['J_i'][0])])
# print(results.columns)
# # plotting
# for j_e in params['J_e']:
#     for j_i in params['J_i']:
#         print(j_e, j_i)
#         ax = plot_timeseries(results[str(j_e)][str(j_i)], title=f"J_e={j_e}, J_i={j_i}")
# plt.show()
