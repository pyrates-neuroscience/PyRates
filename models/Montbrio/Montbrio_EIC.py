# Internal imports
import time as t

# External imports
import pandas as pd

# PyRates internal imports
from pyrates.utility import grid_search
from pyrates.cluster_compute.montbrio_eval_funcs import *
from pyrates.cluster_compute.cluster_compute import read_cgs_results

if __name__ == "__main__":

    param_map = {'k_e': {'var': [('Op_e.0', 'k_ee'), ('Op_i.0', 'k_ie')],
                         'nodes': ['E.0', 'I.0']},
                 'k_i': {'var': [('Op_e.0', 'k_ei'), ('Op_i.0', 'k_ii')],
                         'nodes': ['E.0', 'I.0']}
                 }

    # params = {'k_e': np.linspace(10., 30., 3), 'k_i': np.linspace(10., 30., 3)}
    # params = {'k_e': np.linspace(10., 15., 11), 'k_i': np.linspace(25., 30., 11)}
    # params = {'k_e': np.linspace(15., 20., 21), 'k_i': np.linspace(15., 20., 21)}
    params = {'k_e': np.linspace(25., 30., 3), 'k_i': np.linspace(10., 25., 3)}

    T = 2.0

    ###############
    # Grid Search #
    ###############

    t0_run = t.time()
    results = grid_search(circuit_template="/data/hu_salomon/PycharmProjects/PyRates/models/Montbrio/Montbrio.EI_Circuit",
                          param_grid=params,
                          param_map=param_map,
                          dt=1e-5,
                          simulation_time=T,
                          inputs={},
                          outputs={"r": ("E", "Op_e.0", "r")},
                          sampling_step_size=1e-3,
                          permute_grid=False)

    t_run = t.time()-t0_run
    print(f'Grid search elapsed time: {t_run} seconds')
    print(results)

    t0_save = t.time()
    results.to_hdf('/nobackup/spanien1/salomon/EIC_spike_detection/GS_EIC_spike_rate_tschad.h5', key='Data')
    t_save = t.time()-t0_save
    print(f'Saving results elapsed time: {t_save} seconds')

    #############
    # Load Data #
    #############
    # result_file = "/nobackup/spanien1/salomon/ClusterGridSearch/Montbrio/Computation_1/Results/DefaultGrid_0/CGS_result_DefaultGrid_0.h5"
    # results = read_cgs_results(result_file, key='Data')
    # params = {'k_e': np.linspace(10., 30., 21), 'k_i': np.linspace(10., 30., 21)}
    # T = 10.0
    # dt = params['k_e'][1] - params['k_e'][0]

    ########
    # Plot #
    ########
    plot_avrg_peak_dist(results, parameters=params, simulation_time=T, dt=dt, tick_size=5)
