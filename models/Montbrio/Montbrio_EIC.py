# Internal imports
import time as t

# External imports

# PyRates internal imports
from pyrates.utility import grid_search
from pyrates.cluster_compute.montbrio_eval_funcs import *

if __name__ == "__main__":

    param_map = {'k_e': {'var': [('Op_e.0', 'k_ee'), ('Op_i.0', 'k_ie')],
                         'nodes': ['E.0', 'I.0']},
                 'k_i': {'var': [('Op_e.0', 'k_ei'), ('Op_i.0', 'k_ii')],
                         'nodes': ['E.0', 'I.0']}
                 }

    params = {'k_e': np.linspace(20., 30., 11), 'k_i': np.linspace(10., 20., 11)}

    T = 2.0
    dt = 1e-5
    ###############
    # Grid Search #
    ###############

    t0_run = t.time()
    results = grid_search(circuit_template="/data/hu_salomon/PycharmProjects/PyRates/models/Montbrio/Montbrio.EI_Circuit",
                          param_grid=params,
                          param_map=param_map,
                          dt=dt,
                          simulation_time=T,
                          inputs={},
                          outputs={"r": ("E", "Op_e.0", "r"),
                                   "v": ("E", "Op_e.0", "v")},
                          sampling_step_size=1e-3,
                          permute_grid=True)

    ########
    # Plot #
    ########
    plot_avrg_peak_dist(results, parameters=params, simulation_time=T, dt=dt, tick_size=5)
