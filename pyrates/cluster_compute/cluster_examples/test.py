# Internal imports
import time as t

# External imports
import scipy.signal as sp
import numpy as np


# PyRates internal imports
from pyrates.utility.grid_search import grid_search, linearize_grid

tau_temp = np.linspace(1.0, 3.0, 101)
alpha_temp = np.linspace(0.0, 0.2, 101)
tau, alpha = np.meshgrid(tau_temp, alpha_temp)
tau = tau.reshape(tau_temp.size*alpha_temp.size, 1).squeeze()
alpha = alpha.reshape(tau_temp.size*alpha_temp.size, 1).squeeze()
Cs = 20.0

ei_ratio = 1.0
io_ratio = 1.2

param_map = {'k_ee': {'var': [('Op_e.0', 'k_ee')],
                      'nodes': ['E.0']},
             'k_ei': {'var': [('Op_e.0', 'k_ei')],
                      'nodes': ['E.0']},
             'k_ii': {'var': [('Op_i.0', 'k_ii')],
                      'nodes': ['I.0']},
             'k_ie': {'var': [('Op_i.0', 'k_ie')],
                      'nodes': ['I.0']},
             'alpha': {'var': ['Op_e.0', 'alpha'],
                       'nodes': ['E.0']},
             'tau': {'var': ['Op_e.0', 'tau'],
                     'nodes': ['E.0']}
             }

dts = 1e-2

k_ee = [Cs]*(tau_temp.size*alpha_temp.size)
k_ei = [Cs / (ei_ratio * io_ratio)]*(tau_temp.size*alpha_temp.size)
k_ie = [Cs / io_ratio]*(tau_temp.size*alpha_temp.size)
k_ii = [Cs / ei_ratio]*(tau_temp.size*alpha_temp.size)

params = {'k_ee': k_ee, 'k_ei': k_ei, 'k_ie': k_ie, 'k_ii': k_ii, 'alpha': alpha, 'tau': tau}
params = linearize_grid(params, permute=False)

results = grid_search(circuit_template="/data/hu_salomon/PycharmProjects/PyRates/models/Montbrio/Montbrio.EI_Circuit",
                      param_grid=params,
                      param_map=param_map,
                      dt=5e-4,
                      simulation_time=150.0,
                      inputs={},
                      outputs={"r": ("E", "Op_e.0", "r")},
                      sampling_step_size=dts,
                      permute_grid=False)

