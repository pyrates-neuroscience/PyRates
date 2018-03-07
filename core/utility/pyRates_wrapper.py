"""Includes functions that wrap PyRates functionalities to use them with other tools
"""

from core.circuit import *
from core.utility import set_instance
import numpy as np

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


#####################
# wrapper functions #
#####################


def circuit_wrapper(circuit_type, circuit_params, simulation_params, target_var_idx=0, population_idx=None,
                    time_window=None):
    """Instantiates circuit and returns its simulated behavior.
    """

    # instantiate circuit
    if circuit_type == 'Circuit':
        circuit = Circuit(**circuit_params)
    else:
        try:
            circuit = set_instance(Circuit, circuit_type, **circuit_params)
        except AttributeError:
            try:
                circuit = set_instance(CircuitFromScratch, circuit_type, **circuit_params)
            except AttributeError:
                try:
                    circuit = set_instance(CircuitFromPopulations, circuit_type, **circuit_params)
                except AttributeError:
                    print('Could not find passed circuit type.')
                    raise

    # simulate circuit behavior over time
    circuit.run(**simulation_params)

    # check state variable extraction params
    if not population_idx:
        population_idx = range(circuit.n_populations)
    if not time_window:
        time_window = [0., simulation_params['simulation_time']]

    return circuit.get_population_states(target_var_idx, population_idx, time_window=time_window).flatten()


def jansenrit_wrapper(rand_vars, step_size=1e-3, max_synaptic_delay=0.5, input_mean=220, input_var=22,
                      simulation_time=2.0):
    """Uses circuit_wrapper to simulate JR circuit behavior for a specified excitatory synapse efficacy and 
    connectivity scaling."""

    synapse_params = [{'efficacy': rand_vars[1]}, dict()]
    circuit_params = {'connectivity_scaling': rand_vars[0], 'step_size': step_size,
                      'max_synaptic_delay': max_synaptic_delay, 'synapse_params': synapse_params}
    sim_steps = int(simulation_time / step_size)
    synaptic_input = np.zeros((sim_steps, 3, 2))
    synaptic_input[:, 0, 0] = input_var * np.random.randn(sim_steps) + input_mean
    simulation_params = {'simulation_time': simulation_time, 'synaptic_inputs': synaptic_input}

    return np.mean(circuit_wrapper('JansenRitCircuit', circuit_params, simulation_params,
                                   population_idx=[0], time_window=[simulation_time-1, simulation_time]))


###############
# GPC example #
###############

from pygpc.ni import run_reg_adaptive2
from pygpc.grid import norm
from matplotlib.pyplot import *

# parameter definition
######################

# random variables
gpc_vars = ['connectivity_scaling', 'He']
pdftype = ['norm', 'norm']
pdfshape = [[150., 3.25e-3], [10., 5e-4]]
limits = [[10., 1e-3], [290., 6e-3]]

# gpc params
eps = 1e-3
order_start = 2

# perform gpx analysis
######################

gpc_obj, gpc_output = run_reg_adaptive2(gpc_vars, pdftype, pdfshape, limits, jansenrit_wrapper,
                                        order_start=order_start, eps=eps)

# create target values to compare gpc output against
####################################################

params = gpc_obj.grid.coords
target_vals = np.zeros(params.shape[0])
for i in range(params.shape[0]):
    target_vals[i] = jansenrit_wrapper(params[i, :])

# estimate gpc predictions for above defined parameter values
#############################################################

params_normalized = norm(params, pdftype, pdfshape, limits)
gpc_coefs= gpc_obj.expand(gpc_output)
predicted_vals = gpc_obj.evaluate(gpc_coefs, params_normalized)

# plot results
##############

