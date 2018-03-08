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

    # define parameters to call circuit_wrapper with
    ################################################

    # circuit parameters
    synapse_params = [{'efficacy': rand_vars[1]}, dict()]
    circuit_params = {'connectivity_scaling': rand_vars[0], 'step_size': step_size,
                      'max_synaptic_delay': max_synaptic_delay, 'synapse_params': synapse_params}

    # simulation parameters
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

# extract parameter samples used for training the gpc
params = gpc_obj.grid.coords

# create new test parameter values within the range of the training data
n_samples = 10
params_test = np.zeros((n_samples, 2))
params_test[:, 0] = np.linspace(np.min(params[:, 0]), np.max(params[:, 0]), n_samples)
params_test[:, 1] = np.linspace(np.min(params[:, 1]), np.max(params[:, 1]), n_samples)

# loop over each parameter combination and simulate model behavior (target data)
target_vals = np.zeros((n_samples, n_samples))
for i, c in enumerate(params_test[:, 0]):
    for j, e in enumerate(params_test[:, 1]):
        target_vals[i, j] = jansenrit_wrapper([c, e])

# estimate gpc predictions for above defined parameter values
#############################################################

# normalize the parameter values to [-1, 1]
params_normalized = norm(params_test, pdftype, pdfshape, limits)

# extract the fitted coefficients from gpc object
gpc_coefs= gpc_obj.expand(gpc_output)

# predict model behavior given the fitted coefficients and normalized parameter values
predicted_vals = np.zeros((n_samples, n_samples))
for i, c in enumerate(params_normalized[:, 0]):
    for j, e in enumerate(params_normalized[:, 1]):
        predicted_vals[i, j] = gpc_obj.evaluate(gpc_coefs, np.array([[c, e]]))

# plot results
##############

fig, axes = subplots(1, 4, figsize=(15, 6), gridspec_kw={'width_ratios': [5, 1, 5, 1]})

# plot target data
im1 = axes[0].matshow(target_vals)
axes[0].set_title('Goal Function')
axes[0].set_xlabel('Connectivity Scaling')
axes[0].set_ylabel('Excitatory Synaptic Efficacy')

# set axis ticks and colorbar
x_tick_labels = list()
y_tick_labels = list()
x_ticks = axes[0].get_xticks()
y_ticks = axes[0].get_yticks()
for x, y in zip(x_ticks, y_ticks):
    if (x < 0) or (x >= n_samples):
        x_tick_labels.append('')
        y_tick_labels.append('')
    else:
        x_tick_labels.append(round(params_test[int(x), 0], 1))
        y_tick_labels.append(round(params_test[int(y), 1], 4))
axes[0].set_xticklabels(x_tick_labels)
axes[0].set_yticklabels(y_tick_labels)
fig.colorbar(im1, cax=axes[1], orientation='vertical')

# plot gpc predictions
im2 = axes[2].matshow(predicted_vals)
axes[2].set_title('Goal Function')
axes[2].set_xlabel('Connectivity Scaling')
axes[2].set_ylabel('Excitatory Synaptic Efficacy')

# set axis ticks and colorbar
x_tick_labels = list()
y_tick_labels = list()
x_ticks = axes[0].get_xticks()
y_ticks = axes[0].get_yticks()
for x, y in zip(x_ticks, y_ticks):
    if (x < 0) or (x >= n_samples):
        x_tick_labels.append('')
        y_tick_labels.append('')
    else:
        x_tick_labels.append(round(params_test[int(x), 0], 1))
        y_tick_labels.append(round(params_test[int(y), 1], 4))
axes[2].set_xticklabels(x_tick_labels)
axes[2].set_yticklabels(y_tick_labels)
fig.colorbar(im2, cax=axes[3], orientation='vertical')

tight_layout()
fig.show()
