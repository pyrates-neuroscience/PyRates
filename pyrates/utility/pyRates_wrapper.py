"""Includes functions that wrap PyRates functionalities to use them with other tools
"""

# external packages

# pyrates internal imports
from pyrates.circuit import *
from pyrates.utility import set_instance

# meta infos
__author__ = "Richard Gast"
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


def jansenrit_wrapper(rand_vars, step_size=1e-3, max_synaptic_delay=0.5, input_mean=220., input_var=22.,
                      simulation_time=2.0, cutoff_time=1., seed=0):
    """Uses circuit_wrapper to simulate JR circuit behavior for a specified excitatory synapse efficacy and 
    connectivity scaling."""

    # define parameters to call circuit_wrapper with
    ################################################

    # circuit parameters
    synapse_params = [{'efficacy': rand_vars[1]}, dict()]
    circuit_params = {'connectivity_scaling': rand_vars[0], 'step_size': step_size,
                      'max_synaptic_delay': max_synaptic_delay, 'synapse_params': synapse_params}

    # simulation parameters
    np.random.seed(seed)
    sim_steps = int(simulation_time / step_size)
    synaptic_input = np.zeros((sim_steps, 3, 2))
    synaptic_input[:, 0, 0] = input_var * np.random.randn(sim_steps) + input_mean
    simulation_params = {'simulation_time': simulation_time, 'synaptic_inputs': synaptic_input}

    return np.mean(circuit_wrapper('JansenRitCircuit', circuit_params, simulation_params,
                                   population_idx=[0], time_window=[simulation_time-cutoff_time, simulation_time]))


###############
# GPC example #
###############

import pygpc
from matplotlib.pyplot import *

# parameter definition
######################

# transfer function parameters
simulation_time = 6.
cutoff_time = 3.
step_size = 1e-3
input_mean = 220.
input_var = 22.

# random variables
gpc_vars = ['connectivity_scaling', 'He']
pdftype = ['beta', 'beta']
pdfshape = [[1., 1.], [1., 1.]]
limits = [[80., 1e-3], [180., 6e-3]]

# gpc params
eps = 1e-3
order = [3, 5, 7, 9, 11, 13]
interaction_order = 2

# create target values to compare gpc output against
####################################################

# create new test parameter values within the range of the training data
n_samples = 25
params_test = np.zeros((n_samples, 2))
params_test[:, 0] = np.linspace(limits[0][0], limits[1][0], n_samples)
params_test[:, 1] = np.linspace(limits[0][1], limits[1][1], n_samples)

# loop over each parameter combination and simulate model behavior (target data)
target_vals = np.zeros((n_samples, n_samples))
for i, c in enumerate(params_test[:, 0]):
    for j, e in enumerate(params_test[:, 1]):
        target_vals[i, j] = jansenrit_wrapper(rand_vars=[c, e], simulation_time=simulation_time,
                                              cutoff_time=cutoff_time, step_size=step_size, input_mean=input_mean,
                                              input_var=input_var)

# normalize the parameter values to [-1, 1] for gpc
params_normalized = pygpc.grid.norm(params_test, pdftype, pdfshape, limits)

# perform gpc analysis for each order
#####################################

predictions = list()
for o in order:

    # create gpc object
    N_coeffs = pygpc.calc_Nc_sparse(p_d=[o, o], p_g=o, p_i=interaction_order, dim=2)
    grid = pygpc.randomgrid(pdftype, pdfshape, limits, np.ceil(N_coeffs*3.), seed=None)
    gpc_obj = pygpc.reg(pdftype, pdfshape, limits, [o, o], o, interaction_order, grid, random_vars=gpc_vars)

    # create training values for gpc to fit
    training_vals = list()
    params = gpc_obj.grid.coords
    for p in params:
        training_vals.append([jansenrit_wrapper(rand_vars=p, step_size=step_size, simulation_time=simulation_time,
                                                cutoff_time=cutoff_time, input_mean=input_mean, input_var=input_var)])

    # fit gpc to training data
    gpc_coefs = gpc_obj.expand(np.array(training_vals))

    # predict model behavior given the fitted coefficients and normalized parameter values
    predicted_vals = np.zeros((n_samples, n_samples))
    for i, c in enumerate(params_normalized[:, 0]):
        for j, e in enumerate(params_normalized[:, 1]):
            predicted_vals[i, j] = gpc_obj.evaluate(gpc_coefs, np.array([[c, e]]))

    predictions.append(predicted_vals)

# plot results
##############

# choose synapse efficacies for which to plot membrane potentials over connection strengths
efficacies_to_plot = [5, 12, 20]

# plot for each gpx object
for i, predicted_vals in enumerate(predictions):

    fig, axes = subplots(1, 3, figsize=(18, 6))

    # plot target data grid
    im1 = axes[0].matshow(target_vals)
    axes[0].set_title('Goal Function')
    axes[0].set_xlabel('Connectivity Scaling')
    axes[0].set_ylabel('Excitatory Synaptic Efficacy')

    # set axis ticks and colorbar of target data grid
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

    # plot target values over parameter 1 for selected values of parameter 2
    axes[2].plot(params_test[:, 0], target_vals[efficacies_to_plot, :].T)
    axes[2].set_xticklabels(x_tick_labels)
    axes[2].set_title('Selected Goal Function Values')
    axes[2].set_xlabel('Connectivity Scaling')
    axes[2].set_ylabel('Excitatory Synaptic Efficacy')
    axes[2].legend([f'H_e = {params_test[efficacies_to_plot[0],1]}', f'H_e = {params_test[efficacies_to_plot[1],1]}',
                       f'H_e = {params_test[efficacies_to_plot[2],1]}'])

    # plot target values over parameter 1 for selected values of parameter 2
    lines = axes[2].lines.copy()
    for j, l in enumerate(lines):
        axes[2].plot(params_test[:, 0], predicted_vals[efficacies_to_plot[j], :].T, '--', color=l.get_color())
    axes[2].set_xticklabels(x_tick_labels)
    axes[2].set_title('Selected GPC-Predicted Function Values')
    axes[2].set_xlabel('Connectivity Scaling')
    axes[2].set_ylabel('Excitatory Synaptic Efficacy')
    axes[2].legend([f'H_e = {params_test[efficacies_to_plot[0],1]}', f'H_e = {params_test[efficacies_to_plot[1],1]}',
                    f'H_e = {params_test[efficacies_to_plot[2],1]}'])

    # plot gpc predictions grid
    im2 = axes[1].matshow(predicted_vals, clim=[np.min(target_vals), np.max(target_vals)])
    axes[1].set_title(f'GPC Prediction (order = {order[i]})')
    axes[1].set_xlabel('Connectivity Scaling')
    axes[1].set_ylabel('Excitatory Synaptic Efficacy')

    # set axis ticks and colorbar of predictions grid
    x_tick_labels = list()
    y_tick_labels = list()
    x_ticks = axes[1].get_xticks()
    y_ticks = axes[1].get_yticks()
    for x, y in zip(x_ticks, y_ticks):
        if (x < 0) or (x >= n_samples):
            x_tick_labels.append('')
            y_tick_labels.append('')
        else:
            x_tick_labels.append(round(params_test[int(x), 0], 1))
            y_tick_labels.append(round(params_test[int(y), 1], 4))
    axes[1].set_xticklabels(x_tick_labels)
    axes[1].set_yticklabels(y_tick_labels)

    # plot colorbars
    fig.colorbar(im1, ax=axes[0], orientation='vertical')
    fig.colorbar(im2, ax=axes[1], orientation='vertical')

    tight_layout()
    fig.show()

print('finished')