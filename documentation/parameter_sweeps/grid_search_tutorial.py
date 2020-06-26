"""

A Single Parameter Grid Search
==============================

In this tutorial, you will learn how to use the :code:`pyrates.utility.grid_search()` function, which allows you to
perform simulations of multiple parameterizations of a model in parallel. You will learn in which ways you can specify
your parameter grid, how you can customize your simulation settings and how to handle the output of the
:code:`grid_search()` function.

As an example, we will use the Jansen-Rit model (check out the model introduction for the Jansen-Rit model, to learn
about the mathematics behind the model and about its implementation in PyRates) [1]_.
We will perform a 1D parameter sweep over its connectivity scaling parameter :math:`C`.

References
^^^^^^^^^^

.. [1] B.H. Jansen & V.G. Rit (1995) *Electroencephalogram and visual evoked potential generation in a mathematical
       model of coupled cortical columns.* Biological Cybernetics, 73(4): 357-366.

"""

# %%
# First, let's import the :code:`grid_search()` function from PyRates

from pyrates.utility import grid_search

# %%
# The parameter C
# ---------------
#
# The Jansen-Rit model consists of 3 populations: Pyramidal cells (PCs), excitatory interneurons (EINs), and inhibitory
# interneurons (IINs). Between those populations, there exist 4 synaptic connections: PC to EIN, PC to IIN, EIN to PC,
# and IIN to PC. They are all scaled by a single base connectivity strength :math:`C`. We will now perform a parameter
# sweep over this parameter, i.e. examine the Jansen-Rit model behavior for a number of different values of this general
# synaptic strength scaling. This parameter sweep has also been performed in the original publication in which Jansen
# and Rit introduced their model. Check out [1]_ to compare the results of this parameter sweep with the published
# results.

# %%
# The parameter grid definition
# -----------------------------
#
# To perform a parameter sweep, we first define a range of values for C:

param_grid = {'C': [68., 128., 135., 270., 675., 1350.]}

# %%
# As can be seen, parameter grids are defined as dictionaries, where each parameter that should be changed during the
# sweep receives a separate key, with the sweep values following as a list or numpy array afterwards.
# Furthermore, we need to define over which model parameter to perform the sweep. This is done via a parameter map that
# maps each key in the parameter grid to a certain variable in the model:

param_map = {'C': {'vars': ['JRC_op/c'], 'nodes': ['JRC']}}

# %%
# For each parameter in the sweep, you can define multiple variables that should take on the values of this parameter.
# These variables can even be placed in different nodes in the network. Simply provide the names of all nodes as a list
# under the key :code:`nodes` and the operator and variable names as a list under the key :code:`vars`,
# following the format :code:`op_name/var_name`.
#
# These are the basic ingredients that are required by the :code:`grid_search()` function in addition to the standard
# arguments that are needed for the model initialization and simulation. These arguments are explained in detail in the
# documentation of the :code:`pyrates.ir.CircuitIR.compile()` and :code:`pyrates.ir.CircuitIR.run()` methods and in the
# example gallery sections for model compilation and simulation.

# %%
# Performing the parameter sweep
# ------------------------------
#
# To perform the parameter sweep, execute the following call to the :code:`grid_search()` function:

results, results_map = grid_search(circuit_template="model_templates.jansen_rit.simple_jansenrit.JRC_simple",
                                   param_grid=param_grid,
                                   param_map=param_map,
                                   simulation_time=10.0,
                                   step_size=1e-4,
                                   sampling_step_size=1e-3,
                                   inputs={},
                                   outputs={'V_pce': 'JRC/JRC_op/PSP_pc_e', 'V_pci': 'JRC/JRC_op/PSP_pc_i'},
                                   init_kwargs={'backend': 'numpy', 'solver': 'scipy'}
                                   )

# %%
# After performing the parameter sweep, :code:`grid_search()` returns a tuple with 2 entries:
#   - a 2D :code:`pandas.DataFrame` that contains the simulated timeseries (1. dimension) for each output variable for
#     each model parametrization (2. dimension)
#   - a 2D :code:`pandas.DataFrame` that contains a mapping between the column names of the timeseries in the first
#     tuple entry (1. dimension) and the parameter values of the parameters that were defined in the paramter grid
#     (2. dimension)
#
# Now, lets visualize the results of this parameter sweep for each value of :math:`C`:

from pyrates.utility import create_cmap, plot_timeseries
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=len(param_grid['C']), figsize=(8, 12))

# create the color map
cmap = create_cmap('pyrates_blue', as_cmap=False, n_colors=1, reverse=True)

# sort the results map via the values of C
results_map.sort_values('C', inplace=True)

# plot the raw output variable for each condition
for i, ax in enumerate(axes):
    key = results_map.index[i]
    psp_e = results.loc[1.0:, ('V_pce', key)]
    psp_i = results.loc[1.0:, ('V_pci', key)]
    plot_timeseries(psp_e - psp_i, ax=ax, cmap=cmap, ylabel='PSP')
    ax.legend([f"C = {results_map.at[key, 'C']}"], loc='upper right')

plt.show()

# %%
# Note that, since the parameter values are arranged in a :code:`pandas.DataFrame`, sorting their values is very
# straight forward. For each value of :math:`C` in ascending order, we extract the name of the respective columns in
# :code:`results` to receive our 2 stored output variables. The difference between those two state variables resembles
# the average membrane potential of the PC population of the Jansen-Rit model. This is what we plot for each condition.
# If you compare these results to the results in [1]_, you will notice that they differ. This is, because Jansen and Rit
# used noisy input to the model. To receive the same results as in [1]_, we will have to define extrinsic noise to
# drive the model with. This can be done the following way:

import numpy as np

T = 10.0
dt = 1e-4
noise = np.random.uniform(120.0, 320.0, size=(int(np.round(T/dt, decimals=0)), 1))

results, results_map = grid_search(circuit_template="model_templates.jansen_rit.simple_jansenrit.JRC_simple",
                                   param_grid=param_grid,
                                   param_map=param_map,
                                   simulation_time=10.0,
                                   step_size=1e-4,
                                   sampling_step_size=1e-3,
                                   inputs={'JRC/JRC_op/u': noise},
                                   outputs={'V_pce': 'JRC/JRC_op/PSP_pc_e', 'V_pci': 'JRC/JRC_op/PSP_pc_i'},
                                   init_kwargs={'backend': 'numpy', 'solver': 'euler'}
                                   )

# %%
# Here, we used numpy to create an array of random numbers, uniformly distributed between 120.0 and 320.0 (equivalent to
# what Jansen and Rit used). We then provided this array as input to the model parameter :math:`u`, which usually is a
# constant. This way, it is replaced by a timeseries and thus represents a forcing term in the differential equations.
#
# Now, lets have a look at the results again:

fig, axes = plt.subplots(nrows=len(param_grid['C']), figsize=(8, 12))

# create the color map
cmap = create_cmap('pyrates_blue', as_cmap=False, n_colors=1, reverse=True)

# sort the results map via the values of C
results_map.sort_values('C', inplace=True)

# plot the raw output variable for each condition
for i, ax in enumerate(axes):
    key = results_map.index[i]
    psp_e = results.loc[1.0:, ('V_pce', key)]
    psp_i = results.loc[1.0:, ('V_pci', key)]
    plot_timeseries(psp_e - psp_i, ax=ax, cmap=cmap, ylabel='PSP')
    ax.legend([f"C = {results_map.at[key, 'C']}"], loc='upper right')

plt.show()

# %%
# Now, the results look much more like the timeseries in [1]_. Of course they are not the same, because the random
# process cannot be exactly the same. However, the timeseries represent the same model behavior.
