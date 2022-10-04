"""
Izhikevich Neuron Mean-Field Model
==================================

Here, we will introduce the Izhikevich population mean-field model, which has been derived from a population of
all-to-all coupled Izhikevich neurons (see [1]_ for an introduction to the Izhikevich neuron model) in [2]_.
The model equations are given by:

.. math::

    \\tau \\dot r &= \\frac{\\Delta}{\\pi\\tau} +  r(2 v - \\alpha - g \\tau s), \n
    \\tau \\dot v &= v^2 - \\alpha v + \\bar\\eta + I(t) - u + g s \\tau (E-v) - (\\pi r \\tau)^2, \n
    \\dot u &= a(b v - u) + d r, \n
    \\tau_s \\dot s &= -s + \\tau_s J r,

where :math:`r` is the average firing rate, :math:`v` is the average membrane potential, :math:`u` is an average
recovery variable, and :math:`s` is a post-synaptic current.
It is governed by the following parameters:
    - :math:`\\tau` --> the population time constant
    - :math:`\\bar \\eta` --> the mean of a Lorenzian distribution over the neural excitability in the population
    - :math:`\\Delta` --> the half-width at half maximum of the Lorenzian distribution over the neural excitability
    - :math:`alpha` --> controls the leaking properties of the membrane potential dynamics
    - :math:`g` --> the maximal synaptic conductance
    - :math:`E` --> the reversal potential of the synapse
    - :math:`a`, :math:`b` and :math:`d` --> control parameters for the recovery variable
    - :math:`J` and :math:`\\tau_s` --> Strength and time constant of the post-synaptic current

This mean-field model is a representation of the macroscopic dynamics of a spiking neural network consisting of
dimensionless Izhikevich neurons with Lorentzian distributed background excitabilities [2]_.
In the sections below, we will demonstrate how to load the model template into pyrates, perform
simulations with it and visualize the results.
Note that PyRates also provides model templates for the biophysical Izhikevich mean-field model with distributed
spike-threshold heterogeneity as derived in [3]_. The model templates are available in the same YAML file.

References
^^^^^^^^^^

.. [1] E. Izhikevich (2007) *Dynamical Systems in Neuroscience: The Geometry of Excitability and Bursting.* MIT Press.

.. [2] L. Chen, S.A. Campbell (2022) *Exact mean-field models for spiking neural networks with adaptation.*
       arXiv:2203.08341v1, https://arxiv.org/abs/2203.08341.

.. [3] R. Gast, S. A. Solla, A. Kennedy (2022) *Effects of Neural Heterogeneity on Spiking Neural Network Dynamics*
       arXiv:2206.08813v1, https://arxiv.org/abs/2206.08813.

"""

# %%
#
# First, we will use the high-level interfaces from PyRates to perform model loading and simulation in a single step for
# the default parametrization of the Izhikevich model.

# %%
# Step 1: Numerical simulation of a the model behavior in time
# ------------------------------------------------------------
#
# Here, we use the :code:`integrate` function imported from PyRates. As a first argument to this function, either a path
# to a YAML-based model definition or a :code:`CircuitTemplate` instance can be provided. The function will then compile
# the model and solve the initial value problem of the above defined differential equations for a time interval from
# 0 to the given simulation time. This solution will be calculated numerically by a differential equation solver in
# the backend, starting with a defined step-size. Here, we use the default backend and solver. Furthermore,
# we provide a step-function extrinsic input that excites the population model in a time window from :code:`start` to
# :code:`stop`. This input is defined on a time vector with fixed time steps of size :code:`step_size`.
# Check out the arguments of the :code:`CircuitTemplate.run()` method for a detailed explanation of the
# arguments that you can use to adjust this numerical procedure.

from pyrates import integrate
import numpy as np

# define simulation time and input start and stop
T = 1000.0
step_size = 5e-4
sampling = 1e-2
start = 400.0
stop = 600.0

# extrinsic input definition
steps = int(np.round(T/step_size))
I_ext = np.zeros((steps,)) + 0.15
I_ext[int(start/step_size):int(stop/step_size)] -= 0.15

# perform simulation
results = integrate("model_templates.neural_mass_models.ik.ik_nodim", step_size=step_size, simulation_time=T,
                    outputs={'r': 'p/ik_nodim_op/r'}, inputs={'p/ik_nodim_op/I_ext': I_ext}, solver='scipy', clear=True,
                    sampling_step_size=sampling)

# %%
# Step 2: Visualization of the solution
# -------------------------------------
#
# The output of the :code:`simulate()` function is a :code:`pandas.Dataframe`, which allows for direct plotting of the
# timeseries it contains.
# This timeseries represents the numerical solution of the initial value problem solved in step 1 with respect to the
# state variable :math:`r` of the model. You can observe that the model is situated in an asynchronous regime initially
# and expresses transient synchronization in response to the extrinsic input.

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(results)
ax.set_xlabel('time')
ax.set_ylabel('r')

# %%
# Step 3: Changing the model parametrization
# -------------------------------------------
#
# Now, lets change the model parameters and repeat the simulation from above. As a target parameter to change, we alter
# the maximal synaptic conductance :math:`g`.

from pyrates import CircuitTemplate

# load model template
ik = CircuitTemplate.from_yaml("model_templates.neural_mass_models.ik.ik_nodim")

# change model parameters
ik.update_var(node_vars={"p/ik_nodim_op/g": 1.5})

# perform simulation
results2 = ik.run(simulation_time=T, step_size=step_size, outputs={'r': 'p/ik_nodim_op/r'},
                  inputs={'p/ik_nodim_op/I_ext': I_ext}, solver='scipy', sampling_step_size=sampling, clear=True)

fig2, ax2 = plt.subplots()
ax2.plot(results2)
ax2.set_xlabel('time')
ax2.set_ylabel('r')
plt.show()

# %%
# you can see, changing the synaptic conductance drove the Izhikevich model in an oscillating regime and the extrinsic
# input now suppresses these oscillations. Check out [2]_ if you would like to test out some different parameter
# regimes. For a detailed introduction on how to handle model definitions via YAML files, have a look at
# the `model definition use example <https://pyrates.readthedocs.io/en/latest/auto_implementations/yaml_definitions.html>`_.
