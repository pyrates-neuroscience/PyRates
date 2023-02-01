"""
Quadratic Integrate-and-Fire (QIF) Neuron Mean-Field Model
==========================================================

Here, we will introduce the QIF population mean-field model, which has been derived from a population of all-to-all
coupled QIF neurons in [1]_. The model equations are given by:

.. math::

    \\tau \\dot r &= \\frac{\\Delta}{\\pi\\tau} + 2 r v, \n
    \\tau \\dot v &= v^2 +\\bar\\eta + I(t) + J r \\tau - (\\pi r \\tau)^2,

where :math:`r` is the average firing rate and :math:`v` is the average membrane potential of the QIF population [1]_.
It is governed by 4 parameters:
    - :math:`\\tau` --> the population time constant
    - :math:`\\bar \\eta` --> the mean of a Cauchy distribution over the neural excitability in the population
    - :math:`\\Delta` --> the half-width at half maximum of the Cauchy distribution over the neural excitability
    - :math:`J` --> the strength of the recurrent coupling inside the population
This mean-field model is an exact representation of the macroscopic firing rate and membrane potential dynamics of a
spiking neural network consisting of QIF neurons with `Cauchy <https://en.wikipedia.org/wiki/Cauchy_distribution>`_ distributed background excitabilities.
While the mean-field derivation is mathematically only valid for all-to-all coupled populations of infinite size,
it has been shown that there is a close correspondence between the mean-field model and neural populations with
sparse coupling and population sizes of a few thousand neurons [2]_. In the same work, it has been demonstrated how to
extend the model by adding synaptic dynamics or additional adaptation currents to the single cell network, that can be
carried through the mean-field derivation performed in [1]_. For example, a QIF population with
spike-frequency adaptation would be given by the following 3D system:

.. math::

    \\tau \\dot r &= \\frac{\\Delta}{\\pi\\tau} + 2 r v, \n
    \\tau \\dot v &= v^2 +\\bar\\eta + I(t) + J r \\tau - a - (\\pi r \\tau)^2, \n
    \\tau_a \\dot a &= -a + \\alpha \\tau_a r,

where the evolution equation for :math:`a` expresses a convolution of :math:`r` with a mono-exponential kernel, with
adaptation strength :math:`\\alpha` and time constant :math:`\\tau_a`.

In the sections below, we will demonstrate for each model how to load the model template into pyrates, perform
simulations with it and visualize the results.

**References**

.. [1] E. Montbrió, D. Pazó, A. Roxin (2015) *Macroscopic description for networks of spiking neurons.* Physical
       Review X, 5:021028, https://doi.org/10.1103/PhysRevX.5.021028.

.. [2] R. Gast, H. Schmidt, T.R. Knösche (2020) *A Mean-Field Description of Bursting Dynamics in Spiking Neural
       Networks with Short-Term Adaptation.* Neural Computation 32 (9): 1615-1634, https://doi.org/10.1162/neco_a_01300.

"""

# %%
# Basic QIF Model
# ^^^^^^^^^^^^^^^
#
# We will start out by a tutorial of how to use the QIF model without adaptation. To this end, we will use the
# high-level interfaces from PyRates which allow to perform model loading and simulation in a single step.
# For a step-to-step tutorial of the more low-level interfaces, have a look at the Jansen-Rit example.

# %%
# Step 1: Numerical simulation of a the model behavior in time
# ------------------------------------------------------------
#
# Here, we use the :code:`integrate` function imported from PyRates. As a first argument to this function, either a path
# to a YAML-based model definition or a :code:`CircuitTemplate` instance can be provided. The function will then compile
# the model and solve the initial value problem of the above defined differential equations for a time interval from
# 0 to the given simulation time. This solution will be calculated numerically by a differential equation solver in
# the backend, starting with a defined step-size. Here, we use the default backend and solver. Furthermore,
# we provide a step-function extrinsic input that excites all QIF neurons in a time window from :code:`start` to
# :code:`stop`. This input is defined on a time vector with fixed time steps of size :code:`step_size`.
# Check out the arguments of the :code:`CircuitTemplate.run()` method for a detailed explanation of the
# arguments that you can use to adjust this numerical procedure.

from pyrates import integrate
import numpy as np

# define simulation time and input start and stop
T = 100.0
step_size = 5e-4
start = 20.0
stop = 80.0

# extrinsic input definition
steps = int(np.round(T/step_size))
I_ext = np.zeros((steps,))
I_ext[int(start/step_size):int(stop/step_size)] = 3.0

# perform simulation
results = integrate("model_templates.neural_mass_models.qif.qif", step_size=step_size, simulation_time=T,
                    outputs={'r': 'p/qif_op/r'}, inputs={'p/qif_op/I_ext': I_ext}, clear=True)

# %%
# Step 2: Visualization of the solution
# -------------------------------------
#
# The output of the :code:`simulate()` function is a :code:`pandas.Dataframe`, which allows for direct plotting of the
# timeseries it contains.
# This timeseries represents the numerical solution of the initial value problem solved in step 1 with respect to the
# state variable :math:`r` of the model. You can observe that the model expresses hysteresis and converges to a
# different steady-state after the input is turned off, than the steady-state it started from when the input was turned
# on. This behavior has been described in detail in [1]_.

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(results)
ax.set_xlabel('time')
ax.set_ylabel('r')

# %%
# QIF SFA Model
# ^^^^^^^^^^^^^
#
# Now, lets have a look at the QIF model with spike-frequency adaptation. We will follow the same steps as outlined
# above.

results2 = integrate("model_templates.neural_mass_models.qif.qif_sfa", simulation_time=T, step_size=step_size,
                     outputs={'r': 'p/qif_sfa_op/r'}, inputs={'p/qif_sfa_op/I_ext': I_ext}, clear=True)

fig2, ax2 = plt.subplots()
ax2.plot(results2)
ax2.set_xlabel('time')
ax2.set_ylabel('r')
plt.show()

# %%
# you can see that, by adding the adaptation variable to the model, we introduced synchronized bursting behavior to
# the model. Check out [2]_ if you would like to test out some different parameter regimes and would like to know what
# kind of model behavior to expect if you make changes to the adaptation parameters. To change the parameters, you need
# to derive a new operator template from the given operator template in a yaml file and simply set the parameter you
# would like to change. For a detailed introduction on how to handle model definitions via YAML files, have a look at
# the `model definition use example <https://pyrates.readthedocs.io/en/latest/auto_implementations/yaml_definitions.html>`_.
