"""
The Wilson-Cowan Neural Mass Model
================================

Here, we will introduce the Wilson-Cowan model, a popular neural mass model of the dynamic interactions between
2 populations: excitatory projection neurons (E), and inhibitory interneurons (I).

The model is of phenomenological nature, but captures well the steady-states in average firing rate of populations of
leaky integrate-and-fire neurons [1]_. It is probably the most-used existing neural mass model and has been applied to
a wide range of brain systems and neuroscientific questions.
The dynamics of each population are described by a 1st-order ordinary differential equation (ODE):

.. math::
        \\tau \\dot r &= -r + (k - q r) S(m),

where :math:`r` represents the average firing rate of the population, :math:`\tau` is a lumped population time constant,
:math:`k` is a coupling term, :math:`q` is a refractory term, and :math:`S(m)` is the synaptic input.
Typically, :math:`S` is chosen as a sigmoidal transfer function that transforms the pre-synaptic membrane potential
into an average firing rate:

.. math::
        S(m) = \\frac{1}{1 + e^{-m}}.

Considering both the E and the I population, the Wilson-Cowan model can be described via two coupled ODEs:

.. math::

        \\tau_e \\dot r_e &= -r_e + (k_e - q_e r_e) S(c_{ee} r_e + c_{ei} r_i + I_e(t)), \n
        \\tau_i \\dot r_i &= -r_i + (k_i - q_i r_i) S(c_{ie} r_e + c_{ii} r_i + I_i(t))

where additional extrinsic inputs :math:`I_e` and :math:`I_i` have been added.
Below, we will demonstrate how to load this a model into pyrates and perform numerical simulations with it.

References
^^^^^^^^^^

.. [1] H.R. Wilson and J.D. Cowan (1972) *Excitatory and Inhibitory Interactions in Localized Populations of Model
       Neurons.* Biophysical Journal 12.

"""

# %%
# Step 1: Importing the frontend class for defining models
# --------------------------------------------------------
#
# As a first step, we import the :code:`pyrates.frontend.CircuitTemplate` class, which allows us to set up a model
# definition in PyRates.

from pyrates.frontend import CircuitTemplate

# %%
# Step 2: Loading a model template from the `model_templates` library
# -------------------------------------------------------------------
#
# In the second step, we load a model template for the Wilson-Cowan model that comes with PyRates via the
# :code:`from_yaml()` method of the :code:`CircuitTemplate`. This method returns a :code:`CircuitTemplate` instance.
# Have a look at the yaml definition of the model that can be found via the path used for the :code:`from_yaml()`
# method. You will see that all variables and parameters are already defined there. As an alternative, you can also use
# the :code:`circuit_from_yaml()` function that you can simply import via :code:`from pyrates import circuit_from_yaml`.
# These are the basic steps you perform, if you want to load a model that is defined inside a yaml file.
# To check out the different model templates provided by PyRates, have a look at the :code:`PyRates.model_templates`
# module.

wc = CircuitTemplate.from_yaml("model_templates.neural_mass_models.wilsoncowan.WC")

# %%
# Step 3: Numerical simulation of a the model behavior in time
# ------------------------------------------------------------
#
# After loading the model template, numerical simulations can be performed via the :code:`run()` method.
# Calling this function will solve the initial value problem of the above defined differential equations for a time
# interval from 0 to the given simulation time.
# This solution will be calculated numerically by a differential equation solver in the backend, starting with a defined
# step-size. As part of this process, a :code:`pyrates.backend.BaseBackend` instance is created that contains a graph
# representation of the network equations. How exactly these equations will be solved, can be customized via the
# arguments of the :code:`run()` method. You can choose between different backends, numerical solvers and integration
# step-sizes, for instance. Here, we choose the `scipy` backend, which uses a 4(5)th order Runge-Kutta method as default
# numerical solver. The default of the :code:`run()` method is the forward Euler method that is implemented in PyRates
# itself.

import numpy as np

# simulation parameters
T = 125.0
dt = 5e-4
dts = 1e-2
steps = int(np.round(T/dt))
inp = np.zeros((steps,))
inp[int(25/dt):int(100/dt)] = 1.0

# perform simulation
results = wc.run(simulation_time=T,
                 step_size=dt,
                 sampling_step_size=dts,
                 outputs={'E': 'e/rate_op/r',
                          'I': 'i/rate_op/r'},
                 inputs={'e/se_op/r_ext': inp},
                 backend='default',
                 solver='euler')

# %%
# Step 4: Visualization of the solution
# -------------------------------------
#
# The output of the :code:`run()` method is a :code:`pandas.Dataframe`, which comes with a :code:`plot()` method for
# plotting the timeseries it contains.
# This timeseries represents the numerical solution of the initial value problem solved in step 4 with respect to the
# state variables :math:`r_e` and :math:`r_i` of the model.

import matplotlib.pyplot as plt
plt.plot(results)
plt.show()
