"""
Theta Neuron Mean-Field Model
=============================

Here, we will introduce the mean-field model of a network of all-to-all coupled theta neurons. The theta neuron model
is also known as the `Ermentrout-Kopell <http://www.scholarpedia.org/article/Ermentrout-Kopell_canonical_model>`_ model
and takes the form

.. math::

    \\dot \\theta_i &= (1 - \\cos(\\theta_i)) + I(t) (1 + \\cos(\\theta_i)),

with neural phase :math:`\\theta_i` and mean-field input :math:`I(t)` [1]_. The phase variable lies on the unit circle
and when :math:`\\theta = \\pi` a spike is counted. Considering an all-to-all coupled network of :math:`N` theta neurons
in the continuum limit :math:`N \\rightarrow \\infty`, the evolution equations for the Kuramoto order parameter
:math:`z` can be derived [2]_. For :math:`I(t) = \\eta_i + \\frac{J}{N}\\sum_{i=1}^{N} P_n(\\theta_i)` with :math:`P_n` being
a pulse-like synaptic current centered around :math:`\\theta_i = \\pi`, these mean-field equations take the form

.. math::

        \\dot z = \\frac{1}{2} [(i*(\\eta + s v_s) - \\Delta) (z+1)^2 - s(z^2-1) - i(z-1)^2], \\\\
        s = \\frac{J (1-|z|)^2}{(1+z+\\bar z+ |z|^2}.

They are governed by 3 parameters:
    - :math:`\\eta` --> the mean of a Lorenzian distribution over the neural excitability in the population
    - :math:`\\Delta` --> the half-width at half maximum of the Lorenzian distribution over the neural excitability
    - :math:`J` --> the strength of the recurrent coupling inside the population
Also, :math:`\\bar z` represents the complex conjugate of :math:`z`.
This mean-field model is an exact representation of the macroscopic synchronization dynamics of a
spiking neural network consisting of theta neurons with Lorentzian distributed background excitabilities.

Below, we will demonstrate ow to load the mhodel template into pyrates, perform simulations with it and visualize the
results.

References
^^^^^^^^^^

.. [1] G.B. Ermentrout and N. Kopell (1986) *Parabolic bursting in an excitable system coupled with a slow oscillation.* SIAM-J.-Appl.-Math 46: 233-253.

.. [2] T.B. Luke, E. Barreto and P. So (2013) *Complete Classification of the Macroscopic Behavior of a Heterogeneous Network of Theta Neurons* Neural Computation 25: 3207-3234.

"""

# %%
# Step 1: Numerical simulation of a the model behavior in time
# ------------------------------------------------------------
#
# Here, we use the :code:`integrate` function imported from PyRates. As a first argument to this function, either a path
# to a YAML-based model definition or a :code:`CircuitTemplate` instance can be provided. The function will then compile
# the model and solve the initial value problem of the above defined differential equations for a time interval from
# 0 to the given simulation time. This solution will be calculated numerically by a differential equation solver in
# the backend, starting with a defined step-size. Here, we use the default backend and scipy solver. Furthermore,
# we provide a step-function extrinsic input that excites all theta neurons in a time window from :code:`start` to
# :code:`stop`. This input is defined on a time vector with fixed time steps of size :code:`step_size`.
# Check out the arguments of the code:`CircuitTemplate.run()` method for a detailed explanation of the
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
results = integrate("model_templates.neural_mass_models.theta.kmo_theta", step_size=step_size, simulation_time=T,
                    outputs={'z': 'p/theta_op/z'}, inputs={'p/theta_op/I_ext': I_ext}, clear=True, solver='scipy')

# %%
# Step 2: Visualization of the solution
# -------------------------------------
#
# The output of the :code:`simulate()` function is a :code:`pandas.Dataframe`, which allows for direct plotting of the
# timeseries it contains.
# This timeseries represents the numerical solution of the initial value problem solved in step 1 with respect to the
# state variable :math:`z` of the model. Since it is a complex-valued variable, we plot the absolute value of it here.

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(np.abs(results))
ax.set_xlabel('time')
ax.set_ylabel(r'|z|')
plt.show()
