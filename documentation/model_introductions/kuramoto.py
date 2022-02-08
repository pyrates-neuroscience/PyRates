"""

The Kuramoto Oscillator
=======================

Here, we will introduce the Kuramoto model, a generic phase oscillator model with a wide range of applications [1]_.
In its simplest form, each Kuramoto oscillator is governed by a non-linear, 1st order ODE:

.. math::
        \\dot \\theta_i &= \\omega + \\sum_j J_{ij} sin(\\theta_j - \\theta_i),

with phase :math:`\theta` and intrinsic frequency :math:`\omega`. The sum represents sinusoidal coupling with all other
oscillators in the network with coupling strengths :math:`J_{ij}`.

In a first step, we'll consider two coupled Kuramoto oscillators, with an additional extrinsic input :math:`P(t)`
entering at one of them:

.. math::

        \\dot \\theta_1 &= \\omega_1 + P(t) + J_1 sin(\\theta_2 - \\theta_1), \n
        \\dot \\theta_2 &= \\omega_2 + J_2 sin(\\theta_1 - \\theta_2)

Below, we will first demonstrate how this model can be used and examined via PyRates.

References
^^^^^^^^^^

.. [1] Y. Kuramoto (1991) *Collective synchronization of pulse-coupled oscillators and excitable units.* Physia D:
Nonlinear Phenomena 50(1): 15-30.

"""

# %%
# 2 Coupled Kuramoto Oscillators
# ==============================
#
# Step 1: Numerical simulation of a the model behavior in time
# ------------------------------------------------------------
#
# Here, we use the :code:`integrate` function imported from PyRates. As a first argument to this function, either a path
# to a YAML-based model definition or a :code:`CircuitTemplate` instance can be provided. The function will then compile
# the model and solve the initial value problem of the above defined differential equations for a time interval from
# 0 to the given simulation time. This solution will be calculated numerically by a differential equation solver in
# the backend, starting with a defined step-size. Here, we use the default backend and solver. Furthermore,
# we provide a step-function extrinsic input to one of the Kuramoto oscillators in a time window from :code:`start` to
# :code:`stop`. This input is defined on a time vector with fixed time steps of size :code:`step_size`.
# Check out the arguments of the code:`CircuitTemplate.run()` method for a detailed explanation of the
# arguments that you can use to adjust this numerical procedure.

from pyrates import integrate
import numpy as np

# define simulation time and input start and stop
T = 1.0
step_size = 1e-4
start = 0.2
stop = 0.8

# extrinsic input definition
steps = int(np.round(T/step_size))
I_ext = np.zeros((steps,))
I_ext[int(start/step_size):int(stop/step_size)] = 1.0

# perform simulation
results = integrate("model_templates.coupled_oscillators.kuramoto.kmo_2coupled", step_size=step_size, simulation_time=T,
                    outputs={'theta_1': 'p1/phase_op/theta', 'theta_2': 'p2/phase_op/theta'},
                    inputs={'p1/phase_op/ext_in': I_ext}, clear=True)

# plot resulting phases
import matplotlib.pyplot as plt
plt.plot(np.sin(results*2*np.pi))
plt.show()
