"""
The Kuramoto Oscillator
=======================

Here, we will introduce the `Kuramoto model <https://en.wikipedia.org/wiki/Kuramoto_model>`_,
a generic phase oscillator model with a wide range of applications [1]_.
In its simplest form, each Kuramoto oscillator is governed by a non-linear, 1st order ODE:

.. math::
        \\dot \\theta_i &= \\omega + \\sum_j J_{ij} sin(\\theta_j - \\theta_i),
        :label: eq1

with phase :math:`\\theta` and intrinsic frequency :math:`\\omega`. The sum represents sinusoidal coupling with all
other oscillators in the network with coupling strengths :math:`J_{ij}`.

In a first step, we'll consider two coupled Kuramoto oscillators, with an additional extrinsic input :math:`P(t)`
entering at one of them:

.. math::
        \\dot \\theta_1 &= \\omega_1 + P(t) + J_1 sin(\\theta_2 - \\theta_1), \\\\
        \\dot \\theta_2 &= \\omega_2 + J_2 sin(\\theta_1 - \\theta_2).
        :label: eq2

Below, we will first demonstrate how this model can be used and examined via PyRates.

References
^^^^^^^^^^

.. [1] Y. Kuramoto (1991) *Collective synchronization of pulse-coupled oscillators and excitable units.* Physia D: Nonlinear Phenomena 50(1): 15-30.

.. [2] Ott and Antonsen (2008) *Low dimensional behavior of large systems of globally coupled oscillators.* Chaos 18(3): 1054-1500.

"""

# %%
# 2 Coupled Kuramoto Oscillators
# ------------------------------
#
# Here, we use the :code:`integrate` function imported from PyRates. As a first argument to this function, either a path
# to a YAML-based model definition or a :code:`CircuitTemplate` instance can be provided. The function will then compile
# the model and solve the initial value problem of the above defined differential equations for a time interval from
# 0 to the given simulation time. This solution will be calculated numerically by a differential equation solver in
# the backend, starting with a defined step-size. Here, we use the default backend and solver. Furthermore,
# we provide a step-function extrinsic input to one of the Kuramoto oscillators in a time window from :code:`start` to
# :code:`stop`. This input is defined on a time vector with fixed time steps of size :code:`step_size`.
# Check out the arguments of the :code:`CircuitTemplate.run()` method for a detailed explanation of the
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
results = integrate("model_templates.oscillators.kuramoto.kmo_2coupled", step_size=step_size, simulation_time=T,
                    outputs={'theta_1': 'p1/phase_op/theta', 'theta_2': 'p2/phase_op/theta'},
                    inputs={'p1/phase_op/s_ext': I_ext}, clear=True)

# plot resulting phases
import matplotlib.pyplot as plt
plt.plot(np.sin(results*2*np.pi))
plt.show()

# %%
# Kuramoto Order Parameter Dynamics
# ---------------------------------
#
# The `Kuramoto order parameter <https://mathinsight.org/applet/kuramoto_order_parameters>`_ :math:`z` of a system of
# coupled phase oscillators is given by
#
# .. math::
#         z = \sigma e^{i\phi} = \frac{1}{N} \sum_{j=1}^N e^{i \theta_j},
#         :label: eq3
#
# where :math:`\sigma` and :math:`\phi` are the phase coherence and average phase of the phase oscillators, respectively.
# In 2008, Ott and Antonsen derived the evolution equations for these order parameters for a system of all-to-all
# coupled Kuramoto oscillators of the form :eq:`eq1`. While the evolution of the average phase is determined by a mere
# constant, the evolution equation of :math:`z` is given by
#
# .. math::
#        \dot z = z(i\omega - \Delta) - \frac{\bar{J z} z^2 - J z}{2},
#        :label: eq4
#
# where :math:`\bar z` denotes the complex conjugate of :math:`z`, and :math:`\omega` and :math:`\Delta` represent the
# center and half-width-at-half-maximum of a Lorentzian distribution over the intrinsic frequencies :math:`\omega_i` of
# the individual Kuramoto oscillators (see [2]_ for a detailed derivation of the mean-field equation).
# Below, we simulate the dynamics of the dynamics of :math:`z` of an all-to-all coupled system of Kuramoto oscillators
# in response to a step-function input (similar to the simulation above). Note that :math:`z` is a complex variable and
# we plot its absolute value :math:`|z|` to receive the coherence of the system over time.

# define simulation time and input start and stop
T = 40.0
step_size = 1e-4
start = 10.0
stop = 30.0

# extrinsic input definition
steps = int(np.round(T/step_size))
I_ext = np.zeros((steps,))
I_ext[int(start/step_size):int(stop/step_size)] = 2.0

# perform simulation
results = integrate("model_templates.oscillators.kuramoto.kmo_mf", step_size=step_size, simulation_time=T,
                    outputs={'z': 'p/kmo_op/z'}, inputs={'p/kmo_op/s_ext': I_ext}, clear=True, solver='scipy')

# plot resulting coherence dynamics
import matplotlib.pyplot as plt
plt.plot(np.abs(results))
plt.show()

# %%
# As can be seen, the system engaged in synchronized oscillations within the input period.
