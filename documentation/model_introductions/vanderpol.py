"""
The Van der Pol Oscillator
==========================

Here, we will introduce the Van der Pol model, a widely studied non-linear oscillator model [1]_.
In its two-dimensional form, the Van der Pol oscillator is governed by the following non-linear ODEs:

.. math::
        \\dot x &= z, \n
        \\dot z &= \\mu (1-x^2) z - x,

with damping constant :math:`\mu`. Depending on the latter parameter, the periodic solutions of the Van der Pol oscillator change,
with simple harmonic oscillations for :math:`\mu = 0` and non-harmonic limit cycle oscillations for :math:`\mu > 0`.
In this gallery, we will simulate and visualize the model dynamics for :math:`\mu = 0` and :math:`\mu = 0`.

References
^^^^^^^^^^

.. [1] T. Kanamaru (2007) *Van der Pol oscillator* Scholarpedia 2(1): 2202. DOI: 10.4249/scholarpedia.2202.

"""

# %%
# Van der Pol oscillator dynamics
# -------------------------------
#
# Step 1: Numerical simulation of the model dynamics for :math:`\mu = 0`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# First, we will load a :code:`CircuitTemplate` instance via the path to the model definition.
# We will the use the :code:`CircuitTemplate.run` method to solve the initial value problem of the above defined
# differential equations for a time interval from 0 to the given simulation time.
# This solution will be calculated numerically by a differential equation solver in the backend, starting with a
# defined step-size. Here, we use the default backend and a Runge-Kutta 2(3) solver.
# Check out the arguments of the :code:`CircuitTemplate.run()` method for a detailed explanation of the
# arguments that you can use to adjust this numerical procedure.

from pyrates import CircuitTemplate, clear

# define simulation time and input start and stop
T = 100.0
step_size = 1e-2

# load model template
model = CircuitTemplate.from_yaml("model_templates.oscillators.vanderpol.vdp")

# set mu to 0
model.update_var({'p/vdp_op/mu': 0.0})
results = model.run(step_size=step_size, simulation_time=T, outputs={'x': 'p/vdp_op/x'}, solver='scipy', method='RK23')

# plot resulting phases
import matplotlib.pyplot as plt
plt.plot(results)
plt.show()

# clear results
clear(model)

# %%
#
# Step 2: Numerical simulation of the model dynamics for :math:`\mu = 5`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# As expected, we found harmonical oscillations for :math:`\mu = 0`.
# Now, we will repeat the procedure for :math:`\mu = 5`.

# load model template
model = CircuitTemplate.from_yaml("model_templates.oscillators.vanderpol.vdp")

# set mu to 0
model.update_var({'p/vdp_op/mu': 5.0})
results = model.run(step_size=step_size, simulation_time=T, outputs={'x': 'p/vdp_op/x'}, solver='scipy', method='RK23')

# plot resulting phases
plt.plot(results)
plt.show()

# %%
# We find a much less harmonical waveform of the oscillations. For investigations of the Van der Pol oscillator driven
# by a periodic input signal, see the `entrainment example <https://pyrates.readthedocs.io/en/latest/auto_analysis/entrainment.html>`_
# in the use examples section `Model analysis`.
