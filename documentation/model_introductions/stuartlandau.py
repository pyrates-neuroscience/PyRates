"""
The Stuart-Landau Oscillator
============================

Here, we will introduce the Stuart Landau equations, a model of non-linear oscillating systems near a Hopf bifurcation [1]_.
In cartesian coordinates, the evolution equations of the Stuart Landau oscillator can be written as:

.. math::
        \\dot x &= \\omega*z + x*(1-z^2-x^2), \n
        \\dot z &= -\\omega*x + z*(1-z^2-x^2),

with intrinsic angular frequency :math:`\omega`. It can be used as the generating model of a harmonic oscillation with
frequency :math:`\omega` and is thus useful to incorporate periodic forcing into dynamical systems models.
Below, we provide an example where we apply periodic forcing to a Van der Pol oscillator (for a detailed description of
the latter, see the corresponding gallery example in the section `Model introductions`).

References
^^^^^^^^^^

.. [1] L.D. Landau (1944) *On the problem of turbulence* Conference paper: Dokl. Akad. Nauk UDSSR 44: 311.

"""

# %%
# Using the Stuart-Landau oscillator model for periodic forcing
# -------------------------------------------------------------
#
# Step 1: The Stuart-Landau oscillator dynamics
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Let us first have a look at the signal that the Stuart-Landau oscillator generates.
# To this end, we will load a :code:`CircuitTemplate` instance via the path to the Stuart-Landau model definition.
# We will the use the :code:`CircuitTemplate.run` method to solve the initial value problem of the above defined
# differential equations for a time interval from 0 to the given simulation time.
# This solution will be calculated numerically by a differential equation solver in the backend, starting with a
# defined step-size. Here, we use the default backend and a Runge-Kutta 2(3) solver.
# Check out the arguments of the code:`CircuitTemplate.run()` method for a detailed explanation of the
# arguments that you can use to adjust this numerical procedure.

from pyrates import CircuitTemplate, clear
import numpy as np

# define simulation time and input start and stop
T = 100.0
step_size = 1e-2

# load model template
model = CircuitTemplate.from_yaml("model_templates.oscillators.stuartlandau.sl")

# define omega
omega = 2*np.pi/12.0
model.update_var({'p/sl_op/omega': omega})
results = model.run(step_size=step_size, simulation_time=T, outputs={'x': 'p/sl_op/x'}, solver='scipy', method='RK23')

# visualize model dynamics
import matplotlib.pyplot as plt
plt.plot(results)
plt.show()

# clear results
clear(model)

# %%
#
# Step 2: Dynamics of the Van der Pol oscillator
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In a second step, lets examine the dynamics of the autonomous Van der Pol oscillator system.

# simulate model dynamics
from pyrates import integrate
results = integrate("model_templates.oscillators.vanderpol.vdp",
                    step_size=step_size, simulation_time=T, outputs={'x': 'p/vdp_op/x'},
                    solver='scipy', method='RK23', clear=True)

# visualize model dynamics
plt.plot(results)
plt.show()

# %%
#
# Step 3: Periodic forcing of the Van der Pol oscillator
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now in a final step, lets see how the Van der Pol model dynamics change in response to periodic forcing
# as generated by the Start Landau equations.

from pyrates import NodeTemplate

# define nodes
vpo = NodeTemplate.from_yaml("model_templates.oscillators.vanderpol.vdp_pop")
sl = NodeTemplate.from_yaml("model_templates.oscillators.stuartlandau.sl_pop")

# define circuit
model = CircuitTemplate(name='vpo_forced', nodes={'vpo': vpo, 'sl': sl},
                        edges=[('sl/sl_op/x', 'vpo/vdp_op/inp', None, {'weight': 10.0})])

# set omega to the value defined above
model.update_var({'sl/sl_op/omega': omega})
results = model.run(step_size=step_size, simulation_time=T, outputs={'x': 'vpo/vdp_op/x'}, solver='scipy', method='RK23')

# plot resulting phases
plt.plot(results)
plt.show()
