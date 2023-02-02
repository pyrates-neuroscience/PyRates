"""
Numerical Simulations
=====================

In this tutorial, you will learn the different options that PyRates provides for performing numerical simulations.
Given a dynamical system with state vector :math:`\\mathbf{y}`, the evolution of which is given by

.. math::
    \\dot{\\mathbf{y}} = \\mathbf{f}(\\mathbf{y}, t)

with vector field :math:`f`, we are interested in the state of the system at each time point :math:`t`.
Given an initial time :math:`t_0` and an initial state :math:`y_0`, this problem can be solved by evaluating

.. math::
    \\mathbf{y}(t) = \\int_{t_0}^{t} \\mathbf{f}(\\mathbf{y}, t') dt'.

This is known as the `initial value problem <https://en.wikipedia.org/wiki/Initial_value_problem>`_ (IVP) and there
exist various algorithms for approximating the solution to the IVP for systems where an analytic solution is intractable.
Many of these algorithms are available via PyRates, and we will demonstrate below how to use a sub-set of them.
To this end, we will solve the IVP for the QIF mean-field model, for which a detailed model introduction exists in the
model introduction gallery.
"""

# %%
#
# Numerical simulations of pre-existing models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# First, lets look at the most direct way to perform numerical simulations via PyRates - the :code:`integrate` function:

import matplotlib.pyplot as plt
from pyrates import integrate

# simulation parameters
model = "model_templates.neural_mass_models.qif.qif"
T = 80.0
dt = 1e-3
dts = 1e-2

# perform simulation
res = integrate(model, simulation_time=T, step_size=dt, sampling_step_size=dts, outputs={'r': 'p/qif_op/r'}, clear=True)

# visualize model dynamics
plt.plot(res)
plt.show()

# %%
# A numerical simulation performed that way merely returns a :code:`pandas.DataFrame` containing the resulting model
# dynamics.
#
# Customizing the model prior to simulations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Using an alternative interface to PyRates, the default model parameters can be adjusted before performing
# the simulation:

from pyrates.frontend import CircuitTemplate

# load the model
qif = CircuitTemplate.from_yaml(model)

# increase the background input of the QIF model (default value: -5.0)
qif.update_var(node_vars={'p/qif_op/eta': -2.0})

# perform the simulation
res = qif.run(simulation_time=T, step_size=dt, sampling_step_size=dts, outputs={'r': 'p/qif_op/r'}, clear=True,
              in_place=False)

# visualize the model dynamics
plt.plot(res)
plt.show()

# %%
#
# Customizing the numerical simulation settings
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Numerical simulations can be further customized. For example, extrinsic inputs can be added, which must be
# :code:`numpy.ndarray` objects. Each entry in the input array refers to the value of the input at a specific time
# point, with time steps increasing by the defined step-size.

import numpy as np

# input definition
steps = int(T/dt)
start = int(20.0/dt)
stop = int(60/dt)
inp = np.zeros((steps,))
inp[start:stop] = -3.0

# perform simulation
res = qif.run(simulation_time=T, step_size=dt, sampling_step_size=dts, outputs={'r': 'p/qif_op/r'}, clear=True,
              in_place=False, inputs={'p/qif_op/I_ext': inp}, solver='heun')

# visualize the model dynamics
plt.plot(res)
plt.show()

# %%
# Further customizations of the numerical simulation procedure are possible regarding the solving algorithm. The default
# method used in the first two examples is the standard forward `Euler <https://en.wikipedia.org/wiki/Euler_method>`_
# method, whereas the last example used `Heun's method <https://en.wikipedia.org/wiki/Heun%27s_method>`_ which adds a
# second step to Euler's method. More elaborate methods like
# `Runge-Kutta <https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods>`_ algorithms with automatic integration
# step-size adaptation are available via the scipy solver, a direct interface to the :code:`scipy.integrate.solve_ivp`
# function. All keyword arguments available for that function can also be passed to
# the :code:`integrate` function or :code:`CircuitTemplate.run` method. One of these keyword arguments is :code:`method`
# , which allows choosing between the different solving algorithms that are available via
# :code:`scipy.integrate.solve_ivp`.

# clear frontend cache
from pyrates import clear_frontend_caches
clear_frontend_caches()

# perform simulation
res = qif.run(simulation_time=T, step_size=dt, sampling_step_size=dts, outputs={'r': 'p/qif_op/r'}, clear=True,
              in_place=False, inputs={'p/qif_op/I_ext': inp}, solver='scipy', method='RK23')

# visualize the model dynamics
plt.plot(res)
plt.show()

# %%
#
# Using third-party simulation tools with PyRates models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Finally, it is also possible to write out function files that can be used in combination with various other tools that
# provide numerical integration algorithms. As an example, we demonstrate below how to interface the
# :code:`scipy.integrate.solve_ivp` method via a PyRates-generated function file for the evaluation of
# :math:`\mathbf{f}(\mathbf{y}, t)`.

from scipy.integrate import solve_ivp
clear_frontend_caches()

# generate the vector-field evaluation function file
func, args, _, _ = qif.get_run_func(func_name='f', file_name='qif_eval', step_size=dt, inputs={'p/qif_op/I_ext': inp},
                                    backend='default', solver='scipy', clear=False, in_place=True)

# read out function file
f = open('qif_eval.py', 'r')
print('')
print(f.read())
f.close()

# numerical simulation
t0, y0, *func_args = args
results = solve_ivp(func, (t0, T), y0, args=func_args)

# visualization
plt.plot(results['y'].T)
plt.show()

# remove files and cached models
qif.clear()
clear_frontend_caches()
