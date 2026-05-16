"""
Analysis of Delayed Differential Equations
==========================================

In this tutorial, you will learn how to implement and solve delayed differential equation (DDE) systems via PyRates.
DDEs take the general form

.. math::
    \\dot{\\mathbf{y}} = \\mathbf{f}(\\mathbf{y}, t, \\mathbf{y_{\tau}})

with state vector :math:`\\mathbf{y}`, vector field :math:`f`, time point :math:`t` and state history
:math:`\\mathbf{y_{\tau}} = {\\mathbf{y}(\\tau) : \\tau \\leq t}`.
Here, we will address the case of implementing and solving a DDE system with discrete delays of the form

.. math::
    \\dot{\\mathbf{y}} = \\mathbf{f}(t, \\mathbf{y}(t), \\mathbf{y}(t-\\tau)),

with discrete delay :math:`\\tau`. As a specific example, we will use a modified version of the Van der Pol oscillator
with additional delayed feedback:

.. math::
        \\dot x(t) &= z(t), \n
        \\dot z(t) &= \\mu (1-x(t)^2) z - x(t) + k x(t-\\tau),

with delayed feedback strength k. For a detailed introduction of the Van der Pol oscillator model, see the gallery
example in the model introduction section.

Below, we will show how such a DDE model can be implemented and how it can be solved using PyRates'
built-in solvers, as well as how to interface the generated DDE function with third-party tools.
"""
from pyrates import CircuitTemplate, OperatorTemplate, clear
import matplotlib.pyplot as plt

# %%
#
# Step 1: DDE definition
# ^^^^^^^^^^^^^^^^^^^^^^
#
# DDEs can be defined in two different ways in PyRates. The first way would be the usage of the `past(y, tau)` function
# call that indicates to PyRates that :math:`y(t-tau)` should be evaluated instead of :math:`y(t)`. An example of how to
# implement an `OperatorTemplate` representing the delayed Van der Pol equations is provided below.

# define parameters
k = 1.0
tau = 5.0

# define operator
eqs = [
    "x' = z",
    "z' = mu*(1-x**2)*z - x + k*past(x,tau)"
]
variables = {
    'x': 'output(0.0)',
    'z': 'variable(1.0)',
    'mu': 1.0,
    'k': k,
    'tau': tau
}
op = OperatorTemplate(name='vdp_delayed', equations=eqs, variables=variables)

# %%
# This `OperatorTemplate` could then be used to define a PyRates model (see the examples in the model introduction
# section of the gallery for a tutorial on how to do that).
#
# Alternatively, it is also possible to add an edge to a `CircuitTemplate`, including a discrete delay. This edge will
# then automatically translated into a corresponding `past` call by PyRates. Note that this only works if the edge
# source variable is a variable defined by a differential equation. So in the case of the Van der Pol model, both
# :math:`x` and :math:`z` would be valid variables for such an edge definition. Below, we show how to do that using the
# Van der Pol model that is already implemented in PyRates:

vdp = CircuitTemplate.from_yaml("model_templates.oscillators.vanderpol.vdp")
vdp.update_template(edges=[('p/vdp_op/x', 'p/vdp_op/inp', None, {'weight': k, 'delay': tau})], in_place=True)

# %%
#
# Step 2: DDE simulation
# ^^^^^^^^^^^^^^^^^^^^^^
#
# We can perform a simple simulation using a forward Euler algorithm to solve the DDE as follows:

# define simulation time
T = 50.0
step_size = 1e-2

# solve DDE via forward Euler
results = vdp.run(step_size=step_size, simulation_time=T, outputs={'x': 'p/vdp_op/x', 'z': 'p/vdp_op/z'},
                  solver='euler', in_place=False, clear=False)

# plot resulting signal
fig, ax = plt.subplots()
ax.plot(results["x"], label="x")
ax.plot(results["z"], label="z")
ax.legend()
ax.set_xlabel("time")
plt.show()

# %%
# As of PyRates 1.1.0, adaptive step-size DDE solving is available natively via the :code:`solver='scipy'` option.
# This uses a Dormand-Prince (dopri5) integrator under the hood — no third-party DDE packages required.

res_scipy = vdp.run(step_size=step_size, simulation_time=T, outputs={'x': 'p/vdp_op/x', 'z': 'p/vdp_op/z'},
                    solver='scipy', in_place=False, clear=False)

# compare fixed-step Euler vs. adaptive scipy/dopri5
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(results["x"], label="x (Euler)")
ax.plot(results["z"], label="z (Euler)")
lines = ax.get_lines()
ax.plot(res_scipy["x"], label="x (scipy/dopri5)", color=lines[0].get_color(), linestyle="dashed")
ax.plot(res_scipy["z"], label="z (scipy/dopri5)", color=lines[1].get_color(), linestyle="dashed")
ax.set_xlabel('time')
ax.legend()
plt.show()

# %%
# The dynamics agree closely; small quantitative differences arise from the discretization error of the
# fixed-step Euler method.
#
# The Julia backend additionally exposes the full suite of algorithms available in
# :code:`DifferentialEquations.jl`. To use it, pass :code:`backend='julia'` and
# :code:`solver='julia_dde'` to :code:`run`. PyRates will automatically generate a
# :code:`MethodOfSteps` wrapper and solve the problem via an adaptive algorithm (default: Tsit5).

# %%
# Inspecting the generated DDE function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# You can also call :code:`get_run_func` to write out the DDE vector-field function to a file and
# inspect it — or pass it to any other DDE solver of your choice. The generated function accepts
# a :code:`hist` callable as its third argument, through which the solver can query the state at any
# past time :math:`t - \tau`.

func, args, _, _ = vdp.get_run_func(func_name='vdp_dde_run', file_name='vdp_dde', solver='scipy',
                                    step_size=step_size, vectorize=False, backend='default', in_place=False)

f = open('vdp_dde.py', 'r')
print(f.read())
f.close()

clear(vdp)
