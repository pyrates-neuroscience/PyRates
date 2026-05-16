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

Below, we will show three ways to define a DDE in PyRates, and how to solve it using the built-in
fixed-step (Euler) and adaptive (scipy/dopri5) solvers, the Julia :code:`DifferentialEquations.jl`
backend, and how to interface the generated function with third-party tools.

As of PyRates 1.1.0, DDEs can also be written using the compact :math:`x(t-\\tau)` notation directly
inside equation strings, which PyRates automatically rewrites to :code:`past(x, tau)` before parsing.
"""
from pyrates import CircuitTemplate, OperatorTemplate, clear
import matplotlib.pyplot as plt

# %%
#
# Step 1: DDE definition
# ^^^^^^^^^^^^^^^^^^^^^^
#
# DDEs can be defined in three different ways in PyRates. The first is the :code:`past(y, tau)` function
# call, which explicitly tells PyRates to evaluate :math:`y(t-\tau)` instead of :math:`y(t)`. An example
# :code:`OperatorTemplate` for the delayed Van der Pol model is shown below.

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
# **New in PyRates 1.1.0** — A second, more compact way to write the same DDE is to use the
# :code:`x(t-tau)` notation directly in the equation string. PyRates automatically rewrites any
# expression of the form :code:`varname(t-delay)` to :code:`past(varname, delay)` before parsing,
# so both forms produce identical models. The delayed Van der Pol equations become:

eqs_compact = [
    "x' = z",
    "z' = mu*(1-x**2)*z - x + k*x(t-tau)"
]
op_compact = OperatorTemplate(name='vdp_compact', equations=eqs_compact, variables=variables)

# %%
# This shorthand is purely syntactic sugar — the same :code:`DDEHistory` mechanism and all solvers
# apply equally to both forms. It is especially convenient when transcribing mathematical DDE notation
# straight into code without mentally replacing every :math:`x(t-\tau)` with :code:`past(x, tau)`.
#
# A third way to introduce a delay is to add an edge to a :code:`CircuitTemplate` with a discrete
# :code:`delay` attribute. PyRates translates this edge into a corresponding :code:`past` call
# automatically. Note that this only works when the edge source variable is defined by a differential
# equation (both :math:`x` and :math:`z` qualify in the Van der Pol model). Below we use the
# pre-implemented Van der Pol template to show this approach:

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

# %%
# Step 3: Julia backend — DifferentialEquations.jl
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# If the Julia backend is installed (:code:`pip install pyrates[backends]` pulls in the Julia bindings),
# PyRates exposes the full suite of adaptive DDE solvers available in :code:`DifferentialEquations.jl`.
# The call is identical to the scipy case — only :code:`backend` and :code:`solver` change.
# PyRates automatically generates a :code:`MethodOfSteps` wrapper and solves via Tsit5 by default:
#
# .. code-block:: python
#
#     res_julia = vdp.run(
#         step_size=step_size, simulation_time=T,
#         outputs={'x': 'p/vdp_op/x', 'z': 'p/vdp_op/z'},
#         backend='julia', solver='julia_dde',
#         in_place=False, clear=False,
#         julia_path='julia',   # path to the Julia executable; adjust if not on PATH
#     )
#
# A different adaptive algorithm can be requested via the :code:`method` keyword, e.g.
# :code:`method='RosShamp4'` for stiff systems.  PyRates passes the list of known constant delays
# to :code:`DDEProblem` as :code:`constant_lags` automatically, enabling discontinuity tracking.
#
# The result is a :code:`pandas.DataFrame` with the same structure as the scipy/Euler outputs, so
# all subsequent analysis and plotting code is backend-agnostic.

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
