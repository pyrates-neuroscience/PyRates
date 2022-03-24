"""
Run Time Optimization
=====================

In this tutorial, we will go through some of the options you have to improve the run-time of vector-field evaluations
based on the equation files generated by PyRates.
As an example, we will use a model of :math:`N = 100` interconnected leaky integrator units, the evolution equations of
which are given by

.. math::
        \\dot x_i = - \\frac{x_i}{\\tau} + \\sum_{j=1}^N J_{ij} \\tanh(x_j),

with individual rates :math:`x_i`, a global time scale :math:`\\tau` and coupling weights :math:`J_{ij}`.
Below, we will construct a network of randomly coupled leaky integrators, generate its vector-field evaluation function
via PyRates and examine how fast that function evaluates.
We will test out through different backends and options for generating the function to examine how the evaluation time
is affected.
"""

# external imports
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from time import perf_counter

# pyrates imports
from pyrates import CircuitTemplate, NodeTemplate, clear

# %%
# First, lets load the model into PyRates:

# node definition
li = NodeTemplate.from_yaml("model_templates.base_templates.tanh_node")
N = 10
nodes = [f"p{i+1}" for i in range(N)]
net = CircuitTemplate(name="li_coupled", nodes={key: li for key in nodes})

# add random connections between nodes
C = np.random.uniform(low=-10.0, high=10.0, size=(N, N))
net.add_edges_from_matrix(source_var="tanh_op/m", target_var="li_op/m_in", nodes=nodes, weight=C)

# %%
# Next, we will generate the run function for that model using the default backend and no vectorization of the model
# equations.

func, args = net.get_run_func(func_name="li_eval", file_name="li_novec", step_size=1e-3, in_place=False,
                              vectorize=False)

# %%
# Now, lets define a function that allows us to calculate the average evaluation time of the generated function:

n_eval = 1000
def eval_time(func, args):

    # time the evaluation process multiple times
    times = []
    for _ in range(n_eval):
        t0 = perf_counter()
        func(*args)
        t1 = perf_counter()
        times.append(t1 - t0)

    # calculate average evaluation time
    return np.mean(times)

# %%
# Next, we can evaluate the evaluation time of the function generated by PyRates

T0 = eval_time(func, args)
print(f"Average evaluation time without vectorization: T = {T0} s.")
clear(net)

# %%
# Let's compare this against the evaluation time for the vectorized model:

# generate the function
func, args = net.get_run_func(func_name="li_eval", file_name="li_vec", step_size=1e-3, in_place=False,
                              vectorize=True)

# calculate average evaluation time
T1 = eval_time(func, args)
print(f"Average evaluation time with vectorization: T = {T1} s.")
clear(net)

# %%
# As you can see, vectorization sped up the evaluation by nearly an order of magnitude!
# As an additional option for optimizing the evaluation time, we can try out adding function decorators.
# The function decorator :code:`njit` from the Python toolbox `Numba <https://numba.pydata.org/>`_ will translate our
# evaluation function into optimized machine code during runtime based on the LLVM compiler library ().
# Let's see, how this helps with our vectorized and our non-vectorized models:

# njit optimization of non-vectorized model
func, args = net.get_run_func(func_name="li_eval", file_name="li_novec_numba", step_size=1e-3, in_place=False,
                              vectorize=False, decorator=njit)
func(*args)
T2 = eval_time(func, args)
print(f"Average evaluation time of numba-optimized model without vectorization: T = {T2} s.")
clear(net)

# njit optimization of non-vectorized model
func, args = net.get_run_func(func_name="li_eval", file_name="li_vec_numba", step_size=1e-3, in_place=False,
                              vectorize=True, decorator=njit)
func(*args)
T3 = eval_time(func, args)
print(f"Average evaluation time of numba-optimized model without vectorization: T = {T3} s.")
clear(net)

# %%
# While the non-vectorized model was not sped up a lot, we were cut the evaluation time of the vectorized model in half.
# Further options that can be tried out to improve evaluation times involve different choices of backends.
# The default backend chosen here employs :code:`numpy` to implement the model equations.
# For additional backend options, see the documentation of
# the :code:`pyrates.frontend.template.circuit.CircuitTemplate.run` method available at
# `readthedocs <https://pyrates.readthedocs.io/en/latest/frontend.template.html>`_.