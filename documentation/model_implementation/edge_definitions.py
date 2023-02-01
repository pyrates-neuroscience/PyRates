r"""
Edge Definitions
==============

In this tutorial, we will go through the different options you have to define edge templates, i.e. formulations of the
dynamics of signal propagation along the edges in a network.
As an example, we will use a model of :math:`N = 2` interconnected leaky integrator units, the evolution
equations of which are given by

.. math::
        \dot r_i = - \frac{(r_0 - r_i)}{\tau} + u(t) + \sum_{j=1}^N J_{ij} f(r_j),

with individual rates :math:`r_i`, a global time scale :math:`\tau`, extrinsic input :math:`u(t)`,
coupling weights :math:`J_{ij}` and a coupling function :math:`f`.
The latter represents the signal transformation that can happen during signal propagation along an edge
Below, we will go through the options you have to specify :math:`f` in PyRates.
"""

# %%
# As preparation, lets import all required packages and load the leaky integrator model into PyRates.

# external imports
import numpy as np
import matplotlib.pyplot as plt

# pyrates imports
from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate, OperatorTemplate, clear

# node definition
li = NodeTemplate.from_yaml("model_templates.base_templates.li_node")

# %%
# The node template `li` includes the leaky-integrator equation, but we have not specified :math:`f` yet. For this, we
# can use an `EdgeTemplate`. Below, we will go through several choices of edge templates.
#
# Example 1: A simple instantaneous transform
# ------------------------------------------
#
# We start with a simple instantaneous transform, i.e. a function :math:`f` that does not depend on time.
# Aas an example, we will use the `hyperbolic tangent <https://en.wikipedia.org/wiki/Hyperbolic_functions>`_
# as our coupling function.

tanh_op = OperatorTemplate(name="tanh_op", equations="m = tanh(x)",
                           variables={"m": "output", "x": "input"})
tanh_edge = EdgeTemplate(name="tanh_edge", operators=[tanh_op])

# %%
# In the above code line, we first created an operator template that implements the hyperbolic tangent.
# We then used this operator template as the single operator from which to create an edge template, thus defining
# :math:`f = tanh` as our coupling function.
# Now, we can use this edge template to define a circuit:

tanh_net = CircuitTemplate(name="tanh_net", nodes={"li1": li, "li2": li},
                           edges=[("li1/li_op/r", "li2/li_op/m_in", tanh_edge, {"weight": 5.0})])

# %%
# This circuit contains a single edge from node `li1` to node `li2`, which uses the `tanh_edge` as its coupling
# function. To make sure the edge works, lets perform a numerical simulation, where we ramp up the extrinsic input
# :math:`u(t)` to `li1`. We would expect (1) the input to continuously increase :math:`r_1`, (2) that increases in
# :math:`r_1` lead to an increase in :math:`r_2` due to the coupling, and (3) that increases in :math:`r_2` will
# eventually hit a ceiling due to the hyperbolic tangent being the coupling function. Let's see if our expectations are
# met:

T = 100.0
dt = 1e-2
inp = np.linspace(0.0, 10.0, int(T/dt))

res = tanh_net.run(T, dt, inputs={"li1/li_op/u": inp}, outputs={"r1": "li1/li_op/r", "r2": "li2/li_op/r"},
                   solver="scipy", method="RK23")

res.plot()
plt.show()

# %%
# As we can see, our expectations were all met.
#
# Example 2: A kernel convolution
# -------------------------------
#
# Even though we used a very simple coupling function above, edge templates can be made just as complex as node
# templates. In this example, we will use a time-dependent coupling function :math:`f(r, t)`.
# A typical example for signal propagation in biophysical systems is the convolution of the source signal of an edge
# with an impulse response function (or response kernel).
# Here, we choose a convolution with an alpha kernel by defining
#
# .. math::
#       f(r, t) = \int_0^{t} = \frac{t-t'}{\tau_{\alpha}^2} \exp(\frac{t-t'}{\tau_{\alpha}}) r(t') dt',
#
# with alpha kernel time constant :math:`\tau_{\alpha}`.
# This convolution integral can be solved analytically, yielding the following set of coupled differential equations:
#
# .. math::
#       \tau_{\alpha} \dot x = y, \\
#       \tau_{\alpha} \dot y = -2y - x + \tau_{\alpha} r.
#
# Thus, we can implement the alpha kernel convolution via an operator governed by these two differential equations:

# set up alpha convolution operator
alpha_op = OperatorTemplate(name="alpha_op",
                            equations=["x' = z/tau",
                                       "z' = r_in - (2*z + x)/tau"],
                            variables={"x": "output", "z": "variable", "tau": 10.0, "r_in": "input"})

# set up edge with alpha convolution operator
alpha_edge = EdgeTemplate(name="alpha_edge", operators=[alpha_op])

# create circuit with alpha edge
alpha_net = CircuitTemplate(name="alpha_net", nodes={"li1": li, "li2": li},
                            edges=[("li1/li_op/r", "li2/li_op/m_in", alpha_edge, {"weight": 1.0})])

# %%
# Since the alpha kernel convolution does not have the same ceiling effect as the hyperbolic tangent,
# we expect that :math:`r_2` increases continuously when we use the same simulation setup as previously.
# However, the increase should happen more slowly than before, due to the slow time constant of
# :math:`\tau_{\alpha} = 10` that we defined.

res = alpha_net.run(T, dt, inputs={"li1/li_op/u": inp}, outputs={"r1": "li1/li_op/r", "r2": "li2/li_op/r"},
                    solver="scipy", method="RK23")

res.plot()
plt.show()

# %%
# Indeed, we see that the increase in :math:`r_2` is considerably delayed in comparison to the response of :math:`r_1`.
# For more options to implement delay coupling in PyRates, see the example on `delay coupling` in the model definition
# section.
#
# Example 3: Two-operator edge
# ----------------------------
#
# In both examples above, we used a rather simple edge template with just a single operator template. However,
# just as with node templates, it is possible to combine multiple operator templates into a single edge template.
# To demonstrate this, we will simply combine add a hyperbolic tangent operator to the output of the alpha kernel
# convolution operator.

combined_edge = EdgeTemplate(name="comb_edge", operators=[alpha_op, tanh_op])

# %%
# We can just the previously defined operators, since we specified :math:`x` as the output variable of `alpha_op`
# and as the input variable of `tanh_op`. Thus, PyRates will know that it should feed the output of the former to the
# input of the latter.

combined_net = CircuitTemplate(name="alpha_net", nodes={"li1": li, "li2": li},
                               edges=[("li1/li_op/r", "li2/li_op/m_in", combined_edge, {"weight": 5.0})])

# %%
# The circuit with the combined edge should now yield both effects: A ceiling effect due to the hyperbolic tangent as
# well as a delayed response due to the alpha kernel.

res = combined_net.run(T, dt, inputs={"li1/li_op/u": inp}, outputs={"r1": "li1/li_op/r", "r2": "li2/li_op/r"},
                       solver="scipy", method="RK23")

res.plot()
plt.show()

# %%
# ...which is what we see as a result!
#
# Example 4: Edges with multiple inputs
# -------------------------------------
#
# As a final example, we will demonstrate how to define edges that receive inputs from multiple sources.
# A common example for this is the Kuramoto oscillator, where the sinusoidal coupling function depends on difference
# between the phases of the coupled oscillators.
# Similarly, we will adapt the edge from example 2 to depend on the difference :math:`r_1 - r2` rather than just
# :math:`r_1`:

# set up difference-dependent operator
diff_op = OperatorTemplate(name="diff_op",
                           equations=["x' = z/tau",
                                      "z' = r_s - r_t - (2*z + x)/tau"],
                           variables={"x": "output", "z": "variable", "tau": 10.0, "r_s": "input", "r_t": "input"})

# set up edge with the difference-dependent operator
diff_edge = EdgeTemplate(name="diff_edge", operators=[diff_op])

# create circuit where multiple inputs have to be mapped to the edge template
diff_net = CircuitTemplate(name="alpha_net", nodes={"li1": li, "li2": li},
                           edges=[("li1/li_op/r", "li2/li_op/m_in", diff_edge,
                                   {"weight": 1.0,
                                    "diff_edge/diff_op/r_s": "source",
                                    "diff_edge/diff_op/r_t": "li2/li_op/r"})]
                           )

# %%
# As demonstrated above, multiple inputs to an edge template can simply be resolved by specifying which source variable
# to use for each input in the edge attributes dictionary. Note that `source` can be used as a keyword to refer to the
# source variable of the edge (in this case `li1/li_op/r`).
# Since this edge only propagates the difference between :math:`r_1` and :math:`r_2` and only :math:`r_1` receives
# extrinsic input, we expect that :math:`r_2 < r_1 \forall t`.

res = diff_net.run(T, dt, inputs={"li1/li_op/u": inp}, outputs={"r1": "li1/li_op/r", "r2": "li2/li_op/r"},
                   solver="scipy", method="RK23")

res.plot()
plt.show()
