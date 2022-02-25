"""
Delay Coupling
==============

In this tutorial, we will go through the different options you have to define delay-coupled dynamical systems in
PyRates. As an example, we will use a model of :math:`N = 2` interconnected leaky integrator units, the evolution
equations of which are given by

.. math::
        \\dot x_i = - \\frac{x_i}{\\tau} + \\sum_{j=1}^N J_{ij} \\tanh(x_j),

with individual rates :math:`x_i`, a global time scale :math:`\\tau` and coupling weights :math:`J_{ij}`.
This model does not include any delays in the coupling yet.
Below, we will go through the two main options for introducing delay-coupling to this model.

References
----------
.. [1] H. Smith. (2011) *An Introduction to Delay Differential Equations with Applications to the Life Sciences.*
       Springer, New York.
.. [2] R. Gast, R. Gong, H. Schmidt, H.G.E. Meijer, T.R. Knoesche (2021) *On the Role of Arkypallidal and Prototypical
       Neurons for Phase Transitions in the External Pallidum.* Journal of Neuroscience 41(31): 6673-6683.
"""

# %%
# As preparation, lets import all required packages and load the leaky integrator model into PyRates.

# external imports
import numpy as np
import matplotlib.pyplot as plt

# pyrates imports
from pyrates import CircuitTemplate, NodeTemplate, clear

# node definition
li = NodeTemplate.from_yaml("model_templates.base_templates.tanh_node")

# %%
# To get an idea of how the model behaves without delays, lets connect two leaky integrators into a circuit and simulate
# its dynamics.

# define connection weights
J_21 = 5.0
J_12 = -5.0

# define circuit model
net = CircuitTemplate(name="nodelays", nodes={"p1": li, "p2": li},
                      edges=[("p1/tanh_op/m", "p2/li_op/m_in", None, {"weight": J_21}),
                             ("p2/tanh_op/m", "p1/li_op/m_in", None, {"weight": J_12})])

# define simulation parameters
T = 10.0
dt = 1e-5
dts = 1e-3
times = np.linspace(0, T, int(np.round(T/dt)))
inp = 1.0/(1.0 + np.exp(10.0*np.sin(2.0*np.pi*0.7*times)))

# plot input
plt.plot(times, inp)
plt.show()

# perform simulation
res = net.run(simulation_time=T, step_size=dt, sampling_step_size=dts, vectorize=True, in_place=False,
              inputs={"p1/li_op/u": inp}, outputs={"p1": "p1/li_op/r", "p2": "p2/li_op/r"})

# plot the results
plt.plot(res)
plt.legend(res.columns.values)
plt.show()

clear(net)

# %%
# We can see that the system dynamics mostly follow the periodic input. Lets see how they are affected by the
# introduction of delays.
#
# Option 1: Using Delayed Differential Equations
# ==============================================
#
# PyRates allows for the implementation of delayed differential equation (DDE) systems. In the case of our exemplary
# model, we would like to implement a leaky integrator system with the following evolution equations:
#
# .. math::
#
#         \\dot x_i(t) = - \\frac{x_i(t)}{\\tau} + \\sum_{j=1}^N J_{ij} \\tanh(x_j(t-d_{ij})),
#
# where :math:`d_{ij}` are scalar delays specific for the connection from :math:`j` to :math:`i`.
# This can be achieved by using the :code:`delay` keyword to define :math:`d_{ij}` for an edge.

# define delays
d_21 = 0.2
d_12 = 0.4

# define circuit model with discrete delays
net = CircuitTemplate(name="dde", nodes={"p1": li, "p2": li},
                      edges=[("p1/tanh_op/m", "p2/li_op/m_in", None, {"weight": J_21, "delay": d_21}),
                             ("p2/tanh_op/m", "p1/li_op/m_in", None, {"weight": J_12, "delay": d_12})]
                      )

# perform simulation
res = net.run(simulation_time=T, step_size=dt, sampling_step_size=dts, vectorize=True, in_place=False,
              inputs={"p1/li_op/u": inp}, outputs={"p1": "p1/li_op/r", "p2": "p2/li_op/r"})

# plot the results
plt.plot(res)
plt.legend(res.columns.values)
plt.show()

clear(net)

# %%
# It seems like the addition of discrete delays led to the emergence of a stable periodic solution, the
# frequency of which differs from the driving frequency.
# Note that the definition of DDEs does not permit the use of adaptive step-size solvers at the moment.
#
# Option 2: Using Distributed Delays
# ==================================
#
# As an alternative, you can define distributed delays in PyRates as convolutions of the source variable of an edge with
# a gamma kernel. This procedure is explained in detail in [1]_ and has been applied to a system of coupled QIF neurons
# in [2]_. In our specific example, this would change the model equations as follows
#
# .. math::
#
#         \\dot x_i = - \\frac{x_i}{\\tau} + \\sum_{j=1}^N J_{ij} \\tanh(\\Gamma_{ij} * x_j),
#         \\Gamma_{ij}(t) = \\frac{a_{ij}^{b_{ij}} t^{b_{ij}-1} e^{a_{ij}t}}{(b_{ij}-1)!},
#
# where :math:`*` is the convolution operator and :math:`a_{ij}` and :math:`b_{ij}` are the parameters of the gamma
# kernel :math:`\\Gamma_{ij}`. In PyRates, such a convolution can simply be added by specifying an additional keyword
# :code:`spread` for a given edge definition:

# define variances
v_21 = 0.1
v_12 = 0.1

# define circuit model with distributed delays
net = CircuitTemplate(name="gamma", nodes={"p1": li, "p2": li},
                      edges=[("p1/tanh_op/m", "p2/li_op/m_in", None, {"weight": J_21, "delay": d_21, "spread": v_21}),
                             ("p2/tanh_op/m", "p1/li_op/m_in", None, {"weight": J_12, "delay": d_12, "spread": v_12})]
                      )

# %%
# In that case, :code:`delay` and :code:`spread` are interpreted as the mean :math:`\\mu` and variance :math:`\\sigma^2`
# of the gamma kernel that the source variable should be convoluted with, respectively.
# These quantities are related to :math:`a` and :math:`b` via :math:`\\mu = \\frac{a}{b}` and
# :math:`\\sigma^2 = \\frac{a}{b^2}`. In addition, PyRates automatically uses the linear chain trick described in [1]_
# and [2]_ to translate the convolution operation into a set of coupled ODEs that will be added to the model equations.
# Let's see how these changes to our model affected the dynamics.

# perform simulation
res = net.run(simulation_time=T, step_size=dt, sampling_step_size=dts, vectorize=True, in_place=False,
              inputs={"p1/li_op/u": inp}, outputs={"p1": "p1/li_op/r", "p2": "p2/li_op/r"}, solver="scipy")

# plot the results
plt.plot(res)
plt.legend(res.columns.values)
plt.show()

clear(net)

# %%
# We can see that the period of the periodic solution of the model dynamics were further slowed down by the
# addition of distributed delays in comparison to scalar delays.
