"""
Entrainment
===========

In this tutorial, we will examine entrainment of a Kuramoto oscillator (KO) that receives oscillatory input from another
KO. To this end, we will use the :code:`pyrates.integrate()` the :code:`pyrates.grid_search()` functions from pyrates to
generate time series of the KO dynamics and time series analysis methods from scipy to quantify the entrainment.

To learn more about the Kuramoto oscillator model, have a look at the respective gallery example in
*Model Introductions*.

"""

# %%
# First, let's define the model:

from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate
import matplotlib.pyplot as plt
import numpy as np

# define network nodes
k = NodeTemplate.from_yaml("model_templates.coupled_oscillators.kuramoto.phase_pop")
k_in = NodeTemplate.from_yaml("model_templates.coupled_oscillators.kuramoto.phase_pop")
nodes = {'k1': k, 'k2': k_in}

# define network edges
edge_op = EdgeTemplate.from_yaml("model_templates.coupled_oscillators.kuramoto.sin_edge")
edges = [('k2/phase_op/theta', 'k1/phase_op/net_in', edge_op, {'weight': 1.0})]

# define network
net = CircuitTemplate(name="ko_entrainment", nodes=nodes, edges=edges)

# %%
# This defines a simple KO model with an input KO and a target KO, where we are interested in the entrainment of the
# latter to the intrinsic frequency of the former. Lets first have a look at the model dynamics for two different
# intrinsic frequencies :math:`\omega_1` and :math:`\omega_2`:

# change the intrinsic KO frequencies
net.update_var(node_vars={'k1/phase_op/omega': 10.0, 'k2/phase_op/omega': 14.0})

# simulate the model dynamics
T = 10.0
dt = 1e-3
res = net.run(simulation_time=T, step_size=dt, solver='scipy', method='RK23',
              outputs={'p1': 'k1/phase_op/theta', 'p2': 'k2/phase_op/theta'},
              in_place=False)

# map the raw phases to the unit circle
phases = np.sin(res*2*np.pi)

# plot the results
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(phases.iloc[1000:2000, :])
plt.legend(list(res.columns.values))
ax.set_xlabel('timestep #')
ax.set_ylabel('KO Phase')
plt.show()

# %%
# We can see from these dynamics that :code:`k1` does not entrain to the intrinsic
# frequency of :code:`k2` for that particular parametrization of the model. Instead, both oscillators
# express periodic activity close to their intrinsic frequency.
#
# As a next step, lets calculate a metric called *coherence*, which can be used to quantify the
# entrainment of :code:`k1` to the intrinsic frequency of :code:`k2`. The coherence :math:`C_12` between :code:`k2`
# and :code:`k1` is given by:
# .. math::
#         C_{12} = \frac{|P_{12}|}{P_{11} P_{22}},
#
# where :math:`P_{11}` and :math:`P_{22}` are the power spectral densities of :code:`k1` and :code:`k2`, respectively,
# and :math:`P_{12}` is the cross-spectral density between the two. This measure can be easily calculated using the
# :code:`scipy.signal.coherence` method, which uses Welch's method for the spectral density estimation:

from scipy.signal import coherence

# calculate the coherence
freq, coh = coherence(phases['p1'].squeeze(), phases['p2'].squeeze(), fs=1/dt)

# plot the coherence as a function of the frequency of interest
fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.plot(freq[freq < 40], coh[freq < 40])
ax2.set_xlabel('frequency in Hz')
ax2.set_ylabel('C')
plt.show()

# %%
# As we can see, the coherence at the input frequency (20 Hz) is around :math:'C_{12} = 0.05'
# for this set of parameters. In a next step, we'll examine how this changes as we
# systematically alter the input frequency :math:`\omega_2` as well as the input
# strength, which is given by the weight of the edge between the two KOs.
# We will label the latter :math:`J_12` to avoid confusion.
# Again, we will first simulate the model dynamics for each parameter set, using the
# :code:`pyrates.grid_search` function:

# define parameter sweep
n_om = 10
n_wei = 10
omegas = np.linspace(5, 15, num=n_om)
weights = np.linspace(1.0, 10.0, num=n_om)

# map sweep parameters to network parameters
params = {'omega_2': omegas, 'J_12': weights}
param_map = {'omega_2': {'vars': ['phase_op/omega'], 'nodes': ['k2']},
             'J_12': {'vars': ['weight'],
                      'edges': [('k2/phase_op/theta', 'k1/phase_op/net_in')]}}

# perform parameter sweep
from pyrates import grid_search
results = grid_search(circuit_template=net, param_grid=params, param_map=param_map,
                      simulation_time=T, step_size=dt, solver='scipy', method='RK23',
                      outputs={'p1': 'k1/phase_op/theta', 'p2': 'k2/phase_op/theta'},
                      inputs=None, vectorize=True, clear=False, file_name='kuramoto_entrainment',
                      permute_grid=True)

# %%
# The :code:`pyrates.grid_search` function automatically creates a vectorized network
# representation that allows to simulate the dynamics of all the different KO model parameterizations
# in parallel. We added a few optional keyword arguments here to visualize this. Below
# we print the python file that has been generated by PyRates to perform the simulation and that contains
# the vectorized network equations:

f = open('kuramoto_entrainment.py', 'r')
print('')
print(f.read())

# %%
# As you can see, all KOs have been grouped together into a single vector, allowing to evaluate their
# evolution equations in a single line of code.
#
# But lets get back to the problem at hand: Having obtained the time series, we will now
# calculate the coherence for each model parameterization.

