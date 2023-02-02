"""
Non-linear Oscillator Entrainment
=================================

In this tutorial, we will examine entrainment of a
`Van der Pol oscillator (VPO) <https://en.wikipedia.org/wiki/Van_der_Pol_oscillator>`_ that receives oscillatory input
from a simple `Kuramoto oscillator (KO) <https://en.wikipedia.org/wiki/Kuramoto_model>`_.
To this end, we will use the :code:`pyrates.integrate()` the :code:`pyrates.grid_search()` functions from pyrates to
generate time series of the VPO dynamics and time series analysis methods from scipy to quantify the entrainment.

To learn more about the Van der Pol oscillator model, have a look at the respective gallery example in
*Model Introductions*.

"""

# %%
# Step 1: Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# First, let's define the model:

import matplotlib.pyplot as plt
import numpy as np
from pyrates import CircuitTemplate, NodeTemplate

# define network nodes
VPO = NodeTemplate.from_yaml("model_templates.oscillators.vanderpol.vdp_pop")
KO = NodeTemplate.from_yaml("model_templates.oscillators.kuramoto.sin_pop")
nodes = {'VPO': VPO, 'KO': KO}

# define network edges
edges = [('KO/sin_op/s', 'VPO/vdp_op/inp', None, {'weight': 1.0})]

# define network
net = CircuitTemplate(name="VPO_forced", nodes=nodes, edges=edges)

# %%
# This defines a simple VPO model that receives periodic input from a KO, where we are interested in the entrainment of the
# former to the intrinsic frequency of the latter.
#
# Step 2: Simulation of the oscillator dynamics
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Lets first have a look at the model dynamics for a driving frequency
# of :math:`\omega = 0.5`:

# change the intrinsic KO frequency
omega = 0.5
net.update_var(node_vars={'KO/phase_op/omega': omega, 'VPO/vdp_op/mu': 2.0})

# simulate the model dynamics
T = 1100.0
dt = 1e-3
dts = 1e-2
cutoff = 100.0
res = net.run(simulation_time=T, step_size=dt, solver='scipy', method='DOP853',
              outputs={'VPO': 'VPO/vdp_op/x', 'KO': 'KO/phase_op/theta'},
              in_place=False, cutoff=cutoff, sampling_step_size=dts)

# extract phases from signals
from scipy.signal import hilbert, butter, sosfilt
def get_phase(signal, N, freqs, fs):
    filt = butter(N, freqs, output="sos", btype='bandpass', fs=fs)
    s_filtered = sosfilt(filt, signal)
    return np.unwrap(np.angle(hilbert(s_filtered)))

p1 = np.sin(get_phase(res['VPO'].values, N=10, freqs=(omega-0.3*omega, omega+0.3*omega),
                      fs=1/dts))
p2 = np.sin(2*np.pi*res['KO'].values)

# plot the results
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(p1[90000:])
ax.plot(p2[90000:])
plt.legend(list(res.columns.values))
ax.set_xlabel('timestep #')
ax.set_ylabel('Output')
plt.show()

# %%
# We can see from these dynamics that :code:`VPO` does not entrain to the intrinsic
# frequency of :code:`KO` for that particular parametrization of the model. Instead, both oscillators
# express periodic activity close to their intrinsic frequency.
#
# Step 3: Coherence calculation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# As a next step, lets calculate a metric called `coherence <https://en.wikipedia.org/wiki/Coherence_(physics)>`_,
# which can be used to quantify the entrainment of :code:`VPO` to the intrinsic frequency of :code:`KO`.
# The coherence :math:`C` between :code:`VPO` and :code:`KO` is given by:
#
# .. math::
#
#         C = \frac{|P_{x\theta}|}{P_{x} P_{\theta}},
#
# where :math:`P_{x}` and :math:`P_{\theta}` are the power spectral densities of :code:`VPO` and :code:`KO`,
# respectively, and :math:`P_{x\theta}` is the cross-spectral density between the two.
# This measure can be easily calculated using the :code:`scipy.signal.coherence` method,
# which uses `Welch's method <https://en.wikipedia.org/wiki/Welch%27s_method>`_ for the spectral density estimation:

from scipy.signal import coherence

# calculate the coherence
nps = 1024
window = 'hamming'
freq, coh = coherence(np.sin(p1.squeeze()), p2.squeeze(), fs=1/dts, nperseg=nps, window=window)

# plot the coherence as a function of the frequency of interest
fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.plot(freq[freq < 2], coh[freq < 2])
ax2.set_xlabel('frequency in Hz')
ax2.set_ylabel('C')
plt.show()

# %%
# As we can see, the coherence at the input frequency (0.5 Hz) is :math:`C \approx 0.15`
# for this set of parameters.
#
# Step 4: Parallelized parameter sweep
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In a next step, we'll examine how this changes as we
# systematically alter the input frequency :math:`\omega` as well as the input
# strength :math:`J`.
# Again, we will first simulate the model dynamics for each parameter set, using the
# :code:`pyrates.grid_search` function:

# define parameter sweep
n_om = 20
n_J = 20
omegas = np.linspace(0.3, 0.5, num=n_om)
weights = np.linspace(0.0, 2.0, num=n_J)

# map sweep parameters to network parameters
params = {'omega': omegas, 'J': weights}
param_map = {'omega': {'vars': ['phase_op/omega'], 'nodes': ['KO']},
             'J': {'vars': ['weight'], 'edges': [('KO/sin_op/s', 'VPO/vdp_op/inp')]}}

# perform parameter sweep
from pyrates import grid_search
results, res_map = grid_search(circuit_template=net, param_grid=params, param_map=param_map,
                               simulation_time=T, step_size=dt, solver='scipy', method='DOP853',
                               outputs={'VPO': 'VPO/vdp_op/x', 'KO': 'KO/phase_op/theta'},
                               inputs=None, vectorize=True, clear=False, file_name='vpo_forced',
                               permute_grid=True, cutoff=cutoff, sampling_step_size=dts)

# %%
# The :code:`pyrates.grid_search` function automatically creates a vectorized network
# representation that allows to simulate the dynamics of all the different model parameterizations
# in parallel. We added a few optional keyword arguments here to visualize this. Below
# we print the python file that has been generated by PyRates to perform the simulation and that contains
# the vectorized network equations:

import os
f = open('vpo_forced.py', 'r')
print('')
print(f.read())
f.close()
os.remove('vpo_forced.py')

# %%
# As you can see, all KOs have been grouped together into a single vector, allowing to evaluate their
# evolution equations in a single line of code.
#
# But lets get back to the problem at hand: Having obtained the time series, we will now
# calculate the coherence for each model parameterization.

# calculate and store coherences
coherences = np.zeros((n_J, n_om))
for key in res_map.index:

    # extract parameter set
    omega = res_map.at[key, 'omega']
    J = res_map.at[key, 'J']

    # collect phases
    tf = np.maximum(0.01, freq[np.argmin(np.abs(omegas - omega))])
    p1 = np.sin(get_phase(results['VPO'][key].squeeze(), N=10,
                          freqs=(tf-0.3*tf, tf+0.3*tf), fs=1/dts))
    p2 = np.sin(2 * np.pi * results['KO'][key].squeeze())

    # calculate coherence
    freq, coh = coherence(p1, p2, fs=1/dts, nperseg=nps, window=window)

    # find coherence matrix position that corresponds to these parameters
    idx_r = np.argmin(np.abs(weights - J))
    idx_c = np.argmin(np.abs(omegas - omega))

    # store coherence value at driving frequency
    tf = freq[np.argmin(np.abs(freq - omega))]
    coherences[idx_r, idx_c] = np.max(coh[(freq >= tf-0.3*tf) * (freq <= tf+0.3*tf)])

# plot the coherence at the driving frequency for each pair of omega and J
fix, ax = plt.subplots(figsize=(12, 8))
cax = ax.imshow(coherences[::-1, :], aspect='equal')
ax.set_xlabel(r'$\omega$')
ax.set_ylabel(r'$J$')
ax.set_xticks(np.arange(0, n_om, 3))
ax.set_yticks(np.arange(0, n_J, 3))
ax.set_xticklabels(np.round(omegas[::3], decimals=2))
ax.set_yticklabels(np.round(weights[::-3], decimals=2))
plt.title("Coherence between VPO and KO")
plt.colorbar(cax)
plt.show()
