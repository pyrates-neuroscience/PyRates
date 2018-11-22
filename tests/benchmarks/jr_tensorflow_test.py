"""Runs JRC with tensorflow on population basis
"""

import tensorflow as tf
from networkx import MultiDiGraph
from pyrates.backend import ComputeGraph, Network
from matplotlib.pyplot import *

# Comment this out for making GPU available for Tensorflow.
###########################################################
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
##########################################################

# parameter definition
######################

#np.random.seed(4)

# general
step_size = 1e-3
simulation_time = 1.0
n_steps = int(simulation_time / step_size)

# Connection Percentage (If Low that means Connections are few!!)
sparseness_e = 0.
sparseness_i = sparseness_e * 0.

# No_of_JansenRitCircuit
n_jrcs = 1

# connectivity parameters
c_intra = 135.
c_inter = 0.

# No of nodes triple the circuit size.
n_nodes = int(n_jrcs * 3)

# synapse parameters
H = []
tau = []

for _ in range(n_jrcs):
    H += [[3.25e-3, -22e-3],
          [3.25e-3, 0.],
          [3.25e-3, 0.]]
    tau += [[10e-3, 20e-3],
            [10e-3, 1.],
            [10e-3, 1.]]

C_e = np.zeros((n_nodes, n_nodes))
C_i = np.zeros((n_nodes, n_nodes))

# inter-circuit connectivity
for i in range(n_jrcs):
    for j in range(n_jrcs - 1):
        if i != j:
            weight_e = np.random.uniform()
            if weight_e > (1 - sparseness_e):
                C_e[i * 3, j * 3] = np.random.uniform()
            weight_i = np.random.uniform()
            if weight_i > (1 - sparseness_i):
                C_e[i * 3 + 2, j * 3] = np.random.uniform()

for i in range(n_nodes):
    ce_sum = np.sum(C_e[i, :])
    if ce_sum > 0:
        C_e[i, :] /= ce_sum
C_e *= c_inter

# intra-circuit connectivity
for i in range(n_jrcs):
    C_e[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = np.array([[0., 0.8 * c_intra, 0.],
                                                          [1.0 * c_intra, 0., 0.],
                                                          [0.25 * c_intra, 0., 0.]])
    C_i[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = np.array([[0., 0., 0.25 * c_intra],
                                                          [0., 0., 0.],
                                                          [0., 0., 0.]])

# define backend dictionary
###########################

graph = MultiDiGraph()
for i in range(0, n_jrcs):
    data = {'operators': {'RPO_e_pc.0': {
                               'equations': ["d/dt * x = H/tau * (m_in + u) - 2. * 1./tau * x - (1./tau)^2. * psp",
                                             "d/dt * psp = x"],
                               'inputs': {},
                               'output': 'psp'},
                          'RPO_i_pc.0': {
                              'equations': ["d/dt * x = H/tau * (m_in + u) - 2. * 1./tau * x - (1./tau)^2. * psp",
                                            "d/dt * psp = x"],
                              'inputs': {},
                              'output': 'psp'},
                          'PRO.0': {
                              'equations': ["m_out = m_max / (1. + exp(r * (v_th - psp)))"],
                              'inputs': {'psp': {'sources': ['RPO_e_pc.0', 'RPO_i_pc.0'],
                                                 'reduce_dim': True}},
                              'output': 'm_out'},
                          },
            'operator_order': ['RPO_e_pc.0', 'RPO_i_pc.0', 'PRO.0'],
            'operator_args': {'RPO_e_pc.0/m_in': {'vtype': 'state_var',
                                                          'dtype': 'float32',
                                                          'shape': (),
                                                          'value': 0.},
                              'RPO_i_pc.0/m_in': {'vtype': 'state_var',
                                                          'dtype': 'float32',
                                                          'shape': (),
                                                          'value': 0.},
                              'PRO.0/m_out': {'vtype': 'state_var',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 0.16},
                              'PRO.0/psp': {'vtype': 'state_var',
                                                   'dtype': 'float32',
                                                   'shape': (),
                                                   'value': 0.},
                              'RPO_e_pc.0/psp': {'vtype': 'state_var',
                                                         'dtype': 'float32',
                                                         'shape': (),
                                                         'value': 0.},
                              'RPO_i_pc.0/psp': {'vtype': 'state_var',
                                                         'dtype': 'float32',
                                                         'shape': (),
                                                         'value': 0.},
                              'RPO_e_pc.0/x': {'vtype': 'state_var',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': 0.},
                              'RPO_i_pc.0/x': {'vtype': 'state_var',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': 0.},
                              'RPO_e_pc.0/H': {'vtype': 'constant',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': 3.25e-3},
                              'RPO_i_pc.0/H': {'vtype': 'constant',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': -22e-3},
                              'RPO_e_pc.0/tau': {'vtype': 'constant',
                                                         'dtype': 'float32',
                                                         'shape': (),
                                                         'value': 10e-3},
                              'RPO_i_pc.0/tau': {'vtype': 'constant',
                                                         'dtype': 'float32',
                                                         'shape': (),
                                                         'value': 20e-3},
                              'PRO.0/m_max': {'vtype': 'constant',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 5.},
                              'PRO.0/r': {'vtype': 'constant',
                                                 'dtype': 'float32',
                                                 'shape': (),
                                                 'value': 560.},
                              'PRO.0/v_th': {'vtype': 'constant',
                                                    'dtype': 'float32',
                                                    'shape': (),
                                                    'value': 6e-3},
                              'RPO_i_pc.0/u': {'vtype': 'constant',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': 0.},
                              'RPO_e_pc.0/u': {'vtype': 'placeholder',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': 220.},
                              },
            'inputs': {}
            }
    graph.add_node(f'PC.{i}', **data)

    data = {'operators': {'RPO_e.0': {
                               'equations': ["d/dt * x = H/tau * (m_in + u) - 2. * 1./tau * x - (1./tau)^2. * psp",
                                             "d/dt * psp = x"],
                               'inputs': {},
                               'output': 'psp'},
                          'PRO.0': {
                              'equations': ["m_out = m_max / (1. + exp(r * (v_th - psp)))"],
                              'inputs': {'psp': {'sources': ['RPO_e.0'],
                                                 'reduce_dim': False}},
                              'output': 'm_out'}},
            'operator_order': ['RPO_e.0', 'PRO.0'],
            'operator_args': {'RPO_e.0/m_in': {'vtype': 'state_var',
                                                        'dtype': 'float32',
                                                        'shape': (),
                                                        'value': 0.},
                              'PRO.0/m_out': {'vtype': 'state_var',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 0.16},
                              'PRO.0/psp': {'vtype': 'state_var',
                                                   'dtype': 'float32',
                                                   'shape': (),
                                                   'value': 0.},
                              'RPO_e.0/psp': {'vtype': 'state_var',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': 0.},
                              'RPO_e.0/x': {'vtype': 'state_var',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 0.},
                              'RPO_e.0/H': {'vtype': 'constant',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 3.25e-3},
                              'RPO_e.0/tau': {'vtype': 'constant',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': 10e-3},
                              'PRO.0/m_max': {'vtype': 'constant',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 5.},
                              'PRO.0/r': {'vtype': 'constant',
                                                 'dtype': 'float32',
                                                 'shape': (),
                                                 'value': 560.},
                              'PRO.0/v_th': {'vtype': 'constant',
                                                    'dtype': 'float32',
                                                    'shape': (),
                                                    'value': 6e-3},
                              'RPO_e.0/u': {'vtype': 'constant',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 0.},
                              },
            'inputs': {}
            }
    graph.add_node(f'EIN.{i}', **data)

    data = {'operators': {'RPO_e.0': {
                               'equations': ["d/dt * x = H/tau * (m_in + u) - 2. * 1./tau * x - (1./tau)^2. * psp",
                                             "d/dt * psp = x"],
                               'inputs': {},
                               'output': 'psp'},
                          'PRO.0': {
                              'equations': ["m_out = m_max / (1. + exp(r * (v_th - psp)))"],
                              'inputs': {'psp': {'sources': ['RPO_e.0'],
                                                 'reduce_dim': False}},
                              'output': 'm_out'}},
            'operator_order': ['RPO_e.0', 'PRO.0'],
            'operator_args': {'RPO_e.0/m_in': {'vtype': 'state_var',
                                                        'dtype': 'float32',
                                                        'shape': (),
                                                        'value': 0.},
                              'PRO.0/m_out': {'vtype': 'state_var',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 0.16},
                              'PRO.0/psp': {'vtype': 'state_var',
                                                   'dtype': 'float32',
                                                   'shape': (),
                                                   'value': 0.},
                              'RPO_e.0/psp': {'vtype': 'state_var',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': 0.},
                              'RPO_e.0/x': {'vtype': 'state_var',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 0.},
                              'RPO_e.0/H': {'vtype': 'constant',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 3.25e-3},
                              'RPO_e.0/tau': {'vtype': 'constant',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': 10e-3},
                              'PRO.0/m_max': {'vtype': 'constant',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 5.},
                              'PRO.0/r': {'vtype': 'constant',
                                                 'dtype': 'float32',
                                                 'shape': (),
                                                 'value': 560.},
                              'PRO.0/v_th': {'vtype': 'constant',
                                                    'dtype': 'float32',
                                                    'shape': (),
                                                    'value': 6e-3},
                              'RPO_e.0/u': {'vtype': 'constant',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 0.},
                              },
            'inputs': {}
            }
    graph.add_node(f'IIN.{i}', **data)

# For the Un_vectorized Connection Dict.
########################################

for a in range(0, n_nodes):
    for b in range(0, n_nodes):

        # source stuff
        if b % 3 == 2:
            source = f'IIN.{int(b/3)}'
            c = C_i[a, b]
        elif b % 3 == 1:
            source = f'EIN.{int(b/3)}'
            c = C_e[a, b]
        else:
            source = f'PC.{int(b/3)}'
            c = C_e[a, b]

        if c != 0:
            edge = {}
            if a % 3 == 0:
                target = f'PC.{int(a/3)}'
                if source.split('.')[0] == 'IIN':
                    edge['target_var'] = 'RPO_i_pc.0/m_in'
                else:
                    edge['target_var'] = 'RPO_e_pc.0/m_in'
            elif a % 3 == 1:
                target = f'EIN.{int(a/3)}'
                edge['target_var'] = 'RPO_e.0/m_in'
            else:
                target = f'IIN.{int(a/3)}'
                edge['target_var'] = 'RPO_e.0/m_in'

            edge['source_var'] = 'PRO.0/m_out'
            edge['weight'] = c
            if int(a/3) == int(b/3):
                edge['delay'] = 0.
            else:
                edge['delay'] = np.random.uniform(0., 6e-3)

            graph.add_edge(source, target, **edge)

# backend setup
###############

inp = 220. + np.random.randn(int(simulation_time/step_size), n_jrcs) * 0.
from pyrates.frontend.parser.graph import circuit_from_graph
circuit = circuit_from_graph(graph)
net = Network(net_config=circuit, dt=step_size, vectorize='none')

# backend simulation
####################

results, _ = net.run(simulation_time=simulation_time,
                     outputs={'V': ('all', 'PRO.0', 'psp')},
                     sampling_step_size=1e-3,
                     inputs={('PC', 'RPO_e_pc.0', 'u'): inp},
                     out_dir='/tmp/log/'
                     )

# results
#########

from pyrates.utility import plot_timeseries, plot_connectivity, plot_phase, analytic_signal, functional_connectivity, \
    plot_psd, time_frequency, plot_tfr

ax1 = plot_timeseries(data=results, variable='psp[V]', plot_style='line_plot', ci=None)
#phase_amp = analytic_signal(results, fmin=6., fmax=10., nodes=[0, 1])
#ax2 = plot_phase(data=phase_amp)
#fc = functional_connectivity(results, metric='csd', frequencies=[7., 8., 9.])
#ax3 = plot_connectivity(fc, plot_style='circular_graph', auto_cluster=True,
#                        xticklabels=results.columns.values, yticklabels=results.columns.values)
#ax4 = plot_psd(results, spatial_colors=False)
#freqs = np.arange(3., 20., 3.)
#power = time_frequency(results.iloc[:, 0:3], method='morlet', freqs=freqs, n_cycles=5)
#ax5 = plot_tfr(power, freqs=freqs, separate_nodes=True)
show()
