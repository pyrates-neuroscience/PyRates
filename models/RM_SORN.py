"""Runs JRC with tensorflow on population basis
"""

import tensorflow as tf
from networkx import MultiDiGraph
from pyrates.backend import ComputeGraph
from matplotlib.pyplot import *
from pyrates.utility import mne_from_dataframe

# Comment this out for making GPU available for Tensorflow.
###########################################################
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
##########################################################

# parameter definition
######################

# general
step_size = 1e-3
simulation_time = 3.0
n_steps = int(simulation_time / step_size)

# Connection Percentage (If Low that means Connections are few!!)
sparseness_e = 0.2
sparseness_i = sparseness_e * 0.5

# No_of_JansenRitCircuit
n_jrcs = 200

# connectivity parameters
c_intra = 135.
c_inter_e = 100. / (n_jrcs * sparseness_e / 0.01)
c_inter_i = 50. / (n_jrcs * sparseness_e / 0.01)

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

for i in range(n_jrcs):
    # intra-circuit connectivity
    C_e[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = np.array([[0., 0.8 * c_intra, 0.],
                                                          [1.0 * c_intra, 0., 0.],
                                                          [0.25 * c_intra, 0., 0.]])
    C_i[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = np.array([[0., 0., 0.25 * c_intra],
                                                          [0., 0., 0.],
                                                          [0., 0., 0.]])
    # inter-circuit connectivity
    for j in range(n_jrcs - 1):
        if i != j:
            weight_e = np.random.uniform()
            if weight_e > (1 - sparseness_e):
                C_e[i * 3, j * 3] = weight_e * c_inter_e
            weight_i = np.random.uniform()
            if weight_i > (1 - sparseness_i):
                C_i[i * 3 + 2, j * 3] = weight_i * c_inter_i

# define backend dictionary
###########################

graph = MultiDiGraph()
for i in range(0, n_jrcs):
    data = {'operators': {'operator_rtp_syn_e': {
                               'equations': ["d/dt * x = H/tau * (m_in + u) - 2./tau * x - 1./tau ^2 * psp",
                                             "d/dt * psp = x"],
                               'inputs': {},
                               'output': 'psp'},
                          'operator_rtp_syn_i': {
                              'equations': ["d/dt * x = H/tau * (m_in + u) - 2./tau * x - 1./tau ^2 * psp",
                                            "d/dt * psp = x"],
                              'inputs': {},
                              'output': 'psp'},
                          'operator_ptr': {
                              'equations': ["v = psp",
                                            "m_out = m_max / (1. + e^(r * (v_th - v)))",
                                            "d/dt * v_th = lr * (m_out - m_target)"],
                              'inputs': {'psp': {'sources': ['operator_rtp_syn_e', 'operator_rtp_syn_i'],
                                                 'reduce_dim': True}},
                              'output': 'm_out'}},
            'op_order': ['operator_rtp_syn_e', 'operator_rtp_syn_i', 'operator_ptr'],
            'operator_args': {'operator_rtp_syn_e/m_in': {'vtype': 'state_var',
                                                          'dtype': 'float32',
                                                          'shape': (),
                                                          'value': 0.},
                              'operator_rtp_syn_i/m_in': {'vtype': 'state_var',
                                                          'dtype': 'float32',
                                                          'shape': (),
                                                          'value': 0.},
                              'operator_ptr/m_out': {'vtype': 'state_var',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 0.16},
                              'operator_ptr/v_th': {'vtype': 'state_var',
                                                    'dtype': 'float32',
                                                    'shape': (),
                                                    'value': 6e-3},
                              'operator_rtp_syn_e/psp': {'vtype': 'state_var',
                                                         'dtype': 'float32',
                                                         'shape': (),
                                                         'value': 0.},
                              'operator_rtp_syn_i/psp': {'vtype': 'state_var',
                                                         'dtype': 'float32',
                                                         'shape': (),
                                                         'value': 0.},
                              'operator_rtp_syn_e/x': {'vtype': 'state_var',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': 0.},
                              'operator_rtp_syn_i/x': {'vtype': 'state_var',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': 0.},
                              'operator_rtp_syn_e/H': {'vtype': 'constant',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': 3.25e-3},
                              'operator_rtp_syn_i/H': {'vtype': 'constant',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': -22e-3},
                              'operator_rtp_syn_e/tau': {'vtype': 'constant',
                                                         'dtype': 'float32',
                                                         'shape': (),
                                                         'value': 10e-3},
                              'operator_rtp_syn_i/tau': {'vtype': 'constant',
                                                         'dtype': 'float32',
                                                         'shape': (),
                                                         'value': 20e-3},
                              'operator_ptr/m_max': {'vtype': 'constant',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 5.},
                              'operator_ptr/r': {'vtype': 'constant',
                                                 'dtype': 'float32',
                                                 'shape': (),
                                                 'value': 560.},
                              'operator_ptr/m_target': {'vtype': 'constant',
                                                        'dtype': 'float32',
                                                        'shape': (),
                                                        'value': 0.16},
                              'operator_ptr/lr': {'vtype': 'constant',
                                                  'dtype': 'float32',
                                                  'shape': (),
                                                  'value': 1e-3},
                              'operator_rtp_syn_e/u': {'vtype': 'constant',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': 220.},
                              'operator_rtp_syn_i/u': {'vtype': 'constant',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': 0.}
                              },
            'inputs': {}
            }
    graph.add_node(f'pc/{i}', **data)

    data = {'operators': {'operator_rtp_syn': {
                               'equations': ["d/dt * x = H/tau * (m_in + u) - 2./tau * x - 1./tau ^2 * psp",
                                             "d/dt * psp = x"],
                               'inputs': {},
                               'output': 'psp'},
                          'operator_ptr': {
                              'equations': ["v = psp", "m_out = m_max / (1. + e^(r * (v_th - v)))"],
                              'inputs': {'psp': {'sources': ['operator_rtp_syn'],
                                                 'reduce_dim': False}},
                              'output': 'm_out'}},
            'op_order': ['operator_rtp_syn', 'operator_ptr'],
            'operator_args': {'operator_rtp_syn/m_in': {'vtype': 'state_var',
                                                        'dtype': 'float32',
                                                        'shape': (),
                                                        'value': 0.},
                              'operator_ptr/m_out': {'vtype': 'state_var',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 0.16},
                              'operator_rtp_syn/psp': {'vtype': 'state_var',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': 0.},
                              'operator_rtp_syn/x': {'vtype': 'state_var',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 0.},
                              'operator_rtp_syn/H': {'vtype': 'constant',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 3.25e-3},
                              'operator_rtp_syn/tau': {'vtype': 'constant',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': 10e-3},
                              'operator_ptr/m_max': {'vtype': 'constant',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 5.},
                              'operator_ptr/r': {'vtype': 'constant',
                                                 'dtype': 'float32',
                                                 'shape': (),
                                                 'value': 560.},
                              'operator_ptr/v_th': {'vtype': 'constant',
                                                    'dtype': 'float32',
                                                    'shape': (),
                                                    'value': 6e-3},
                              'operator_rtp_syn/u': {'vtype': 'constant',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 0.},
                              },
            'inputs': {}
            }
    graph.add_node(f'ein/{i}', **data)

    data = {'operators': {'operator_rtp_syn': {
                               'equations': ["d/dt * x = H/tau * (m_in + u) - 2./tau * x - 1./tau ^2 * psp",
                                             "d/dt * psp = x"],
                               'inputs': {},
                               'output': 'psp'},
                          'operator_ptr': {
                              'equations': ["v = psp", "m_out = m_max / (1. + e^(r * (v_th - v)))"],
                              'inputs': {'psp': {'sources': ['operator_rtp_syn'],
                                                 'reduce_dim': False}},
                              'output': 'm_out'}},
            'op_order': ['operator_rtp_syn', 'operator_ptr'],
            'operator_args': {'operator_rtp_syn/m_in': {'vtype': 'state_var',
                                                        'dtype': 'float32',
                                                        'shape': (),
                                                        'value': 0.},
                              'operator_ptr/m_out': {'vtype': 'state_var',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 0.16},
                              'operator_rtp_syn/psp': {'vtype': 'state_var',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': 0.},
                              'operator_rtp_syn/x': {'vtype': 'state_var',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 0.},
                              'operator_rtp_syn/H': {'vtype': 'constant',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 3.25e-3},
                              'operator_rtp_syn/tau': {'vtype': 'constant',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': 10e-3},
                              'operator_ptr/m_max': {'vtype': 'constant',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 5.},
                              'operator_ptr/r': {'vtype': 'constant',
                                                 'dtype': 'float32',
                                                 'shape': (),
                                                 'value': 560.},
                              'operator_ptr/v_th': {'vtype': 'constant',
                                                    'dtype': 'float32',
                                                    'shape': (),
                                                    'value': 6e-3},
                              'operator_rtp_syn/u': {'vtype': 'constant',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 0.},
                              },
            'inputs': {}
            }
    graph.add_node(f'iin/{i}', **data)

# For the Un_vectorized Connection Dict.
########################################

for a in range(0, n_nodes):
    for b in range(0, n_nodes):

        # source stuff
        if b % 3 == 2:
            source = f'iin/{int(b/3)}'
            c = C_i[a, b]
        elif b % 3 == 1:
            source = f'ein/{int(b/3)}'
            c = C_e[a, b]
        else:
            source = f'pc/{int(b/3)}'
            c = C_e[a, b]

        if c != 0:
            edge = {}
            if a % 3 == 0:
                target = f'pc/{int(a/3)}'
                if source.split('/')[0] == 'iin':
                    edge['target_var'] = 'operator_rtp_syn_i/m_in'
                else:
                    edge['target_var'] = 'operator_rtp_syn_e/m_in'
            elif a % 3 == 1:
                target = f'ein/{int(a/3)}'
                edge['target_var'] = 'operator_rtp_syn/m_in'
            else:
                target = f'iin/{int(a/3)}'
                edge['target_var'] = 'operator_rtp_syn/m_in'

            edge['source_var'] = 'operator_ptr/m_out'
            edge['weight'] = c

            s = source.split('/')[0]
            t = target.split('/')[0]
            graph.add_edge(source, target, **edge)

# backend setup
###############

gr = tf.Graph()
net = ComputeGraph(net_config=graph, tf_graph=gr, key='test_net', dt=step_size, vectorize=True)

# backend simulation
####################

results, ActTime = net.run(simulation_time=simulation_time,
                           outputs={'V': net.nodes['pc/0']['operator_ptr/v']},
                           sampling_step_size=1e-3)

# results
#########

mne_obj = mne_from_dataframe(sim_results=results)
results.plot()
show()
