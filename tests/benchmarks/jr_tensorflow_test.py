"""Runs JRC with tensorflow on population basis
"""

import tensorflow as tf
from networkx import MultiDiGraph
from pyrates.network import Network
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
simulation_time = 5.0
n_steps = int(simulation_time / step_size)

# Connection Percentage (If Low that means Connections are few!!)
sparseness_e = 0.6
sparseness_i = sparseness_e * 0.5

# No_of_JansenRitCircuit
n_jrcs = 10

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

# Input parameters
# inp_mean = 220.
# inp_var = 22.
# inp = np.zeros((n_steps, n_nodes, 2))
# inp[:, 0::3, 0] = inp_mean + np.random.randn(n_steps, n_jrcs) * inp_var

# masks
inp_mask = np.zeros((n_nodes, 2))
inp_mask[0::3, 0] = 1

# hebbian learning mask
hebb_mask = np.ones((n_nodes, n_nodes))
for i in range(n_jrcs):
    hebb_mask[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = 0.
hebb_mask = np.argwhere(hebb_mask)

# define network dictionary
###########################

# The Vectorized Dict.
#####################
# node_dict = {'jrc': {'operator_rtp_syn': ["d/dt * X = H/tau * (m_in + U) - 2./tau * X - 1./tau^2 * V_syn",
#                                           "d/dt * V_syn = X",
#                                           "U = mask * (220. + randn(cint) * 22.)"],
#                      'operator_rtp_soma': ["V = sum(V_syn,1,True)"],
#                      'operator_ptr': ["m_out_old = m_out",
#                                       "m_out = m_max / (1. + e^(r * (v_th - V)))"],
#                      'V': {'name': 'V',
#                            'variable_type': 'state_variable',
#                            'data_type': 'float32',
#                            'shape': (n_nodes, 1),
#                            'initial_value': 0.},
#                      'V_syn': {'name': 'V_syn',
#                                'variable_type': 'state_variable',
#                                'data_type': 'float32',
#                                'shape': (n_nodes, 2),
#                                'initial_value': 0.},
#                      'm_in': {'name': 'm_in',
#                               'variable_type': 'state_variable',
#                               'data_type': 'float32',
#                               'shape': (n_nodes, 2),
#                               'initial_value': 0.},
#                      'm_out': {'name': 'm_out',
#                                'variable_type': 'state_variable',
#                                'data_type': 'float32',
#                                'shape': (n_nodes, 1),
#                                'initial_value': 0.16},
#                      'm_out_old': {'name': 'm_out_old',
#                                    'variable_type': 'state_variable',
#                                    'data_type': 'float32',
#                                    'shape': (n_nodes, 1),
#                                    'initial_value': 0.},
#                      'X': {'name': 'X',
#                            'variable_type': 'state_variable',
#                            'data_type': 'float32',
#                            'shape': (n_nodes, 2),
#                            'initial_value': 0.},
#                      'H': {'name': 'H',
#                            'variable_type': 'constant',
#                            'data_type': 'float32',
#                            'shape': (n_nodes, 2),
#                            'initial_value': H},
#                      'tau': {'name': 'tau',
#                              'variable_type': 'constant',
#                              'data_type': 'float32',
#                              'shape': (n_nodes, 2),
#                              'initial_value': tau},
#                      'm_max': {'name': 'm_max',
#                                'variable_type': 'constant',
#                                'data_type': 'float32',
#                                'shape': (),
#                                'initial_value': 5.},
#                      'r': {'name': 'r',
#                            'variable_type': 'constant',
#                            'data_type': 'float32',
#                            'shape': (),
#                            'initial_value': 560.},
#                      'v_th': {'name': 'v_th',
#                               'variable_type': 'state_variable',
#                               'data_type': 'float32',
#                               'shape': (n_nodes, 1),
#                               'initial_value': 6e-3},
#                      'kappa': {'name': 'kappa',
#                                'variable_type': 'constant',
#                                'data_type': 'float32',
#                                'shape': (),
#                                'initial_value': 0.1},
#                      'm_tar': {'name': 'm_tar',
#                                'variable_type': 'constant',
#                                'data_type': 'float32',
#                                'shape': (),
#                                'initial_value': 0.8},
#                      'U': {'name': 'U',
#                            'variable_type': 'state_variable',
#                            'data_type': 'float32',
#                            'shape': (n_nodes, 2),
#                            'initial_value': 220.},
#                      'randn': {'variable_type': 'raw',
#                                'variable': tf.random_normal},
#                      'cint': {'variable_type': 'raw',
#                               'variable': (n_nodes, 2)
#                               },
#                      'mask': {'name': 'mask',
#                               'variable_type': 'constant',
#                               'data_type': 'float32',
#                               'shape': (n_nodes, 2),
#                               'initial_value': inp_mask}
#                      }
#              }

# The Un-Vectorized Dict.
#########################
graph = MultiDiGraph()
for i in range(0, n_jrcs):
    data = {'operators':
                {'operator_rtp_syn_e':
                     {'equations': ["d/dt * x_e = H_e/tau_e * (m_ein + ue) - 2./tau_e * x_e - 1./tau_e ^2 * psp_e",
                                    "d/dt * psp_e = x_e"],
                      'inputs': [],
                      'output': 'psp_e'},
                 'operator_rtp_syn_i':
                     {'equations': ["d/dt * x_i = H_i/tau_i * (m_iin + ui) - 2./tau_i * x_i - 1./tau_i ^2 * psp_i",
                                    "d/dt * psp_i = x_i"],
                      'inputs': [],
                      'output': 'psp_i'},
                 'operator_rtp_soma_pc':
                     {'equations': [" v = psp_e + psp_i "],
                      'inputs': ['psp_e', 'psp_i'],
                      'output': 'v'},
                 'operator_ptr':
                     {'equations': ["m_out = m_max / (1. + e^(r * (v_th - v)))"],
                      'inputs': ['v'],
                      'output': 'm_out'}
                 },
            'operator_args':
                {'v': {'name': 'v',
                       'vtype': 'state_var',
                       'dtype': 'float32',
                       'shape': (),
                       'value': 0.},
                 'm_ein': {'name': 'm_ein',
                           'vtype': 'state_var',
                           'dtype': 'float32',
                           'shape': (),
                           'value': 0.},
                 'm_iin': {'name': 'm_iin',
                           'vtype': 'state_var',
                           'dtype': 'float32',
                           'shape': (),
                           'value': 0.},
                 'm_out': {'name': 'm_out',
                           'vtype': 'state_var',
                           'dtype': 'float32',
                           'shape': (),
                           'value': 0.16},
                 'psp_e': {'name': 'psp_e',
                           'vtype': 'state_var',
                           'dtype': 'float32',
                           'shape': (),
                           'value': 0.},
                 'psp_i': {'name': 'psp_i',
                           'vtype': 'state_var',
                           'dtype': 'float32',
                           'shape': (),
                           'value': 0.},
                 'x_e': {'name': 'x_e',
                         'vtype': 'state_var',
                         'dtype': 'float32',
                         'shape': (),
                         'value': 0.},
                 'x_i': {'name': 'x_i',
                         'vtype': 'state_var',
                         'dtype': 'float32',
                         'shape': (),
                         'value': 0.},
                 'H_e': {'name': 'H_e',
                         'vtype': 'constant',
                         'dtype': 'float32',
                         'shape': (),
                         'value': 3.25e-3},
                 'H_i': {'name': 'H_i',
                         'vtype': 'constant',
                         'dtype': 'float32',
                         'shape': (),
                         'value': -22e-3},
                 'tau_e': {'name': 'tau_e',
                           'vtype': 'constant',
                           'dtype': 'float32',
                           'shape': (),
                           'value': 10e-3},
                 'tau_i': {'name': 'tau_i',
                           'vtype': 'constant',
                           'dtype': 'float32',
                           'shape': (),
                           'value': 20e-3},
                 'm_max': {'name': 'm_max',
                           'vtype': 'constant',
                           'dtype': 'float32',
                           'shape': (),
                           'value': 5.},
                 'r': {'name': 'r',
                       'vtype': 'constant',
                       'dtype': 'float32',
                       'shape': (),
                       'value': 560.},
                 'v_th': {'name': 'v_th',
                          'vtype': 'constant',
                          'dtype': 'float32',
                          'shape': (),
                          'value': 6e-3},
                 'ue': {'name': 'ue',
                        'vtype': 'constant',
                        'dtype': 'float32',
                        'shape': (),
                        'value': 220.},

                 'ui': {'name': 'ui',
                        'vtype': 'constant',
                        'dtype': 'float32',
                        'shape': (),
                        'value': 0.}
                 }
            }
    graph.add_node(f'pc_{i}', data=data)

    data = {'operators':
                {'operator_rtp':
                     {'equations': ["d/dt * x = H/tau * (m_in + u) - 2./tau * x - 1./tau^2 * v",
                                    "d/dt * v = x"],
                      'inputs': [],
                      'output': 'v'},
                 'operator_ptr':
                     {'equations': ["m_out = m_max / (1. + e^(r * (v_th - v)))"],
                      'inputs': ['v'],
                      'output': 'm_out'}
                 },
            'operator_args':
                {'v': {'name': 'v',
                       'vtype': 'state_var',
                       'dtype': 'float32',
                       'shape': (),
                       'value': 0.},
                 'm_in': {'name': 'm_in',
                          'vtype': 'state_var',
                          'dtype': 'float32',
                          'shape': (),
                          'value': 0.},
                 'm_out': {'name': 'm_out',
                           'vtype': 'state_var',
                           'dtype': 'float32',
                           'shape': (),
                           'value': 0.16},
                 'x': {'name': 'x',
                       'vtype': 'state_var',
                       'dtype': 'float32',
                       'shape': (),
                       'value': 0.},
                 'H': {'name': 'H',
                       'vtype': 'constant',
                       'dtype': 'float32',
                       'shape': (),
                       'value': 3.25e-3},
                 'tau': {'name': 'tau',
                         'vtype': 'constant',
                         'dtype': 'float32',
                         'shape': (),
                         'value': 10e-3},
                 'm_max': {'name': 'm_max',
                           'vtype': 'constant',
                           'dtype': 'float32',
                           'shape': (),
                           'value': 5.},
                 'r': {'name': 'r',
                       'vtype': 'constant',
                       'dtype': 'float32',
                       'shape': (),
                       'value': 560.},
                 'v_th': {'name': 'v_th',
                          'vtype': 'constant',
                          'dtype': 'float32',
                          'shape': (),
                          'value': 6e-3},
                 'u': {'name': 'u',
                       'vtype': 'constant',
                       'dtype': 'float32',
                       'shape': (),
                       'value': 0.}
                 }
            }
    graph.add_node(f'ein_{i}', data=data)

    data = {'operators':
                {'operator_rtp':
                     {'equations': ["d/dt * x = H/tau * (m_in + u) - 2./tau * x - 1./tau^2 * v",
                                    "d/dt * v = x"],
                      'inputs': [],
                      'output': 'v'},
                 'operator_ptr':
                     {'equations': ["m_out = m_max / (1. + e^(r * (v_th - v)))"],
                      'inputs': ['v'],
                      'output': 'm_out'}
                 },
            'operator_args':
                {'v': {'name': 'v',
                       'vtype': 'state_var',
                       'dtype': 'float32',
                       'shape': (),
                       'value': 0.},
                 'm_in': {'name': 'm_in',
                          'vtype': 'state_var',
                          'dtype': 'float32',
                          'shape': (),
                          'value': 0.},
                 'm_out': {'name': 'm_out',
                           'vtype': 'state_var',
                           'dtype': 'float32',
                           'shape': (),
                           'value': 0.16},
                 'x': {'name': 'x',
                       'vtype': 'state_var',
                       'dtype': 'float32',
                       'shape': (),
                       'value': 0.},
                 'H': {'name': 'H',
                       'vtype': 'constant',
                       'dtype': 'float32',
                       'shape': (),
                       'value': 3.25e-3},
                 'tau': {'name': 'tau',
                         'vtype': 'constant',
                         'dtype': 'float32',
                         'shape': (),
                         'value': 10e-3},
                 'm_max': {'name': 'm_max',
                           'vtype': 'constant',
                           'dtype': 'float32',
                           'shape': (),
                           'value': 5.},
                 'r': {'name': 'r',
                       'vtype': 'constant',
                       'dtype': 'float32',
                       'shape': (),
                       'value': 560.},
                 'v_th': {'name': 'v_th',
                          'vtype': 'constant',
                          'dtype': 'float32',
                          'shape': (),
                          'value': 6e-3},
                 'u': {'name': 'u',
                       'vtype': 'constant',
                       'dtype': 'float32',
                       'shape': (),
                       'value': 0.}
                 }
            }
    graph.add_node(f'iin_{i}', data=data)

# node_dict is the collective input dictionary for all the nodes dictionaries.
##############################################################################

# For the Vec. Dict.
####################
# connection_dict = {'coupling_operators': [["input[:,0:1] = C_e @ output",
#                                            "input[:,1:2] = C_i @ output"
#                                            ]],
#                    'coupling_operator_args': {'C_e': {'name': 'C_e',
#                                                       'variable_type': 'state_variable',
#                                                       'data_type': 'float32',
#                                                       'shape': (n_nodes, n_nodes),
#                                                       'initial_value': C_e},
#                                               'C_i': {'name': 'C_i',
#                                                       'variable_type': 'constant',
#                                                       'data_type': 'float32',
#                                                       'shape': (n_nodes, n_nodes),
#                                                       'initial_value': C_i},
#                                               'C_norm': {'name': 'C_norm',
#                                                          'variable_type': 'state_variable',
#                                                          'data_type': 'float32',
#                                                          'shape': (n_nodes, 1),
#                                                          'initial_value': np.sum(C_e, 1)},
#                                               'out_transp': {'variable_type': 'state_variable',
#                                                              'data_type': 'float32',
#                                                              'shape': (1, n_nodes),
#                                                              'name': 'out_transp',
#                                                              'initial_value': 0.},
#                                               'new_inp': {'variable_type': 'state_variable',
#                                                           'data_type': 'float32',
#                                                           'shape': (n_nodes, n_nodes),
#                                                           'name': 'new_inp',
#                                                           'initial_value': 0.},
#                                               'input': {'variable_type': 'target_var',
#                                                         'name': 'm_in'},
#                                               'output': {'variable_type': 'source_var',
#                                                          'name': 'm_out'},
#                                               'old_out': {'variable_type': 'source_var',
#                                                           'name': 'm_out_old'},
#                                               'lr': {'variable_type': 'constant',
#                                                      'data_type': 'float32',
#                                                      'shape': (),
#                                                      'initial_value': 0.1,
#                                                      'name': 'lr'},
#                                               'shape': {'variable_type': 'raw',
#                                                         'variable': (1, n_nodes)},
#                                               'shape2': {'variable_type': 'raw',
#                                                          'variable': (n_nodes, 1)},
#                                               'mask': {'variable_type': 'constant',
#                                                        'data_type': 'int32',
#                                                        'shape': hebb_mask.shape,
#                                                        'initial_value': hebb_mask,
#                                                        'name': 'mask2'},
#                                               },
#                    'sources': ['jrc'],
#                    'targets': ['jrc']
#                    }

# For the Un_vectorized Connection Dict.
########################################

# for i in range(0, n_jrcs):
for a in range(0, n_nodes):
    for b in range(0, n_nodes):

        # source stuff
        if b % 3 == 2:
            source = f'iin_{int(b/3)}'
            c = C_i[a, b]
        elif b % 3 == 1:
            source = f'ein_{int(b/3)}'
            c = C_e[a, b]
        else:
            source = f'pc_{int(b/3)}'
            c = C_e[a, b]

        if c != 0:
            edge = {'operators': {}, 'operator_args': {}}
            if a % 3 == 0:
                target = f'pc_{int(a/3)}'
                if source.split('_')[0] == 'iin':
                    edge['operator_args']['m_in'] = {'vtype': 'target_var',
                                                     'name': 'm_iin'}
                else:
                    edge['operator_args']['m_in'] = {'vtype': 'target_var',
                                                     'name': 'm_ein'}
            elif a % 3 == 1:
                target = f'ein_{int(a/3)}'
            else:
                target = f'iin_{int(a/3)}'

            edge['operators']['coupling_op'] = {'equations': ["m_in = m_out * c"],
                                                'inputs': [],
                                                'output': 'm_in'}
            edge['operator_args']['c'] = {'name': 'c',
                                          'vtype': 'constant',
                                          'dtype': 'float32',
                                          'shape': (),
                                          'value': c * 1.,
                                          }
            edge['operator_args']['m_out'] = {'vtype': 'source_var',
                                              'name': 'm_out'}

            s = source.split('_')[0]
            t = target.split('_')[0]
            graph.add_edge(source, target, key=f'{s}{t}_edge', data=edge)

# network setup
###############
gr = tf.Graph()
net = Network(net_config=graph, tf_graph=gr, key='test_net', dt=step_size)

# output_coll = {}
# for i in range (3, 4):
#     output = {f'V_{i}': net.nodes[f'pcs_{i}'].v}
#     output_coll.update(output)

# network simulation
####################
#
for i, k in net.nodes['BNode']['handle'].__dict__.items():
    if 'vp' in i:
        V = i

results, ActTime = net.run(simulation_time=simulation_time,
                           # inputs={net.nodes['jrc'].U: inp},
                           # inputs = input_coll,
                           outputs={'V': getattr(net.nodes['BNode']['handle'], V)},
                           # outputs=output_coll,
                           sampling_step_size=step_size)

# results
#########

mne_obj = mne_from_dataframe(sim_results=results)
results.plot()
show()
