"""Runs JRC with tensorflow on population basis
"""

import tensorflow as tf
from pyrates.network import Network
from matplotlib.pyplot import *

# Comment this out for making GPU available for Tensorflow.
###########################################################
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
##########################################################

# parameter definition
######################

# general
step_size = 1e-3
simulation_time = 1.0
n_steps = int(simulation_time / step_size)


# Connection Percentage (If Low that means Connections are few!!)
sparseness_e = 0.01
sparseness_i = sparseness_e * 0.5

# No_of_JansenRitCircuit
n_jrcs = 10

# connectivity parameters
c_intra = 135.
c_inter_e = 100. / (n_jrcs*sparseness_e/0.01)
c_inter_i = 50. / (n_jrcs*sparseness_e/0.01)

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
    C_e[i*3:(i+1)*3, i*3:(i+1)*3] = np.array([[0., 0.8 * c_intra, 0.],
                                              [1.0 * c_intra, 0., 0.],
                                              [0.25 * c_intra, 0., 0.]])
    C_i[i*3:(i+1)*3, i*3:(i+1)*3] = np.array([[0., 0., 0.25 * c_intra],
                                              [0., 0., 0.],
                                              [0., 0., 0.]])
    # inter-circuit connectivity
    for j in range(n_jrcs-1):
        if i != j:
            weight_e = np.random.uniform()
            if weight_e > (1 - sparseness_e):
                C_e[i*3, j*3] = weight_e * c_inter_e
            weight_i = np.random.uniform()
            if weight_i > (1 - sparseness_i):
                C_i[i*3 + 2, j*3] = weight_i * c_inter_i

# Input parameters
inp_mean = 220.
inp_var = 22.
inp = np.zeros((n_steps, n_nodes, 2))
inp[:, 0::3, 0] = inp_mean + np.random.randn(n_steps, n_jrcs) * inp_var

# mask
mask = np.zeros((n_nodes, 2))
mask[0::3, 0] = 1

# define network dictionary
###########################

# The Vectorized Dict.
#####################
node_dict = {'jrc': {'operator_rtp_syn': ["d/dt * X = H/tau * (m_in + U) - 2./tau * X - 1./tau^2 * V_syn",
                                          "d/dt * V_syn = X",
                                          "U = mask * (220. + randn(cint) * 22.)"],
                     'operator_rtp_soma': ["V = sum(V_syn,ax,keep)"],
                     'operator_ptr': ["m_out = m_max / (1. + e^(r * (v_th - V)))"],

                     's': {'name': 's',
                           'variable_type': 'state_variable',
                           'data_type': 'float32',
                           'shape': (n_nodes, 1),
                           'initial_value': 0.},
                     'bf': {'name': 'bf',
                            'variable_type': 'state_variable',
                            'data_type': 'float32',
                            'shape': (n_nodes, 1),
                            'initial_value': 0.},
                     'bv': {'name': 'bv',
                            'variable_type': 'state_variable',
                            'data_type': 'float32',
                            'shape': (n_nodes, 1),
                            'initial_value': 0.},
                     'dhg': {'name': 'dhg',
                             'variable_type': 'state_variable',
                             'data_type': 'float32',
                             'shape': (n_nodes, 1),
                             'initial_value': 0.},
                     'bold': {'name': 'bold',
                              'variable_type': 'state_variable',
                              'data_type': 'float32',
                              'shape': (n_nodes, 1),
                              'initial_value': 0.},
                     'V': {'name': 'V',
                           'variable_type': 'state_variable',
                           'data_type': 'float32',
                           'shape': (n_nodes, 1),
                           'initial_value': 0.},
                     'V_syn': {'name': 'V_syn',
                               'variable_type': 'state_variable',
                               'data_type': 'float32',
                               'shape': (n_nodes, 2),
                               'initial_value': 0.},
                     'm_in': {'name': 'm_in',
                              'variable_type': 'state_variable',
                              'data_type': 'float32',
                              'shape': (n_nodes, 2),
                              'initial_value': 0.},
                     'm_out': {'name': 'm_out',
                               'variable_type': 'state_variable',
                               'data_type': 'float32',
                               'shape': (n_nodes, 1),
                               'initial_value': 0.16},
                     'X': {'name': 'X',
                           'variable_type': 'state_variable',
                           'data_type': 'float32',
                           'shape': (n_nodes, 2),
                           'initial_value': 0.},
                     'H': {'name': 'H',
                           'variable_type': 'constant',
                           'data_type': 'float32',
                           'shape': (n_nodes, 2),
                           'initial_value': H},
                     'tau': {'name': 'tau',
                             'variable_type': 'constant',
                             'data_type': 'float32',
                             'shape': (n_nodes, 2),
                             'initial_value': tau},
                     'm_max': {'name': 'm_max',
                               'variable_type': 'constant',
                               'data_type': 'float32',
                               'shape': (),
                               'initial_value': 5.},
                     'r': {'name': 'r',
                           'variable_type': 'constant',
                           'data_type': 'float32',
                           'shape': (),
                           'initial_value': 560.},
                     'v_th': {'name': 'v_th',
                              'variable_type': 'constant',
                              'data_type': 'float32',
                              'shape': (),
                              'initial_value': 6e-3},
                     'U': {'name': 'U',
                           'variable_type': 'state_variable',
                           'data_type': 'float32',
                           'shape': (n_nodes, 2),
                           'initial_value': 220.},
                     'randn': {'variable_type': 'raw',
                              'variable': tf.random_normal},
                     'cint': {'variable_type': 'raw',
                              'variable': (n_nodes, 2)
                              },
                     'mask': {'name': 'mask',
                              'variable_type': 'constant',
                              'data_type': 'float32',
                              'shape': (n_nodes, 2),
                              'initial_value': mask},
                     'reduce_sum': {'variable_type': 'raw',
                                    'variable': tf.reduce_sum},
                     'ax': {'name': 'ax',
                            'variable_type': 'constant',
                            'data_type': 'int32',
                            'shape': (),
                            'initial_value': 1},
                     'keep': {'variable_type': 'raw',
                              'variable': True}
                     }
             }

# The Un-Vectorized Dict.
#########################
# new_dict = {}
# for i in range (0, n_jrcs):
#     node_dict = {f'pcs_{i}': {'operator_rtp_syn': ["d/dt * x_e = H_e/tau_e * (m_ein + u) - 2/tau_e * x_e - 1/tau_e**2 * psp_e",
#                                               "d/dt * psp_e = x_e",
#                                               "d/dt * x_i = H_i/tau_i * m_iin - 2/tau_i * x_i - 1/tau_i**2 * psp_i",
#                                               "d/dt * psp_i = x_i",
#                                               "u = 220. + randn(cint) * 22."
#                                               ],
#                          'operator_rtp_soma': ["v = psp_e + psp_i"],
#                          'operator_ptr': ["m_out = m_max / (1 + expo(r * (v_th - v)))"],
#                          'v': {'name': 'v',
#                                'variable_type': 'state_variable',
#                                'data_type': 'float32',
#                                'shape': (),
#                                'initial_value': 0.},
#                          'm_ein': {'name': 'm_ein',
#                                    'variable_type': 'state_variable',
#                                    'data_type': 'float32',
#                                    'shape': (),
#                                    'initial_value': 0.},
#                          'm_iin': {'name': 'm_iin',
#                                    'variable_type': 'state_variable',
#                                    'data_type': 'float32',
#                                    'shape': (),
#                                    'initial_value': 0.},
#                          'm_out': {'name': 'm_out',
#                                    'variable_type': 'state_variable',
#                                    'data_type': 'float32',
#                                    'shape': (),
#                                    'initial_value': 0.16},
#                          'psp_e': {'name': 'psp_e',
#                                    'variable_type': 'state_variable',
#                                    'data_type': 'float32',
#                                    'shape': (),
#                                    'initial_value': 0.},
#                          'psp_i': {'name': 'psp_i',
#                                    'variable_type': 'state_variable',
#                                    'data_type': 'float32',
#                                    'shape': (),
#                                    'initial_value': 0.},
#                          'x_e': {'name': 'x_e',
#                                  'variable_type': 'state_variable',
#                                  'data_type': 'float32',
#                                  'shape': (),
#                                  'initial_value': 0.},
#                          'x_i': {'name': 'x_i',
#                                  'variable_type': 'state_variable',
#                                  'data_type': 'float32',
#                                  'shape': (),
#                                  'initial_value': 0.},
#                          'H_e': {'name': 'H_e',
#                                  'variable_type': 'constant',
#                                  'data_type': 'float32',
#                                  'shape': (),
#                                  'initial_value': 3.25e-3},
#                          'H_i': {'name': 'H_i',
#                                  'variable_type': 'constant',
#                                  'data_type': 'float32',
#                                  'shape': (),
#                                  'initial_value': -22e-3},
#                          'tau_e': {'name': 'tau_e',
#                                    'variable_type': 'constant',
#                                    'data_type': 'float32',
#                                    'shape': (),
#                                    'initial_value': 10e-3},
#                          'tau_i': {'name': 'tau_i',
#                                    'variable_type': 'constant',
#                                    'data_type': 'float32',
#                                    'shape': (),
#                                    'initial_value': 20e-3},
#                          'm_max': {'name': 'm_max',
#                                    'variable_type': 'constant',
#                                    'data_type': 'float32',
#                                    'shape': (),
#                                    'initial_value': 5.},
#                          'r': {'name': 'r',
#                                'variable_type': 'constant',
#                                'data_type': 'float32',
#                                'shape': (),
#                                'initial_value': 560.},
#                          'v_th': {'name': 'v_th',
#                                   'variable_type': 'constant',
#                                   'data_type': 'float32',
#                                   'shape': (),
#                                   'initial_value': 6e-3},
#                          'u': {'name': 'u',
#                                'variable_type': 'state_variable',
#                                'data_type': 'float32',
#                                'shape': (),
#                                'initial_value': 0.
#                                },
#                          'randn': {'variable_type': 'raw',
#                                'variable': tf.random_normal},
#                          'cint': {'name': 'cint',
#                                   'variable_type': 'constant',
#                                   'data_type':'int32',
#                                   'shape': ([0]),
#                                   'initial_value': 0}
#                          },
#
#                  f'eins_{i}': {'operator_rtp_syn': ["d/dt * x = H/tau * m_in - 2/tau * x - 1/tau**2 * psp",
#                                                "d/dt * psp = x"],
#                           'operator_rtp_soma': ["v = psp"],
#                           'operator_ptr': ["m_out = m_max / (1 + expo(r * (v_th - v)))"],
#                           'v': {'name': 'v',
#                                 'variable_type': 'state_variable',
#                                 'data_type': 'float32',
#                                 'shape': (),
#                                 'initial_value': 0.},
#                           'm_in': {'name': 'm_in',
#                                    'variable_type': 'state_variable',
#                                    'data_type': 'float32',
#                                    'shape': (),
#                                    'initial_value': 0.},
#                           'm_out': {'name': 'm_out',
#                                     'variable_type': 'state_variable',
#                                     'data_type': 'float32',
#                                     'shape': (),
#                                     'initial_value': 0.16},
#                           'psp': {'name': 'psp',
#                                   'variable_type': 'state_variable',
#                                   'data_type': 'float32',
#                                   'shape': (),
#                                   'initial_value': 0.},
#                           'x': {'name': 'x',
#                                 'variable_type': 'state_variable',
#                                 'data_type': 'float32',
#                                 'shape': (),
#                                 'initial_value': 0.},
#                           'H': {'name': 'H',
#                                 'variable_type': 'constant',
#                                 'data_type': 'float32',
#                                 'shape': (),
#                                 'initial_value': 3.25e-3},
#                           'tau': {'name': 'tau',
#                                   'variable_type': 'constant',
#                                   'data_type': 'float32',
#                                   'shape': (),
#                                   'initial_value': 10e-3},
#                           'm_max': {'name': 'm_max',
#                                     'variable_type': 'constant',
#                                     'data_type': 'float32',
#                                     'shape': (),
#                                     'initial_value': 5.},
#                           'r': {'name': 'r',
#                                 'variable_type': 'constant',
#                                 'data_type': 'float32',
#                                 'shape': (),
#                                 'initial_value': 560.},
#                           'v_th': {'name': 'v_th',
#                                    'variable_type': 'constant',
#                                    'data_type': 'float32',
#                                    'shape': (),
#                                    'initial_value': 6e-3}
#                           },
#                  f'iins_{i}': {'operator_rtp_syn': ["d/dt * x = H/tau * m_in - 2/tau * x - 1/tau**2 * psp",
#                                                "d/dt * psp = x"],
#                           'operator_rtp_soma': ["v = psp"],
#                           'operator_ptr': ["m_out = m_max / (1 + expo(r * (v_th - v)))"],
#                           'v': {'name': 'v',
#                                 'variable_type': 'state_variable',
#                                 'data_type': 'float32',
#                                 'shape': (),
#                                 'initial_value': 0.},
#                           'm_in': {'name': 'm_in',
#                                    'variable_type': 'state_variable',
#                                    'data_type': 'float32',
#                                    'shape': (),
#                                    'initial_value': 0.},
#                           'm_out': {'name': 'm_out',
#                                     'variable_type': 'state_variable',
#                                     'data_type': 'float32',
#                                     'shape': (),
#                                     'initial_value': 0.16},
#                           'psp': {'name': 'psp',
#                                   'variable_type': 'state_variable',
#                                   'data_type': 'float32',
#                                   'shape': (),
#                                   'initial_value': 0.},
#                           'x': {'name': 'x',
#                                 'variable_type': 'state_variable',
#                                 'data_type': 'float32',
#                                 'shape': (),
#                                 'initial_value': 0.},
#                           'H': {'name': 'H',
#                                 'variable_type': 'constant',
#                                 'data_type': 'float32',
#                                 'shape': (),
#                                 'initial_value': 3.25e-3},
#                           'tau': {'name': 'tau',
#                                   'variable_type': 'constant',
#                                   'data_type': 'float32',
#                                   'shape': (),
#                                   'initial_value': 10e-3},
#                           'm_max': {'name': 'm_max',
#                                     'variable_type': 'constant',
#                                     'data_type': 'float32',
#                                     'shape': (),
#                                     'initial_value': 5.},
#                           'r': {'name': 'r',
#                                 'variable_type': 'constant',
#                                 'data_type': 'float32',
#                                 'shape': (),
#                                 'initial_value': 560.},
#                           'v_th': {'name': 'v_th',
#                                    'variable_type': 'constant',
#                                    'data_type': 'float32',
#                                    'shape': (),
#                                    'initial_value': 6e-3}
#                           },
#                  }
#     new_dict.update(node_dict)


# For the Vec. Dict.
####################
connection_dict = {'coupling_operators': [["input[:,0:1] = C_e @ output",
                                           "input[:,1:2] = C_i @ output"]],
                   'coupling_operator_args': {'C_e': {'name': 'C_e',
                                                       'variable_type': 'constant',
                                                       'data_type': 'float32',
                                                       'shape': (n_nodes, n_nodes),
                                                       'initial_value': C_e},
                                               'C_i': {'name': 'C_i',
                                                       'variable_type': 'constant',
                                                       'data_type': 'float32',
                                                       'shape': (n_nodes, n_nodes),
                                                       'initial_value': C_i},
                                               'input': 'm_in',
                                               'output': 'm_out'
                                               },
                   'sources': ['jrc'],
                   'targets': ['jrc']
                   }

# # For the Un_vectorized Connection Dict.
# ########################################
# target_list = []
# source_list = []
# c_list = []
# input_list = []
# c_dict_coll = []
#
# # for i in range(0, n_jrcs):
# for a in range(0, n_nodes):
#     for b in range(0, n_nodes):
#
#
#         # source stuff
#         if b % 3 == 2:
#             source = f'iins_{int(b/3)}'
#             c = C_i[a, b]
#         elif b % 3 == 1:
#             source = f'eins_{int(b/3)}'
#             c = C_e[a, b]
#         else:
#             source = f'pcs_{int(b/3)}'
#             c = C_e[a, b]
#
#         if c != 0:
#             if a % 3 == 0:
#                 target = f'pcs_{int(a/3)}'
#                 if 'eins' in source:
#                     input = 'm_ein'
#                 else:
#                     input = 'm_iin'
#             elif a % 3 == 1:
#                 target = f'eins_{int(a/3)}'
#                 input = 'm_in'
#             else:
#                 target = f'iins_{int(a/3)}'
#                 input = 'm_in'
#
#             c_dict = {'name': 'c',
#                       'variable_type': 'constant',
#                       'data_type': 'float32',
#                       'shape': (),
#                       'initial_value': c * 1.0,
#                       }
#
#             c_dict_coll.append(c_dict)
#             target_list.append(target)
#             source_list.append(source)
#             input_list.append(input)

# connection_dict = {'coupling_operators': [["input = output * c"]],
#                    'coupling_operator_args': {'c': c_dict_coll, 'input': input_list, 'output': 'm_out'},
#
#                    'sources': source_list,   #[f'pcs_{i}', f'pcs_{i}', f'eins_{i}', f'iins_{i}'],
#                    'targets': target_list   #[f'eins_{i}', f'iins_{i}', f'pcs_{i}', f'pcs_{i}']
#                    }

# network setup
###############
gr = tf.Graph()
net = Network(node_dict, connection_dict, tf_graph=gr, key='test_net', dt=step_size)

# output_coll = {}
# for i in range (3, 4):
#     output = {f'V_{i}': net.nodes[f'pcs_{i}'].v}
#     output_coll.update(output)

# network simulation
####################

results, ActTime = net.run(simulation_time=simulation_time,
                  # inputs={net.nodes['jrc'].U: inp},
                  # inputs = input_coll,
                  # outputs={'V': net.nodes['jrc'].V},
                  # outputs=output_coll,
                  sampling_step_size=1e-3)

# results
#########
fig, axes = subplots(figsize=(14, 5))
axes.plot(np.squeeze(np.array(results)))
legend(['PCs', 'EINs', 'IINs'])
axes.set_xlabel('timesteps')
axes.set_ylabel('membrane potential')
axes.set_title('Jansen-Rit NMM')
show()
