"""Runs JRC with tensorflow on population basis
"""

# imports
#########

import tensorflow as tf
import numpy as np
from pyrates.network import Network
from matplotlib.pyplot import *

# parameter definition
######################

# general
step_size = 1e-4
simulation_time = 1.0
n_steps = int(simulation_time / step_size)
n_jrcs = 100
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

# connectivity parameters
c_intra = 135.
c_inter_e = 100.
c_inter_i = 50.

sparseness_e = 0.01
sparseness_i = 0.005

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

# input parameters
inp_mean = 220.
inp_var = 22.
inp = np.zeros((n_steps, n_nodes, 2))
inp[:, 0::3, 0] = inp_mean + np.random.randn(n_steps, n_jrcs) * inp_var

# define network dictionary
###########################

node_dict = {'jrc': {'operator_rtp_syn': ["d/dt * idx(X, 0, 1) = H/tau * (m_in + U) - 2/tau * X - 1/tau**2 * V_syn",
                                          "d/dt * V_syn = X"],
                     'operator_rtp_soma': ["V = reduce_sum(V_syn, ax, keep)"],
                     'operator_ptr': ["m_out = m_max / (1 + exp(r * (v_th - V)))"],
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
                           'variable_type': 'placeholder',
                           'data_type': 'float32',
                           'shape': (n_nodes, 2)},
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

# node_dict = {'pcs': {'operator_rtp_syn': ["d/dt * x_e = H_e/tau_e * (m_ein + u) - 2/tau_e * x_e - 1/tau_e**2 * psp_e",
#                                           "d/dt * psp_e = x_e",
#                                           "d/dt * x_i = H_i/tau_i * m_iin - 2/tau_i * x_i - 1/tau_i**2 * psp_i",
#                                           "d/dt * psp_i = x_i"
#                                           ],
#                      'operator_rtp_soma': ["v = psp_e + psp_i"],
#                      'operator_ptr': ["m_out = m_max / (1 + exp(r * (v_th - v)))"],
#                      'v': {'name': 'v',
#                            'variable_type': 'state_variable',
#                            'data_type': 'float32',
#                            'shape': (),
#                            'initial_value': 0.},
#                      'm_ein': {'name': 'm_ein',
#                                'variable_type': 'state_variable',
#                                'data_type': 'float32',
#                                'shape': (),
#                                'initial_value': 0.},
#                      'm_iin': {'name': 'm_iin',
#                                'variable_type': 'state_variable',
#                                'data_type': 'float32',
#                                'shape': (),
#                                'initial_value': 0.},
#                      'm_out': {'name': 'm_out',
#                                'variable_type': 'state_variable',
#                                'data_type': 'float32',
#                                'shape': (),
#                                'initial_value': 0.16},
#                      'psp_e': {'name': 'psp_e',
#                                'variable_type': 'state_variable',
#                                'data_type': 'float32',
#                                'shape': (),
#                                'initial_value': 0.},
#                      'psp_i': {'name': 'psp_i',
#                                'variable_type': 'state_variable',
#                                'data_type': 'float32',
#                                'shape': (),
#                                'initial_value': 0.},
#                      'x_e': {'name': 'x_e',
#                              'variable_type': 'state_variable',
#                              'data_type': 'float32',
#                              'shape': (),
#                              'initial_value': 0.},
#                      'x_i': {'name': 'x_i',
#                              'variable_type': 'state_variable',
#                              'data_type': 'float32',
#                              'shape': (),
#                              'initial_value': 0.},
#                      'H_e': {'name': 'H_e',
#                              'variable_type': 'constant',
#                              'data_type': 'float32',
#                              'shape': (),
#                              'initial_value': 3.25e-3},
#                      'H_i': {'name': 'H_i',
#                              'variable_type': 'constant',
#                              'data_type': 'float32',
#                              'shape': (),
#                              'initial_value': -22e-3},
#                      'tau_e': {'name': 'tau_e',
#                                'variable_type': 'constant',
#                                'data_type': 'float32',
#                                'shape': (),
#                                'initial_value': 10e-3},
#                      'tau_i': {'name': 'tau_i',
#                                'variable_type': 'constant',
#                                'data_type': 'float32',
#                                'shape': (),
#                                'initial_value': 20e-3},
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
#                               'variable_type': 'constant',
#                               'data_type': 'float32',
#                               'shape': (),
#                               'initial_value': 6e-3},
#                      'u': {'name': 'u',
#                            'variable_type': 'placeholder',
#                            'data_type': 'float32',
#                            'shape': ()}
#                      },
#              'eins': {'operator_rtp_syn': ["d/dt * x = H/tau * m_in - 2/tau * x - 1/tau**2 * psp",
#                                            "d/dt * psp = x"],
#                       'operator_rtp_soma': ["v = psp"],
#                       'operator_ptr': ["m_out = m_max / (1 + exp(r * (v_th - v)))"],
#                       'v': {'name': 'v',
#                             'variable_type': 'state_variable',
#                             'data_type': 'float32',
#                             'shape': (),
#                             'initial_value': 0.},
#                       'm_in': {'name': 'm_in',
#                                'variable_type': 'state_variable',
#                                'data_type': 'float32',
#                                'shape': (),
#                                'initial_value': 0.},
#                       'm_out': {'name': 'm_out',
#                                 'variable_type': 'state_variable',
#                                 'data_type': 'float32',
#                                 'shape': (),
#                                 'initial_value': 0.16},
#                       'psp': {'name': 'psp',
#                               'variable_type': 'state_variable',
#                               'data_type': 'float32',
#                               'shape': (),
#                               'initial_value': 0.},
#                       'x': {'name': 'x',
#                             'variable_type': 'state_variable',
#                             'data_type': 'float32',
#                             'shape': (),
#                             'initial_value': 0.},
#                       'H': {'name': 'H',
#                             'variable_type': 'constant',
#                             'data_type': 'float32',
#                             'shape': (),
#                             'initial_value': 3.25e-3},
#                       'tau': {'name': 'tau',
#                               'variable_type': 'constant',
#                               'data_type': 'float32',
#                               'shape': (),
#                               'initial_value': 10e-3},
#                       'm_max': {'name': 'm_max',
#                                 'variable_type': 'constant',
#                                 'data_type': 'float32',
#                                 'shape': (),
#                                 'initial_value': 5.},
#                       'r': {'name': 'r',
#                             'variable_type': 'constant',
#                             'data_type': 'float32',
#                             'shape': (),
#                             'initial_value': 560.},
#                       'v_th': {'name': 'v_th',
#                                'variable_type': 'constant',
#                                'data_type': 'float32',
#                                'shape': (),
#                                'initial_value': 6e-3}
#                       },
#              'iins': {'operator_rtp_syn': ["d/dt * x = H/tau * m_in - 2/tau * x - 1/tau**2 * psp",
#                                            "d/dt * psp = x"],
#                       'operator_rtp_soma': ["v = psp"],
#                       'operator_ptr': ["m_out = m_max / (1 + exp(r * (v_th - v)))"],
#                       'v': {'name': 'v',
#                             'variable_type': 'state_variable',
#                             'data_type': 'float32',
#                             'shape': (),
#                             'initial_value': 0.},
#                       'm_in': {'name': 'm_in',
#                                'variable_type': 'state_variable',
#                                'data_type': 'float32',
#                                'shape': (),
#                                'initial_value': 0.},
#                       'm_out': {'name': 'm_out',
#                                 'variable_type': 'state_variable',
#                                 'data_type': 'float32',
#                                 'shape': (),
#                                 'initial_value': 0.16},
#                       'psp': {'name': 'psp',
#                               'variable_type': 'state_variable',
#                               'data_type': 'float32',
#                               'shape': (),
#                               'initial_value': 0.},
#                       'x': {'name': 'x',
#                             'variable_type': 'state_variable',
#                             'data_type': 'float32',
#                             'shape': (),
#                             'initial_value': 0.},
#                       'H': {'name': 'H',
#                             'variable_type': 'constant',
#                             'data_type': 'float32',
#                             'shape': (),
#                             'initial_value': 3.25e-3},
#                       'tau': {'name': 'tau',
#                               'variable_type': 'constant',
#                               'data_type': 'float32',
#                               'shape': (),
#                               'initial_value': 10e-3},
#                       'm_max': {'name': 'm_max',
#                                 'variable_type': 'constant',
#                                 'data_type': 'float32',
#                                 'shape': (),
#                                 'initial_value': 5.},
#                       'r': {'name': 'r',
#                             'variable_type': 'constant',
#                             'data_type': 'float32',
#                             'shape': (),
#                             'initial_value': 560.},
#                       'v_th': {'name': 'v_th',
#                                'variable_type': 'constant',
#                                'data_type': 'float32',
#                                'shape': (),
#                                'initial_value': 6e-3}
#                       },
#              }

connection_dict = {'coupling_operators': [["idx(input, :, 0) = dot(C_e, output)",
                                           "idx(input, :, 1) = dot(C_i, output)"]],
                   'coupling_operator_args': [{'C_e': {'name': 'C_e',
                                                       'variable_type': 'constant',
                                                       'data_type': 'float32',
                                                       'shape': (n_nodes, n_nodes),
                                                       'initial_value': C_e},
                                               'C_i': {'name': 'C_i',
                                                       'variable_type': 'constant',
                                                       'data_type': 'float32',
                                                       'shape': (n_nodes, n_nodes),
                                                       'initial_value': C_i},
                                               'reduce_sum': {'variable_type': 'raw',
                                                              'variable': tf.reduce_sum},
                                               'input': 'm_in',
                                               'output': 'm_out'
                                               }],
                   'sources': ['jrc'],
                   'targets': ['jrc']
                   }

# connection_dict = {'coupling_operators': [["m_in = m_out * c"],
#                                           ["m_in = m_out * c"],
#                                           ["m_ein = m_out * c"],
#                                           ["m_iin = m_out * c"]],
#                    'coupling_operator_args': [{'c': {'name': 'c',
#                                                      'variable_type': 'constant',
#                                                      'data_type': 'float32',
#                                                      'shape': (),
#                                                      'initial_value': c * 1.0}
#                                                },
#                                               {'c': {'name': 'c',
#                                                      'variable_type': 'constant',
#                                                      'data_type': 'float32',
#                                                      'shape': (),
#                                                      'initial_value': c * 0.25}
#                                                },
#                                               {'c': {'name': 'c',
#                                                      'variable_type': 'constant',
#                                                      'data_type': 'float32',
#                                                      'shape': (),
#                                                      'initial_value': c * 0.8}
#                                                },
#                                               {'c': {'name': 'c',
#                                                      'variable_type': 'constant',
#                                                      'data_type': 'float32',
#                                                      'shape': (),
#                                                      'initial_value': c * 0.25}
#                                                },
#                                               ],
#                    'sources': ['pcs', 'pcs', 'eins', 'iins'],
#                    'targets': ['eins', 'iins', 'pcs', 'pcs']
#                    }


# network setup
###############

gr = tf.Graph()
net = Network(node_dict, connection_dict, tf_graph=gr, key='test_net', dt=step_size)

# network simulation
####################

potentials = net.run(simulation_time=simulation_time,
                     inputs={net.nodes['jrc'].U: inp},
                     outputs=[net.nodes['jrc'].V])

# results
#########

fig, axes = subplots(figsize=(14, 5))
axes.plot(np.squeeze(np.array(potentials)))
legend(['PCs', 'EINs', 'IINs'])
axes.set_xlabel('timesteps')
axes.set_ylabel('membrane potential')
axes.set_title('Jansen-Rit NMM')
fig.show()
