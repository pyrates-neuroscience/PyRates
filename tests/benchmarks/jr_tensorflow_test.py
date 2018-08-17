"""Runs JRC with tensorflow on population basis
"""

import tensorflow as tf
from pyrates.network import Network
from matplotlib.pyplot import *
from pyrates.utility import mne_from_dataframe

# Comment this out for making GPU available for Tensorflow.
###########################################################
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
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
n_jrcs = 100

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
#inp_mean = 220.
#inp_var = 22.
#inp = np.zeros((n_steps, n_nodes, 2))
#inp[:, 0::3, 0] = inp_mean + np.random.randn(n_steps, n_jrcs) * inp_var

# masks
inp_mask = np.zeros((n_nodes, 2))
inp_mask[0::3, 0] = 1

# hebbian learning mask
hebb_mask = np.ones((n_nodes, n_nodes))
for i in range(n_jrcs):
    hebb_mask[i*3:(i+1)*3, i*3:(i+1)*3] = 0.
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
node_dict = {}
pc_node_dic = {}
ein_node_dic = {}
iin_node_dic = {}
for i in range(0, n_jrcs):
    dic = {
        f'pcs_{i}': {
            'operator_rtp_syn2': [[" d/dt * psp_e = x_e "], [" d/dt * psp_i = x_i "]],
            'operator_rtp_syn1': [
                [" d/dt * x_e = H_e / tau_e * ( m_ein + ue ) - 2./ tau_e * x_e - 1./ tau_e ^2 * psp_e "],
                [" d/dt * x_i = H_i / tau_i * ( m_iin + ui ) - 2./ tau_i * x_i - 1./ tau_i ^2 * psp_i "]],
            # "u = 220. + randn(cint) * 22."
            'operator_rtp_soma_pc': [" v = psp_e + psp_i "],
            'operator_ptr': [" m_out = m_max / (1. + e^( r * ( v_th - v )))"],
            'v': {'name': 'v',
                  'variable_type': 'state_variable',
                  'data_type': 'float32',
                  'shape': (),
                  'initial_value': 0.},
            'm_ein': {'name': 'm_ein',
                      'variable_type': 'state_variable',
                      'data_type': 'float32',
                      'shape': (),
                      'initial_value': 0.},
            'm_iin': {'name': 'm_iin',
                      'variable_type': 'state_variable',
                      'data_type': 'float32',
                      'shape': (),
                      'initial_value': 0.},
            'm_out': {'name': 'm_out',
                      'variable_type': 'state_variable',
                      'data_type': 'float32',
                      'shape': (),
                      'initial_value': 0.16},
            'psp_e': {'name': 'psp_e',
                      'variable_type': 'state_variable',
                      'data_type': 'float32',
                      'shape': (),
                      'initial_value': 0.},
            'psp_i': {'name': 'psp_i',
                      'variable_type': 'state_variable',
                      'data_type': 'float32',
                      'shape': (),
                      'initial_value': 0.},
            'x_e': {'name': 'x_e',
                    'variable_type': 'state_variable',
                    'data_type': 'float32',
                    'shape': (),
                    'initial_value': 0.},
            'x_i': {'name': 'x_i',
                    'variable_type': 'state_variable',
                    'data_type': 'float32',
                    'shape': (),
                    'initial_value': 0.},
            'H_e': {'name': 'H_e',
                    'variable_type': 'constant',
                    'data_type': 'float32',
                    'shape': (),
                    'initial_value': 3.25e-3},
            'H_i': {'name': 'H_i',
                    'variable_type': 'constant',
                    'data_type': 'float32',
                    'shape': (),
                    'initial_value': -22e-3},
            'tau_e': {'name': 'tau_e',
                      'variable_type': 'constant',
                      'data_type': 'float32',
                      'shape': (),
                      'initial_value': 10e-3},
            'tau_i': {'name': 'tau_i',
                      'variable_type': 'constant',
                      'data_type': 'float32',
                      'shape': (),
                      'initial_value': 20e-3},
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
            'ue': {'name': 'ue',
                      'variable_type': 'constant',
                      'data_type': 'float32',
                      'shape': (),
                      'initial_value': 220.},

            'ui': {'name': 'ui',
                      'variable_type': 'constant',
                      'data_type': 'float32',
                      'shape': (),
                      'initial_value': 0.},
            # 'u': {'name': 'u',
            #       'variable_type': 'state_variable',
            #       'data_type': 'float32',
            #       'shape': (),
            #       'initial_value': 0.
            #       },
            # 'randn': {'variable_type': 'raw',
            #           'variable': tf.random_normal},
            # 'cint': {'name': 'cint',
            #          'variable_type': 'constant',
            #          'data_type': 'int32',
            #          'shape': ([0]),
            #          'initial_value': 0}

        }, }
    pc_node_dic.update(dic)
    dic = {
        f'ein_{i}': {
            'operator_rtp_syn1': ["d/dt * x = H / tau * ( m_in + uein ) - 2./ tau * x - 1./ tau ^ 2 * psp "],
            'operator_rtp_syn2': ["d/dt * psp = x "],
            'operator_rtp_soma': [" v = psp "],
            'operator_ptr': [" m_out = m_max / (1. + e^( r * ( v_th - v )))"],
            'v': {'name': 'v',
                  'variable_type': 'state_variable',
                  'data_type': 'float32',
                  'shape': (),
                  'initial_value': 0.},
            'm_in': {'name': 'm_in',
                     'variable_type': 'state_variable',
                     'data_type': 'float32',
                     'shape': (),
                     'initial_value': 0.},
            'm_out': {'name': 'm_out',
                      'variable_type': 'state_variable',
                      'data_type': 'float32',
                      'shape': (),
                      'initial_value': 0.16},
            'psp': {'name': 'psp',
                    'variable_type': 'state_variable',
                    'data_type': 'float32',
                    'shape': (),
                    'initial_value': 0.},
            'x': {'name': 'x',
                  'variable_type': 'state_variable',
                  'data_type': 'float32',
                  'shape': (),
                  'initial_value': 0.},
            'H': {'name': 'H',
                  'variable_type': 'constant',
                  'data_type': 'float32',
                  'shape': (),
                  'initial_value': 3.25e-3},
            'tau': {'name': 'tau',
                    'variable_type': 'constant',
                    'data_type': 'float32',
                    'shape': (),
                    'initial_value': 10e-3},
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
            'uein': {'name': 'uein',
                     'variable_type': 'constant',
                     'data_type': 'float32',
                     'shape': (),
                     'initial_value': 0.},

        }, }
    ein_node_dic.update(dic)

    dic = {
        f'iin_{i}': {
            'operator_rtp_syn1': ["d/dt* x = H / tau * ( m_in + uiin ) - 2./ tau * x - 1./ tau ^2 * psp "],
            'operator_rtp_syn2': ["d/dt * psp = x "],
            'operator_rtp_soma': [" v = psp "],
            'operator_ptr': [" m_out = m_max / (1. + e^( r * ( v_th - v )))"],
            'v': {'name': 'v',
                  'variable_type': 'state_variable',
                  'data_type': 'float32',
                  'shape': (),
                  'initial_value': 0.},
            'm_in': {'name': 'm_in',
                     'variable_type': 'state_variable',
                     'data_type': 'float32',
                     'shape': (),
                     'initial_value': 0.},
            'm_out': {'name': 'm_out',
                      'variable_type': 'state_variable',
                      'data_type': 'float32',
                      'shape': (),
                      'initial_value': 0.16},
            'psp': {'name': 'psp',
                    'variable_type': 'state_variable',
                    'data_type': 'float32',
                    'shape': (),
                    'initial_value': 0.},
            'x': {'name': 'x',
                  'variable_type': 'state_variable',
                  'data_type': 'float32',
                  'shape': (),
                  'initial_value': 0.},
            'H': {'name': 'H',
                  'variable_type': 'constant',
                  'data_type': 'float32',
                  'shape': (),
                  'initial_value': 3.25e-3},
            'tau': {'name': 'tau',
                    'variable_type': 'constant',
                    'data_type': 'float32',
                    'shape': (),
                    'initial_value': 10e-3},
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
            'uiin': {'name': 'uiin',
                     'variable_type': 'constant',
                     'data_type': 'float32',
                     'shape': (),
                     'initial_value': 0.},
        },
    }
    iin_node_dic.update(dic)

# node_dict is the collective input dictionary for all the nodes dictionaries.
##############################################################################
node_dict.update(pc_node_dic)
node_dict.update(ein_node_dic)
node_dict.update(iin_node_dic)
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
target_list = []
source_list = []
input_list = []
output_list = []
c_dict_coll = []

# for i in range(0, n_jrcs):
for a in range(0, n_nodes):
    for b in range(0, n_nodes):

        # source stuff
        if b % 3 == 2:
            source = f'iins_{int(b/3)}'
            output = f'm_outiin_{int(b/3)}'
            c = C_i[a, b]
        elif b % 3 == 1:
            source = f'eins_{int(b/3)}'
            output = f'm_outein_{int(b/3)}'
            c = C_e[a, b]
        else:
            source = f'pcs_{int(b/3)}'
            output = f'm_outpcs_{int(b/3)}'
            c = C_e[a, b]

        if c != 0:
            if a % 3 == 0:
                target = f'pcs_{int(a/3)}'
                if 'eins' in source:
                    input = f'm_einpcs_{int(a/3)}'
                else:
                    input = f'm_iinpcs_{int(a/3)}'
            elif a % 3 == 1:
                target = f'eins_{int(a/3)}'
                input = f'm_inein_{int(a/3)}'
            else:
                target = f'iins_{int(a/3)}'
                input = f'm_iniin_{int(a/3)}'

            c_dict = {'name': 'c',
                      'variable_type': 'state_variable',
                      'data_type': 'float32',
                      'shape': (),
                      'initial_value': c * 1.,
                      }

            input_dict = {'variable_type': 'target_var',
                           'name': f'{input}'}
            output_dict = {'variable_type': 'source_var',
                           'name': f'{output}'}

            c_dict_coll.append(c_dict)
            target_list.append(target)
            source_list.append(source)
            input_list.append(input_dict)
            output_list.append(output_dict)

connection_dict = {'coupling_operators': [["input = output * c"]],

                   'coupling_operator_args': {'c': c_dict_coll, 'input': input_list, 'output': output_list},

                   'sources': source_list,
                   'targets': target_list
                   }

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

