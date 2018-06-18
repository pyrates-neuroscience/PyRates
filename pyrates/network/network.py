"""

"""

# external imports
import tensorflow as tf
from typing import Optional
from copy import deepcopy

# pyrates imports
from pyrates.node import Node
from pyrates.edge import Edge


class Network(object):

    def __init__(self,
                 node_dict: dict,
                 connection_dict: dict,
                 dt: float = 1e-3,
                 vectorize: bool = False,
                 tf_graph: Optional[tf.Graph] = None,
                 key: Optional[str] = None
                 ) -> None:

        self.key = key if key else 'net0'
        self.dt = dt
        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()
        self.nodes = dict()

        # initialize nodes
        ##################

        with self.tf_graph.as_default():

            with tf.variable_scope(self.key):

                if vectorize:
                    pass
                else:

                    node_updates = []

                    for node_name, node_info in node_dict.items():

                        node_ops = dict()
                        node_args = dict()
                        for key, val in node_info.items():
                            if 'operator' in key:
                                node_ops[key] = val
                            else:
                                node_args[key] = val

                        node_args['dt'] = self.dt
                        node = Node(node_ops, node_args, node_name, self.tf_graph)
                        self.nodes[node_name] = node
                        node_updates.append(node.update)

                    self.update = tf.tuple(node_updates, name='update')

        # infer circuit structure
        #########################

        coupling_ops = connection_dict['coupling_operators']
        coupling_op_args = connection_dict['coupling_operator_args']

        sources = connection_dict['sources']
        targets = connection_dict['targets']

        if len(coupling_ops) == 1:

            coupling_ops = [coupling_ops for _ in range(len(sources))]
            coupling_op_args = [deepcopy(coupling_op_args) for _ in range(len(sources))]

            for key in coupling_op_args[0].keys():
                val = connection_dict[key]
                for i, cargs in enumerate(coupling_op_args):
                    cargs[key]['initial_value'] = val[i]

        with self.tf_graph.as_default():

            with tf.variable_scope(self.key):

                with tf.control_dependencies(self.update):

                    projections = []

                    for i, (source, target, op, op_args) in enumerate(zip(sources,
                                                                          targets,
                                                                          coupling_ops,
                                                                          coupling_op_args)):
                        edge = Edge(source=self.nodes[source],
                                    target=self.nodes[target],
                                    coupling_op=op,
                                    coupling_op_args=op_args,
                                    tf_graph=self.tf_graph,
                                    key=f'edge_{i}')

                        projections.append(edge.project)

                    self.project = tf.tuple(projections, name='project')

            self.step = tf.group(self.update, self.project, name='step')


node_dict = {'pcs': {'operator_rtp_syn': ["d/dt * x_e = H_e/tau_e * (m_ein + u) - 2/tau_e * x_e - 1/tau_e**2 * psp_e",
                                          "d/dt * psp_e = x_e",
                                          "d/dt * x_i = H_i/tau_i * m_iin - 2/tau_i * x_i - 1/tau_i**2 * psp_i",
                                          "d/dt * psp_i = x_i"
                                          ],

                     'operator_rtp_soma': ["v = psp_e + psp_i"],
                     'operator_ptr': ["m_out = m_max / (1 + exp(r * (v_th - v)))"],
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
                     'u': {'name': 'u',
                           'variable_type': 'placeholder',
                           'data_type': 'float32',
                           'shape': ()}
                     },
             'eins': {'operator_rtp_syn': ["d/dt * x = H/tau * m_in - 2/tau * x - 1/tau**2 * psp",
                                           "d/dt * psp = x"],
                      'operator_rtp_soma': ["v = psp"],
                      'operator_ptr': ["m_out = m_max / (1 + exp(r * (v_th - v)))"],
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
                            'initial_value': [3.25e-3]},
                      'tau': {'name': 'tau',
                              'variable_type': 'constant',
                              'data_type': 'float32',
                              'shape': (),
                              'initial_value': [10e-3]},
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
                               'initial_value': 6e-3}
                      },
             'iins': {'operator_rtp_syn': ["d/dt * x = H/tau * m_in - 2/tau * x - 1/tau**2 * psp",
                                           "d/dt * psp = x"],
                      'operator_rtp_soma': ["v = psp"],
                      'operator_ptr': ["m_out = m_max / (1 + exp(r * (v_th - v)))"],
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
                            'initial_value': [3.25e-3]},
                      'tau': {'name': 'tau',
                              'variable_type': 'constant',
                              'data_type': 'float32',
                              'shape': (),
                              'initial_value': [10e-3]},
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
                               'initial_value': 6e-3}
                      },
             }

connection_dict = {'coupling_operators': [["m_in = m_out * c"],
                                          ["m_in = m_out * c"],
                                          ["m_ein = m_out * c"],
                                          ["m_iin = m_out * c"]],
                   'coupling_operator_args': [{'c': {'name': 'c',
                                                     'variable_type': 'constant',
                                                     'data_type': 'float32',
                                                     'shape': (),
                                                     'initial_value': 135.}
                                               },
                                              {'c': {'name': 'c',
                                                     'variable_type': 'constant',
                                                     'data_type': 'float32',
                                                     'shape': (),
                                                     'initial_value': 35.}
                                               },
                                              {'c': {'name': 'c',
                                                     'variable_type': 'constant',
                                                     'data_type': 'float32',
                                                     'shape': (),
                                                     'initial_value': 128.}
                                               },
                                              {'c': {'name': 'c',
                                                     'variable_type': 'constant',
                                                     'data_type': 'float32',
                                                     'shape': (),
                                                     'initial_value': 35.}
                                               },
                                              ],
                   'sources': ['pcs', 'pcs', 'eins', 'iins'],
                   'targets': ['eins', 'iins', 'pcs', 'pcs']
                   }

from matplotlib.pyplot import *
import numpy as np

gr = tf.Graph()
net = Network(node_dict, connection_dict, tf_graph=gr, key='test_net')

potentials = []
with tf.Session(graph=gr) as sess:

    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        sess.run(net.step, {net.nodes['pcs'].u: np.random.uniform(120, 320)})
        potentials.append([net.nodes['pcs'].v.eval(), net.nodes['eins'].v.eval(), net.nodes['iins'].v.eval()])

plot(np.array(potentials))
