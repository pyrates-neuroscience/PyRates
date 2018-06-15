"""

"""

# external imports
import tensorflow as tf
from typing import Optional

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

                    node = Node(node_ops, node_args, node_name, self.tf_graph)
                    self.nodes[node_name] = node
                    node_updates += node.update

                self.update = tf.tuple(node_updates, name='update')

        # infer circuit structure
        #########################

        coupling_ops = connection_dict['coupling_operators']
        coupling_op_args = connection_dict['coupling_operator_args']

        if 'connectivity' in connection_dict.keys():

            connections = connection_dict['connections']

            if 'nodes' in connection_dict.keys():
                node_labels = connection_dict['nodes']
            else:
                node_labels = self.nodes.keys()

            sources, targets = [], []
            for target in connections.shape[0]:
                for source in connections.shape[1]:
                    if connections[target, source] > 0:
                        sources.append(node_labels[source])
                        targets.append(node_labels[target])

        else:

            sources = connection_dict['sources']
            targets = connection_dict['targets']

        if len(coupling_ops) == 1:
            coupling_ops = [coupling_ops for _ in range(len(sources))]
            coupling_op_args = [coupling_op_args for _ in range(len(sources))]

        with self.tf_graph.as_default():

            with tf.variable_scope(self.key):

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


node_dict = {'pop1': {'operator_rtp': ["v = r_in[0, 0] / 2."],
                      'operator_ptr': ["r_out = v + 5."],
                      'v': {'name': 'v',
                            'variable_type': 'state_variable',
                            'data_type': 'float32',
                            'shape': (),
                            'initial_value': 0.},
                      'r_in': {'name': 'r_in',
                               'variable_type': 'state_variable',
                               'data_type': 'float32',
                               'shape': (100, 1),
                               'initial_value': 0.},
                      'r_out': {'name': 'r_out',
                                'variable_type': 'state_variable',
                                'data_type': 'float32',
                                'shape': (),
                                'initial_value': 0.5},
                      },
             'pop2': {'operator_rtp': ["v = r_in[0, 0] / 2."],
                      'operator_ptr': ["r_out = v + 4."],
                      'v': {'name': 'v',
                            'variable_type': 'state_variable',
                            'data_type': 'float32',
                            'shape': (),
                            'initial_value': 0.},
                      'r_in': {'name': 'r_in',
                               'variable_type': 'state_variable',
                               'data_type': 'float32',
                               'shape': (100, 1),
                               'initial_value': 0.},
                      'r_out': {'name': 'r_out',
                                'variable_type': 'state_variable',
                                'data_type': 'float32',
                                'shape': (),
                                'initial_value': 0.5},
                      }
             }

connection_dict = {'coupling_operators': ["r_in[d, 0] = r_in[d, 0] + r_out * c"],
                   'coupling_operator_args': {'c': {'name': 'c',
                                                    'variable_type': 'constant',
                                                    'data_type': 'float32',
                                                    'shape': (),
                                                    'initial_value': 10.},
                                              'd': {'name': 'd',
                                                    'variable_type': 'constant',
                                                    'data_type': 'int32',
                                                    'shape': (),
                                                    'initial_value': 2}
                                              },
                   'sources': ['pop1', 'pop2'],
                   'targets': ['pop2', 'pop1']
                   }

gr = tf.Graph()
net = Network(node_dict, connection_dict, tf_graph=gr, key='test_net')

with tf.Session(graph=gr) as sess:

    sess.run(tf.global_variables_initializer())

    for step in range(100):
        sess.run(net.project)
        sess.run(net.update)
        print('pop1: ', net.nodes['pop1'].r_in.eval(), net.nodes['pop1'].r_out.eval())
        print('pop2: ', net.nodes['pop2'].r_in.eval(), net.nodes['pop2'].r_out.eval())
