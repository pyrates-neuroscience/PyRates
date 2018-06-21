"""This module provides the network class that should be used to set-up any model. It creates a tensorflow graph that
manages all computations/operations and a networkx graph that represents the network structure (nodes + edges).
"""

# external imports
import tensorflow as tf
from typing import Optional, List
from copy import deepcopy
import time as t

# pyrates imports
from pyrates.node import Node
from pyrates.edge import Edge

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


class Network(object):
    """Network level class used to set up and simulate networks of nodes defined by a set of operators.

    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------

    References
    ----------

    Examples
    --------

    """

    def __init__(self,
                 node_dict: dict,
                 connection_dict: dict,
                 dt: float = 1e-3,
                 vectorize: bool = False,
                 tf_graph: Optional[tf.Graph] = None,
                 key: Optional[str] = None
                 ) -> None:
        """Instantiation of network.
        """

        self.key = key if key else 'net0'
        self.dt = dt
        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()
        self.nodes = dict()

        # initialize nodes
        ##################

        with self.tf_graph.as_default():

            with tf.variable_scope(self.key):

                if vectorize:

                    # TODO: Go through input dicts and group similar operations/nodes into tensors
                    pass

                else:

                    # initialize every node in node_dict
                    ####################################

                    node_updates = []

                    for node_name, node_info in node_dict.items():

                        node_ops = dict()
                        node_args = dict()
                        node_args['dt'] = {'variable_type': 'constant',
                                           'name': 'dt',
                                           'shape': (),
                                           'data_type': 'float32',
                                           'initial_value': self.dt}

                        # split dictionary keys into operators and variables
                        for key, val in node_info.items():
                            if 'operator' in key:
                                node_ops[key] = val
                            else:
                                node_args[key] = val

                        # instantiate node
                        node = Node(node_ops, node_args, node_name, self.tf_graph)
                        self.nodes[node_name] = node

                        # collect update operation of node
                        node_updates.append(node.update)

                    # group the update operations of all nodes
                    self.update = tf.tuple(node_updates, name='update')

        # initialize edges
        ##################

        # collect connectivity information from connection_dict
        coupling_ops = connection_dict['coupling_operators']
        coupling_op_args = connection_dict['coupling_operator_args']
        sources = connection_dict['sources']
        targets = connection_dict['targets']

        # check dimensionality of coupling_ops and coupling_op_args
        if len(coupling_ops) < len(sources):
            coupling_ops = [coupling_ops for _ in range(len(sources))]
        if len(coupling_op_args) < len(coupling_ops):
            coupling_op_args = [deepcopy(coupling_op_args) for _ in range(len(sources))]

        with self.tf_graph.as_default():

            with tf.variable_scope(self.key):

                with tf.control_dependencies(self.update):

                    projections = []

                    for i, (source, target, op, op_args) in enumerate(zip(sources,
                                                                          targets,
                                                                          coupling_ops,
                                                                          coupling_op_args)):
                        # create edge
                        edge = Edge(source=self.nodes[source],
                                    target=self.nodes[target],
                                    coupling_op=op,
                                    coupling_op_args=op_args,
                                    tf_graph=self.tf_graph,
                                    key=f'edge_{i}')

                        # collect project operation of edge
                        projections.append(edge.project)

                    # group project operations of all edges
                    self.project = tf.tuple(projections, name='project')

            # group update and project operation (grouped across all nodes/edges)
            self.step = tf.group(self.update, self.project, name='step')

    def run(self, simulation_time: float, inputs: Optional[dict] = None, outputs: Optional[List[tf.Variable]] = None):
        """Simulate the network behavior over time via a tensorflow session.

        Parameters
        ----------
        simulation_time
        inputs
        outputs

        Returns
        -------

        """

        sim_steps = int(simulation_time / self.dt)
        outputs = [node.v for node in self.nodes.values()] if not outputs else outputs

        # linearize input dictionary
        inp = list()
        for step in range(sim_steps):
            inp_dict = dict()
            for key, val in inputs.items():
                if len(val) > 1:
                    inp_dict[key] = val[step]
                else:
                    inp_dict[key] = val
            inp.append(inp_dict)

        with tf.Session(graph=self.tf_graph) as sess:

            # initialize all variables
            sess.run(tf.global_variables_initializer())

            results = []
            t_start = t.time()

            # simulate network behavior for each time-step
            for step in range(sim_steps):

                sess.run(self.step, inp[step])
                results.append([var.eval() for var in outputs])

            t_end = t.time()

            print(f"{simulation_time}s of network behavior were simulated in {t_end - t_start} s given a simulation "
                  f"resolution of {self.dt} s.")

        return results
