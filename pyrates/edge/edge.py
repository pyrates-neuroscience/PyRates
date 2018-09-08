"""This module contains an edge class that allows to connect variables on a source and a target node via an operator.
"""

# external imports
import tensorflow as tf
from typing import Optional, List

# pyrates imports
from pyrates.operator import Operator
from pyrates.parser import parse_dict
from pyrates.node import Node

# meta infos
__author__ = "Richard Gast"
__status__ = "Development"


class Edge(object):
    """Base edge class to connect a tensorflow variable on source to a tensorflow variable on target via an
    operator.

    Parameters
    ----------
    source
    target
    coupling_op
    coupling_op_args
    key
    tf_graph

    Attributes
    ----------

    Methods
    -------

    Examples
    --------

    References
    ----------

    """
    def __init__(self,
                 source: Node,
                 target: Node,
                 coupling_ops: dict,
                 coupling_op_args: dict,
                 key: str,
                 tf_graph: Optional[tf.Graph] = None):
        """Instantiates edge.
        """

        self.operations = {}
        self.key = key
        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()

        # add variables/operations to tensorflow graph
        ##############################################

        with self.tf_graph.as_default():

            with tf.variable_scope(self.key):

                # handle operation arguments
                ############################

                # replace the coupling operator arguments with the fields from source and target where necessary
                co_args = {}
                for key, val in coupling_op_args.items():

                    if val['vtype'] == 'source_var':

                        var = getattr(source, val['name'])
                        co_args[key] = {'vtype': 'raw', 'value': var}

                    elif val['vtype'] == 'target_var':

                        var = getattr(target, val['name'])
                        co_args[key] = {'vtype': 'raw', 'value': var}

                    else:

                        co_args[key] = val

                # create tensorflow variables from the additional operator args
                tf_vars, var_names = parse_dict(var_dict=co_args,
                                                var_scope=self.key,
                                                tf_graph=self.tf_graph)

                operator_args = dict()

                # bind operator args to edge
                for tf_var, var_name in zip(tf_vars, var_names):
                    setattr(self, var_name, tf_var)
                    operator_args[var_name] = {'var': tf_var, 'dependency': False}

                # instantiate operations
                ########################

                tf_ops = []
                for op_name, op in coupling_ops.items():

                    # store operator equations
                    self.operations[op_name] = op['equations']

                    # set input dependencies
                    for inp in op['inputs']:
                        operator_args[inp]['dependency'] = True
                        if 'op' not in operator_args[inp].keys():
                            raise ValueError(f"Invalid dependencies found in operator: {op['equations']}. Input "
                                             f"Variable {inp} has not been calculated yet.")

                    # create coupling operator
                    operator = Operator(expressions=op['equations'],
                                        expression_args=operator_args,
                                        tf_graph=self.tf_graph,
                                        key=op_name,
                                        variable_scope=self.key)

                    # collect tensorflow operator
                    tf_ops.append(operator.create())

                    # handle dependencies
                    operator_args[op['output']]['op'] = tf_ops[-1]
                    for arg in operator_args.values():
                        arg['dependency'] = False

                    # bind newly created tf variables to edge
                    for var_name, tf_var in operator.args.items():
                        if not hasattr(self, var_name):
                            setattr(self, var_name, tf_var['var'])
                        if var_name not in operator_args.keys():
                            operator_args[var_name]['var'] = tf_var

                # collect tensorflow operations of that edge
                self.project = tf.group(tf_ops, name=f"{self.key}_project")
