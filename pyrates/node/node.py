"""This module contains the node class used to create a backend node from a set of operations.
"""

# external imports
from typing import Dict, List, Optional
import tensorflow as tf

# pyrates imports
from pyrates.operator import Operator
from pyrates.parser import parse_dict

# meta infos
__author__ = "Richard Gast"
__status__ = "Development"


class Node(object):
    """Basic node class. Creates a node from a set of operations plus information about the variables contained therein.
    This node is a tensorflow sub-graph with an `update` operation that can be used to update all state variables
    described by the operators.

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
                 operations: Dict[str, List[str]],
                 operation_args: dict,
                 key: str,
                 tf_graph: Optional[tf.Graph] = None
                 ) -> None:
        """Instantiates node.
        """

        self.key = key
        self.operations = dict()
        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()

        # instantiate operations
        ########################

        with self.tf_graph.as_default():

            with tf.variable_scope(self.key):

                # get tensorflow variables and the variable names from operation_args
                tf_vars, var_names = parse_dict(var_dict=operation_args,
                                                var_scope=self.key,
                                                tf_graph=self.tf_graph)

                operator_args = dict()

                # bind tensorflow variables to node and save them in dictionary for the operator class
                for tf_var, var_name in zip(tf_vars, var_names):
                    setattr(self, var_name, tf_var)
                    operator_args[var_name] = tf_var

                tf_op = tf.no_op()

                for op_name, op in operations.items():

                    # store operator equations
                    self.operations[op_name] = op

                    with tf.control_dependencies([tf_op]):

                        # create operator
                        operator = Operator(expressions=op,
                                            expression_args=operator_args,
                                            tf_graph=self.tf_graph,
                                            key=op_name,
                                            variable_scope=self.key)

                        # collect tensorflow operator
                        tf_op = operator.create()

                    # bind newly created tf variables to node
                    for var_name, tf_var in operator.args.items():
                        if not hasattr(self, var_name):
                            setattr(self, var_name, tf_var)
                            operation_args[var_name] = tf_var

                # group tensorflow versions of all operators
                self.update = tf_op
