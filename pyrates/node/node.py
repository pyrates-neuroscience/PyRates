"""This module contains the node class used to create a network node from a set of operations.
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
                 operations: dict,
                 operation_args: dict,
                 key: str,
                 tf_graph: Optional[tf.Graph] = None
                 ) -> None:
        """Instantiates node.
        """

        self.key = key
        self.operations = dict()
        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()

        # create tensorflow operations/variables on graph
        #################################################

        with self.tf_graph.as_default():

            with tf.variable_scope(self.key):

                # handle operation arguments
                ############################

                operator_args = dict()

                # instantiate operations
                ########################

                tf_ops = []
                for op_name in operations['op_order']:

                    op = operations[op_name]

                    # store operator equations
                    self.operations[op_name] = op['equations']

                    # extract operator-specific arguments from dict
                    op_args_raw = {}
                    for key, val in operation_args.items():
                        op_name_tmp, var_name = key.split('/')
                        if op_name == op_name_tmp or 'all_ops' in op_name_tmp:
                            op_args_raw[var_name] = val

                    # get tensorflow variables and the variable names from operation_args
                    tf_vars, var_names = parse_dict(var_dict=op_args_raw,
                                                    var_scope=op_name,
                                                    tf_graph=self.tf_graph)

                    # bind tensorflow variables to node and save them in dictionary for the operator class
                    op_args_tf = {}
                    for tf_var, var_name in zip(tf_vars, var_names):
                        setattr(self, f'{op_name}/{var_name}', tf_var)
                        op_args_tf[var_name] = {'var': tf_var, 'dependency': False}

                    # set input dependencies
                    ########################

                    for var_name, inp in op['inputs'].items():

                        # collect input variable calculation operations
                        out_ops = []
                        out_vars = []
                        for inp_op in inp['in_col']:
                            out_name = operations[inp_op]['output']
                            out_var = f"{inp_op}/{out_name}"
                            if out_var not in operator_args.keys():
                                raise ValueError(f"Invalid dependencies found in operator: {op['equations']}. Input "
                                                 f"Variable {var_name} has not been calculated yet.")
                            out_ops.append(operator_args[out_var]['op'])
                            out_vars.append(operator_args[out_var]['var'])

                        # add inputs to argument dictionary (in reduced or stacked form)
                        min_shape = min([outvar.shape[0] for outvar in out_vars])
                        out_vars_new = []
                        for out_var in out_vars:
                            shape = out_var.shape[0]
                            if shape > min_shape:
                                if shape % min_shape != 0:
                                    raise ValueError(f"Shapes of inputs do not match: "
                                                     f"{inp['in_col']} cannot be stacked.")
                                multiplier = shape // min_shape - 1
                                for i in range(multiplier):
                                    out_var_tmp = out_var[i*min_shape:(i+1)*min_shape]
                                    out_vars_new.append(out_var_tmp)
                            else:
                                out_vars_new.append(out_var)

                        tf_var = tf.stack(out_vars_new)
                        if inp['reduce']:
                            tf_var = tf.reduce_sum(tf_var, 0)
                        else:
                            tf_var = tf.reshape(tf_var, shape=(tf_var.shape[0]*tf_var.shape[1], 1))
                        op_args_tf[var_name] = {'dependency': True,
                                                'var': tf_var,
                                                'op': tf.group(out_ops)}

                    # create operator
                    operator = Operator(expressions=op['equations'],
                                        expression_args=op_args_tf,
                                        tf_graph=self.tf_graph,
                                        key=op_name,
                                        variable_scope=op_name)

                    # collect tensorflow operator
                    tf_ops.append(operator.create())

                    # bind newly created tf variables to node
                    for var_name, tf_var in operator.args.items():
                        if not hasattr(self, var_name):
                            setattr(self, var_name, tf_var['var'])
                        operator_args[f'{op_name}/{var_name}'] = tf_var

                    # handle dependencies
                    operator_args[f"{op_name}/{op['output']}"]['op'] = tf_ops[-1]
                    for arg in operator_args.values():
                        arg['dependency'] = False

                # group tensorflow versions of all operators
                self.update = tf.group(tf_ops, name=f"{self.key}_update")
