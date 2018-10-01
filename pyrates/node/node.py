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


class Node(dict):
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
                 op_order: list,
                 key: str,
                 tf_graph: Optional[tf.Graph] = None
                 ) -> None:
        """Instantiates node.
        """

        super().__init__()

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
                for op_name in op_order:

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
                        self.update({f'{op_name}/{var_name}': tf_var})
                        op_args_tf[var_name] = {'var': tf_var, 'dependency': False}

                    # set input dependencies
                    ########################

                    for var_name, inp in op['inputs'].items():

                        # collect input variable calculation operations
                        out_ops = []
                        out_vars = []
                        out_var_idx = []

                        for i, inp_op in enumerate(inp['sources']):

                            if type(inp_op) is list and len(inp_op) == 1:
                                inp_op = inp_op[0]

                            if type(inp_op) is str:

                                out_name = operations[inp_op]['output']
                                if '[' in out_name:
                                    idx_start, idx_stop = out_name.find('['), out_name.find(']')
                                    out_var_idx.append(out_name[idx_start+1:idx_stop])
                                    out_var = f"{inp_op}/{out_name[:idx_start]}"
                                else:
                                    out_var_idx.append(None)
                                    out_var = f"{inp_op}/{out_name}"
                                if out_var not in operator_args.keys():
                                    raise ValueError(f"Invalid dependencies found in operator: {op['equations']}. Input"
                                                     f" Variable {var_name} has not been calculated yet.")
                                out_ops.append(operator_args[out_var]['op'])
                                out_vars.append(operator_args[out_var]['var'])

                            else:

                                out_vars_tmp = []
                                out_var_idx_tmp = []
                                out_ops_tmp = []

                                for inp_op_tmp in inp_op:

                                    out_name = operations[inp_op_tmp]['output']
                                    if '[' in out_name:
                                        idx_start, idx_stop = out_name.find('['), out_name.find(']')
                                        out_var_idx.append(out_name[idx_start + 1:idx_stop])
                                        out_var = f"{inp_op_tmp}/{out_name[:idx_start]}"
                                    else:
                                        out_var_idx_tmp.append(None)
                                        out_var = f"{inp_op_tmp}/{out_name}"
                                    if out_var not in operator_args.keys():
                                        raise ValueError(
                                            f"Invalid dependencies found in operator: {op['equations']}. Input"
                                            f" Variable {var_name} has not been calculated yet.")
                                    out_ops_tmp.append(operator_args[out_var]['op'])
                                    out_vars_tmp.append(operator_args[out_var]['var'])

                                tf_var_tmp = tf.parallel_stack(out_vars_tmp)
                                if inp['reduce_dim'][i]:
                                    tf_var_tmp = tf.reduce_sum(tf_var_tmp, 0)
                                else:
                                    tf_var_tmp = tf.reshape(tf_var_tmp,
                                                            shape=(tf_var_tmp.shape[0] * tf_var_tmp.shape[1],))

                                out_vars.append(tf_var_tmp)
                                out_var_idx.append(out_var_idx_tmp)
                                out_ops.append(tf.group(out_ops_tmp))

                        # add inputs to argument dictionary (in reduced or stacked form)
                        if len(out_vars) > 1:

                            min_shape = min([outvar.shape[0] for outvar in out_vars])
                            out_vars_new = []
                            for out_var in out_vars:
                                shape = out_var.shape[0]
                                if shape > min_shape:
                                    if shape % min_shape != 0:
                                        raise ValueError(f"Shapes of inputs do not match: "
                                                         f"{inp['sources']} cannot be stacked.")
                                    multiplier = shape // min_shape
                                    for j in range(multiplier):
                                        out_vars_new.append(out_var[j*min_shape:(j+1)*min_shape])
                                else:
                                    out_vars_new.append(out_var)

                            tf_var = tf.parallel_stack(out_vars_new)
                            if type(inp['reduce_dim']) is bool and inp['reduce_dim']:
                                tf_var = tf.reduce_sum(tf_var, 0)
                            else:
                                tf_var = tf.reshape(tf_var, shape=(tf_var.shape[0]*tf_var.shape[1],))

                            tf_op = tf.group(out_ops)

                        else:

                            tf_var = out_vars[0]
                            tf_op = out_ops[0]

                        op_args_tf[var_name] = {'dependency': True,
                                                'var': tf_var,
                                                'op': tf_op}

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
                            self.update({var_name: tf_var['var']})
                        operator_args[f'{op_name}/{var_name}'] = tf_var

                    # handle dependencies
                    operator_args[f"{op_name}/{op['output']}"]['op'] = tf_ops[-1]
                    for arg in operator_args.values():
                        arg['dependency'] = False

                # group tensorflow versions of all operators
                self.step = tf.group(tf_ops, name=f"{self.key}_update")
