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

                operator_args = dict()

                # instantiate operations
                ########################

                tf_ops = []
                for op_name in coupling_ops['op_order']:

                    op = coupling_ops[op_name]

                    # store operator equations
                    self.operations[op_name] = op['equations']

                    # extract operator-specific arguments from dict
                    op_args_raw = {}
                    for key, val in co_args.items():
                        op_name_tmp, var_name = key.split('/')
                        if op_name == op_name_tmp or 'all_ops' in op_name_tmp:
                            op_args_raw[var_name] = val

                    # create tensorflow variables from the additional operator args
                    tf_vars, var_names = parse_dict(var_dict=op_args_raw,
                                                    var_scope=op_name,
                                                    tf_graph=self.tf_graph)

                    # bind operator args to edge
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
                            out_name = co_args[inp_op]['output']
                            out_var = f"{inp_op}/{out_name}"
                            if out_var not in operator_args.keys():
                                raise ValueError(
                                    f"Invalid dependencies found in operator: {op['equations']}. Input "
                                    f"Variable {var_name} has not been calculated yet.")
                            out_ops.append(operator_args[out_var]['op'])
                            out_vars.append(operator_args[out_var]['var'])

                        # add inputs to argument dictionary (in reduced or stacked form)
                        if len(out_vars) > 1:
                            tf_var = tf.stack(out_vars)
                            if inp['reduce']:
                                tf_var = tf.reduce_sum(tf_var, 0)
                            tf_op = tf.group(out_ops)
                        else:
                            tf_var = out_vars[0]
                            tf_op = out_ops[0]

                        op_args_tf[var_name] = {'dependency': True,
                                                'var': tf_var,
                                                'op': tf_op}

                    # create coupling operator
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
                    if '/' not in op['output']:
                        operator_args[f"{op_name}/{op['output']}"]['op'] = tf_ops[-1]
                        for arg in operator_args.values():
                            arg['dependency'] = False

                # collect tensorflow operations of that edge
                self.project = tf.group(tf_ops, name=f"{self.key}_project")
