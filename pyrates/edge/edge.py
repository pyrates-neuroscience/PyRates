"""

"""

# external imports
import tensorflow as tf
from typing import Optional, Union, List

# pyrates imports
from pyrates.operator import Operator
from pyrates.parser import parse_dict, EquationParser
from pyrates.node import Node

# meta infos

__author__ = "Richard Gast"
__status__ = "Development"


class Edge(object):

    def __init__(self,
                 source: Node,
                 target: Node,
                 coupling_op: List[str],
                 coupling_op_args: dict,
                 key: Optional[str] = None,
                 tf_graph: Optional[tf.Graph] = None):

        self.operator = coupling_op
        self.key = key if key else 'edge0'
        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()

        with self.tf_graph.as_default():

            with tf.variable_scope(self.key):

                # replace the coupling operator arguments with the fields from source and target
                inp = getattr(target, coupling_op_args['input'])
                outp = getattr(source, coupling_op_args['output'])
                coupling_op_args['input'] = {'variable_type': 'tensorflow', 'func': inp}
                coupling_op_args['output'] = {'variable_type': 'tensorflow', 'func': outp}

                tf_vars, var_names = parse_dict(coupling_op_args, self.key, self.tf_graph)

                operator_args = dict()

                for tf_var, var_name in zip(tf_vars, var_names):
                    setattr(self, var_name, tf_var)
                    operator_args[var_name] = tf_var

                operator = Operator(expressions=coupling_op,
                                    expression_args=operator_args,
                                    tf_graph=self.tf_graph,
                                    key=self.key,
                                    variable_scope=self.key)

                self.project = operator.create()
