"""

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

    def __init__(self,
                 operations: Dict[str, List[str]],
                 operation_args: dict,
                 key: str,
                 tf_graph: Optional[tf.Graph] = None
                 ) -> None:

        self.key = key
        self.operations = dict()
        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()

        with self.tf_graph.as_default():

            with tf.variable_scope(self.key):

                tf_vars, var_names = parse_dict(operation_args, self.key, self.tf_graph)

                operator_args = dict()

                for tf_var, var_name in zip(tf_vars, var_names):
                    setattr(self, var_name, tf_var)
                    operator_args[var_name] = tf_var

                tf_ops = []
                for op_name, op in operations.items():

                    self.operations[op_name] = op
                    operator = Operator(expressions=op,
                                        expression_args=operator_args,
                                        tf_graph=self.tf_graph,
                                        key=op_name,
                                        variable_scope=self.key,
                                        dependencies=tf_ops)
                    tf_ops.append(operator.create())

                self.update = tf.group(tf_ops, name='update')
