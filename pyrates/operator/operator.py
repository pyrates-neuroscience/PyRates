"""

"""

# external imports
from typing import List, Optional
import tensorflow as tf

# pyrates internal imports
from pyrates.parser import RHSParser, LHSParser

# meta infos
__author__ = "Richard Gast"
__status__ = "Development"


class Operator(object):

    def __init__(self, expressions: List[str], expression_args: dict, tf_graph: tf.Graph, key: str,
                 variable_scope: str, dependencies: Optional[list] = None):

        self.expressions = []
        self.key = key
        self.tf_graph = tf_graph

        with self.tf_graph.as_default():

            with tf.variable_scope(variable_scope):

                with tf.control_dependencies(dependencies):

                    for i, expr in enumerate(expressions):

                        lhs, rhs = expr.split('=')

                        # right-hand side parsing
                        #########################

                        rhs_parser = RHSParser(rhs, expression_args, tf_graph)
                        rhs_tf_op = rhs_parser.parse()

                        # left-hand-side parsing
                        ########################

                        lhs_parser = LHSParser(lhs, expression_args, rhs_tf_op, tf_graph)
                        lhs_tf_op, _ = lhs_parser.parse()

                        self.expressions.append(lhs_tf_op)

    def create(self):

        with self.tf_graph.as_default():
            grouped_expressions = tf.group(self.expressions, name=self.key)

        return grouped_expressions
