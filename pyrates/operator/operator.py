"""This module contains the operator class used to create executable operations from expression strings.
"""

# external imports
from typing import List, Optional
import tensorflow as tf

# pyrates internal imports
from pyrates.parser import EquationParser

# meta infos
__author__ = "Richard Gast"
__status__ = "Development"


class Operator(object):
    """Basic operator class that turns a list of expression strings into tensorflow operations using the variables
    provided in the expression args dictionary.

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
    def __init__(self, expressions: List[str], expression_args: dict, tf_graph: tf.Graph, key: str,
                 variable_scope: str, dependencies: Optional[list] = None):
        """Instantiates operator.
        """

        self.expressions = []
        self.key = key
        self.tf_graph = tf_graph

        # parse expressions
        ###################

        with self.tf_graph.as_default():

            with tf.variable_scope(variable_scope):

                with tf.control_dependencies(dependencies):

                    for i, expr in enumerate(expressions):

                        expr_parser = EquationParser(expr, expression_args, self.tf_graph)
                        op = expr_parser.lhs_update

                        # lhs, rhs = expr.split('=')
                        #
                        # # parse right-hand side and turn it into tensorflow operation
                        # rhs_parser = RHSParser(rhs, expression_args, tf_graph)
                        # rhs_tf_op = rhs_parser.transform()
                        #
                        # # parse left-hand side and combine it with rhs tensorflow operation
                        # lhs_parser = LHSParser(lhs, expression_args, rhs_tf_op, tf_graph)
                        # op, _ = lhs_parser.transform()

                        # collect resulting tensorflow operation

                        self.expressions.append(op)

    def create(self):
        """Create a single tensorflow operation for the set of parsed expressions.
        """

        with self.tf_graph.as_default():

            # group the tensorflow operations across expressions
            grouped_expressions = tf.group(self.expressions, name=self.key)

        return grouped_expressions
