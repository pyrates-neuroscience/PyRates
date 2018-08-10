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
    def __init__(self, expressions: List[str], expression_args: dict, key: str,
                 variable_scope: str, dependencies: Optional[list] = None, tf_graph: Optional[tf.Graph] = None):
        """Instantiates operator.
        """

        self.expressions = []
        self.key = key
        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()

        if not dependencies:
            dependencies = []

        # parse expressions
        ###################

        with self.tf_graph.as_default():

            with tf.variable_scope(variable_scope):

                for i, expr in enumerate(expressions):

                    with tf.control_dependencies(dependencies + self.expressions):

                        expr_parser = EquationParser(expr, expression_args, engine='tensorflow', tf_graph=self.tf_graph)
                        op = expr_parser.lhs_update

                        # collect resulting tensorflow operation
                        self.expressions.append(op)

    def create(self):
        """Create a single tensorflow operation for the set of parsed expressions.
        """

        with self.tf_graph.as_default():

            # group the tensorflow operations across expressions
            grouped_expressions = tf.group(self.expressions, name=self.key)

        return grouped_expressions
