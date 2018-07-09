"""This module contains the operator class used to create executable operations from expression strings.
"""

# external imports
from typing import List, Optional
import tensorflow as tf

# pyrates internal imports
from pyrates.parser import RHSParser, LHSParser

# meta infos
from pyrates.abc import AbstractBaseTemplate

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

                        lhs, rhs = expr.split('=')

                        # parse right-hand side and turn it into tensorflow operation
                        rhs_parser = RHSParser(rhs, expression_args, tf_graph)
                        rhs_tf_op = rhs_parser.transform()

                        # parse left-hand side and combine it with rhs tensorflow operation
                        lhs_parser = LHSParser(lhs, expression_args, rhs_tf_op, tf_graph)
                        lhs_tf_op, _ = lhs_parser.transform()

                        # collect resulting tensorflow operation
                        self.expressions.append(lhs_tf_op)

    def create(self):
        """Create a single tensorflow operation for the set of parsed expressions.
        """

        with self.tf_graph.as_default():

            # group the tensorflow operations across expressions
            grouped_expressions = tf.group(self.expressions, name=self.key)

        return grouped_expressions


class OperatorTemplate(AbstractBaseTemplate):
    """Generic template for an operator with a name, equation(s), variables and possible
    initialization conditions. The template can be used to create variations of a specific
    equation or variables."""

    def __init__(self, name: str, path: str, equation: str, variables: dict, description: str,
                 options: dict = None):
        """For now: only allow single equation in operator template."""

        super().__init__(name, path, description)

        self.equation = equation
        self.variables = variables

        self.options = options
        # if options:
        #     raise NotImplementedError
