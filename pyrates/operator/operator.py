"""This module contains the operator class used to create executable operations from expression strings.
"""

# external imports
from typing import List, Optional
import tensorflow as tf

# pyrates internal imports
from pyrates.parser import EquationParser


# meta infos
__author__ = "Richard Gast, Daniel Rose"
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

        self.DEs = []
        self.updates = []
        self.key = key
        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()
        self.args = expression_args

        if dependencies is None:
            dependencies = []

        # parse expressions
        ###################

        with self.tf_graph.as_default():

            with tf.variable_scope(variable_scope):

                for i, expr in enumerate(expressions):

                    with tf.control_dependencies(dependencies):

                        # parse equation
                        parser = EquationParser(expr, self.args, engine='tensorflow', tf_graph=self.tf_graph)
                        self.args = parser.args

                        # collect tensorflow variables and update operations
                        if hasattr(parser, 'update'):
                            self.DEs.append((parser.target_var, parser.update))
                        else:
                            self.updates.append(parser.target_var)

    def create(self):
        """Create a single tensorflow operation for the set of parsed expressions.
        """

        with self.tf_graph.as_default():

            # group the tensorflow operations across expressions
            with tf.control_dependencies(self.updates):
                if len(self.DEs) > 0:
                    updates = tf.group([var.assign(upd) for var, upd in self.DEs], name=self.key)
                else:
                    updates = tf.group(self.updates, name=self.key)

        return updates
