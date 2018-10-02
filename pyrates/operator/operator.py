"""This module contains the operator class used to create executable operations from expression strings.
"""

# external imports
from typing import List, Optional
import tensorflow as tf

# pyrates internal imports
from pyrates.parser import parse_equation

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
                 variable_scope: str, tf_graph: Optional[tf.Graph] = None):
        """Instantiates operator.
        """

        self.updates = []
        self.key = key
        self.tf_graph = tf_graph if tf_graph else tf.get_default_graph()
        self.args = expression_args

        # parse expressions
        ###################

        with self.tf_graph.as_default():

            with tf.variable_scope(variable_scope):

                for i, expr in enumerate(expressions):

                    # parse equation
                    update, self.args = parse_equation(expr, self.args, tf_graph=self.tf_graph)
                    self.updates.append(update)

    def create(self):
        """Create a single tensorflow operation for the set of parsed expressions.
        """

        with self.tf_graph.as_default():

            # check which rhs evaluations still have to be assigned to lhs
            updated = []
            not_updated = []
            for upd in self.updates:
                if upd[1] is None:
                    updated.append(upd[0])
                else:
                    not_updated.append(upd)

            # group the tensorflow operations across expressions
            with tf.control_dependencies(updated):
                for upd in not_updated:
                    updated.append(upd[0].assign(upd[1]))

        return tf.group(updated, name=f'{self.key}_updates')
