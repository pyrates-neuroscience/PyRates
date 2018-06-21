"""This module contains a solver class that manages the integration of all ODE's in a circuit.
"""

# external packages
from typing import Union
import tensorflow as tf

# pyrates internal imports
from pyrates.parser import RHSParser

# meta infos
__author__ = "Richard Gast"
__status__ = "Development"


#####################
# base solver class #
#####################


class Solver(object):

    def __init__(self,
                 rhs: Union[tf.Operation, tf.Tensor],
                 state_var: Union[tf.Variable, tf.Tensor],
                 dt: float,
                 tf_graph: tf.Graph
                 ) -> None:

        self.rhs = rhs
        self.state_var = state_var
        self.dt = dt
        self.integration_expressions = ["dt * rhs"]
        self.tf_graph = tf_graph

    def solve(self):

        with self.tf_graph.as_default():

            steps = []
            steps.append(tf.no_op())
            for expr in self.integration_expressions:
                parser = RHSParser(expr, {'dt': self.dt, 'rhs': self.rhs}, self.tf_graph)
                step = parser.transform()
                with tf.control_dependencies([steps[-1]]):
                    steps.append(self.state_var.assign_add(step))
            steps.pop(0)

        return tf.group(steps)
