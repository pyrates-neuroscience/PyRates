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
    """Base solver class (currently only implements basic forward euler).

    Parameters
    ----------
    rhs
        Tensorflow operation that represents right-hand side of a differential equation.
    state_var
        Tensorflow variable that should be integrated over time.
    dt
        Step-size of the integration over time [unit = s].
    tf_graph
        Tensorflow graph on which rhs and state_var have been created.

    Attributes
    ----------

    Methods
    -------

    References
    ----------

    Examples
    --------

    """
    def __init__(self,
                 rhs: Union[tf.Operation, tf.Tensor],
                 state_var: Union[tf.Variable, tf.Tensor],
                 dt: float,
                 tf_graph: tf.Graph
                 ) -> None:
        """Instantiates solver.
        """

        # initialize instance attributes
        ################################

        self.rhs = rhs
        self.state_var = state_var
        self.dt = dt
        self.tf_graph = tf_graph

        # define integration expression
        ###############################

        # TODO: Implement Butcher tableau and its translation into various solver algorithms
        self.integration_expressions = ["dt * rhs"]

    def solve(self) -> Union[tf.Operation, tf.Tensor]:
        """Creates tensorflow method for performing a single differentiation step.
        """

        with self.tf_graph.as_default():

            steps = []
            steps.append(tf.no_op())

            # go through integration expressions to solve DE
            ################################################

            for expr in self.integration_expressions:

                # parse the integration expression
                parser = RHSParser(expr, {'dt': self.dt, 'rhs': self.rhs}, self.tf_graph)
                step = parser.transform()

                # update the target state variable
                with tf.control_dependencies([steps[-1]]):
                    steps.append(self.state_var.assign_add(step))

            steps.pop(0)

        return tf.group(steps)
