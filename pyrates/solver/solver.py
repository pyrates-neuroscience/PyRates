"""This module contains a solver class that manages the integration of all ODE's in a circuit.
"""

# external packages
import scipy.integrate as integ
import numpy as np
from typing import Callable, Union, Optional

# pyrates internal imports

# meta infos
__author__ = "Richard Gast"
__status__ = "Development"


#####################
# base solver class #
#####################


class Solver(object):
    """Base solver class that takes the right-hand side of an ODE as argument.

    Parameters
    ----------
    f
        Right-hand side of the ODE.
    y0
        Initial state of dependent variable(s) of ODE.

    """

    def __init__(self,
                 f: Callable,
                 y0: Optional[Union[float, np.ndarray]] = None,
                 **solver_kwargs
                 ):
        """Instantiates base solver.
        """

        self.func = f
        self.solver = integ.OdeSolver(fun=f, t0=0., y0=y0, t_bound=1e5, **solver_kwargs)

    def solve(self,
              y_old: Union[float, np.ndarray],
              step_size: float
              ) -> Union[float, np.ndarray]:
        """Solves the ODE for a time-interval of dt.

        Parameters
        ----------
        y_old
            Old value of the dependent variable the ODE is solved for.
        step_size
            Time [unit = s] that the ODE solution should advance.

        Returns
        -------
        Union[float, np.ndarray]
            New value of y after the integration over dt.

        """

        t = 0.
        while t < step_size:
            self.solver.step()
            t += self.solver.step_size

        return self.solver.y


##############################
# Forward Euler solver class #
##############################


class ForwardEuler(Solver):
    """Solves ODE using a simple forward Euler method.

    See Also
    --------
    :class:`Solver` docstring for a thorough description of the input parameters and methods.

    """

    def __init__(self,
                 f: Callable,
                 y0: Optional[Union[float, np.ndarray]] = None,
                 **solver_kwargs
                 ):
        """Instantiates base solver.
        """

        self.func = f

    def solve(self,
              y_old: Union[float, np.ndarray],
              step_size: float
              ) -> Union[float, np.ndarray]:
        """Solves the ODE for a time-interval of dt.

        Parameters
        ----------
        y_old
            Old value of the dependent variable the ODE is solved for.
        step_size
            Time [unit = s] that the ODE solution should advance.

        Returns
        -------
        Union[float, np.ndarray]
            New value of y after the integration over dt.

        """

        return y_old + step_size * self.func(step_size, y_old)


################################
# various scipy solver classes #
################################


class RK23(Solver):
    """Solves ODE using an explicit Runge-Kutta method of order 3(2).

    See Also
    --------
    :class:`Solver` docstring for a thorough description of the input parameters and methods.

    """

    def __init__(self,
                 f: Callable,
                 y0: Union[float, np.ndarray],
                 **solver_kwargs):
        """Instantiates base solver.
        """

        self.func = f
        self.solver = integ.RK23(fun=f, t0=0., y0=y0, t_bound=1e5, **solver_kwargs)


class RK45(Solver):
    """Solves ODE using an explicit Runge-Kutta method of order 5(4).

    See Also
    --------
    :class:`Solver` docstring for a thorough description of the input parameters and methods.

    """

    def __init__(self,
                 f: Callable,
                 y0: Union[float, np.ndarray],
                 **solver_kwargs):
        """Instantiates base solver.
        """

        self.func = f
        self.solver = integ.RK45(fun=f, t0=0., y0=y0, t_bound=1e5, **solver_kwargs)


class LSODA(Solver):
    """Solves ODE using the LSODA method.

    See Also
    --------
    :class:`Solver` docstring for a thorough description of the input parameters and methods.

    """

    def __init__(self,
                 f: Callable,
                 y0: Union[float, np.ndarray],
                 **solver_kwargs):
        """Instantiates base solver.
        """

        self.func = f
        self.solver = integ.LSODA(fun=f, t0=0., y0=y0, t_bound=1e5, **solver_kwargs)


class BDF(Solver):
    """Solves ODE using the LSODA method.

    See Also
    --------
    :class:`Solver` docstring for a thorough description of the input parameters and methods.

    """

    def __init__(self,
                 f: Callable,
                 y0: Union[float, np.ndarray],
                 **solver_kwargs):
        """Instantiates base solver.
        """

        self.func = f
        self.solver = integ.BDF(fun=f, t0=0., y0=y0, t_bound=1e5, **solver_kwargs)


class Radau(Solver):
    """Solves ODE using the LSODA method.

    See Also
    --------
    :class:`Solver` docstring for a thorough description of the input parameters and methods.

    """

    def __init__(self,
                 f: Callable,
                 y0: Union[float, np.ndarray],
                 **solver_kwargs):
        """Instantiates base solver.
        """

        self.func = f
        self.solver = integ.Radau(fun=f, t0=0., y0=y0, t_bound=1e5, **solver_kwargs)
