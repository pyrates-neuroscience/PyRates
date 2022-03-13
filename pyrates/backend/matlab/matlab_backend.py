# -*- coding: utf-8 -*-
#
#
# PyRates software framework for flexible implementation of neural
# network model_templates and simulations. See also:
# https://github.com/pyrates-neuroscience/PyRates
#
# Copyright (C) 2017-2018 the original authors (Richard Gast and
# Daniel Rose), the Max-Planck-Institute for Human Cognitive Brain
# Sciences ("MPI CBS") and contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
# CITATION:
#
# Richard Gast and Daniel Rose et. al. in preparation

"""Wraps Matlab such that its low-level functions can be used by PyRates to create and simulate a compute graph.
"""

# pyrates internal _imports
import sys

from ..julia import JuliaBackend
from ..computegraph import ComputeVar
from .matlab_funcs import matlab_funcs

# external _imports
from typing import Optional, Dict, List, Callable, Iterable, Union, Tuple
import numpy as np

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


# backend classes
#################


class MatlabBackend(JuliaBackend):

    def __init__(self,
                 ops: Optional[Dict[str, str]] = None,
                 imports: Optional[List[str]] = None,
                 **kwargs
                 ) -> None:
        """Instantiates Matlab backend.
        """

        # add user-provided operations to function dict
        matlab_ops = matlab_funcs.copy()
        if ops:
            matlab_ops.update(ops)

        # define matlab-specific attributes
        kwargs['idx_left'] = '('
        kwargs['idx_right'] = ')'
        kwargs['start_idx'] = 1

        # call parent method
        super(JuliaBackend, self).__init__(ops=matlab_ops, imports=imports, file_ending='.jl', start_idx=1, **kwargs)

        # define matlab-specific imports
        self._imports.pop(0)

        # define which function calls should not be vectorized during code generation
        self._no_vectorization = ["*("]

        # create matlab session
        import matlab.engine as en
        self._matlab = en.start_matlab()

    def add_var_update(self, lhs: ComputeVar, rhs: str, lhs_idx: Optional[str] = None, rhs_shape: Optional[tuple] = ()):

        super(JuliaBackend, self).add_var_update(lhs=lhs, rhs=rhs, lhs_idx=lhs_idx, rhs_shape=rhs_shape)
        if rhs_shape or lhs_idx:
            line = self.code.pop()
            lhs, rhs = line.split(' = ')
            if not any([rhs[:len(expr)] == expr for expr in self._no_vectorization]):
                rhs = self._matlab.vectorize(rhs)
            self.add_code_line(f"{lhs} = {rhs}")

    def generate_func_tail(self, rhs_var: str = None):

        self.remove_indent()
        self.add_code_line("end")

    def generate_func(self, func_name: str, to_file: bool = True, **kwargs):

        # generate the current function string via the code generator
        func_str = self.generate()

        if to_file:

            # save rhs function to file
            file = f'{self._fdir}/{self._fname}{self._fend}' if self._fdir else f"{self._fname}{self._fend}"
            with open(file, 'w') as f:
                f.writelines(func_str)
                f.close()

            # import function from file
            if self._fdir:
                self._matlab.addpath(self._fdir, nargout=0)
            rhs_eval = exec(f"self._matlab.{self._fname}")

        else:

            # just execute the function string, without writing it to file
            rhs_eval = self._matlab.eval(func_str)

        # apply function decorator
        decorator = kwargs.pop('decorator', None)
        if decorator:
            decorator_kwargs = kwargs.pop('decorator_kwargs', dict())
            rhs_eval = decorator(rhs_eval, **decorator_kwargs)

        return rhs_eval

    def _solve(self, solver: str, func: Callable, args: tuple, T: float, dt: float, dts: float, y0: np.ndarray,
               t0: np.ndarray, times: np.ndarray, **kwargs) -> np.ndarray:

        # perform integration via scipy solver (mostly Runge-Kutta methods)
        if solver == 'euler':

            # solve ivp via forward euler method (fixed integration step-size)
            results = self._solve_euler(func, args, T, dt, dts, y0, t0)

        elif solver == 'scipy':

            # solve ivp via scipy methods (solvers of various orders with adaptive step-size)
            from scipy.integrate import solve_ivp
            kwargs['t_eval'] = times

            # call scipy solver
            results = solve_ivp(fun=func, t_span=(t0, T), y0=y0, first_step=dt, args=args, **kwargs)
            results = results['y'].T

        else:

            # solve ivp via DifferentialEquations.jl solver
            self._jl.eval('using DifferentialEquations')
            model = self._jl.ODEProblem(func, y0, [0.0, T], args[1:])
            method = kwargs.pop('method', 'Tsit5')
            if hasattr(self._jl, method):
                method = getattr(self._jl, method)
            else:
                method = self._jl.Tsit5
            results = self._jl.solve(model, method(), saveat=times, reltol=1e-6, abstol=1e-6)
            results = np.asarray(results).T

        return results

    def _add_func_call(self, name: str, args: Iterable, return_var: str = 'dy'):
        self.add_code_line(f"function {return_var} = {name}({','.join(args)})")
