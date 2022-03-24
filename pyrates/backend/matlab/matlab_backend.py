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
from ..julia.julia_funcs import julia_funcs
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
        julia_funcs.update(matlab_ops)

        # call parent method
        super(JuliaBackend, self).__init__(ops=julia_funcs, imports=imports, file_ending='.m', start_idx=1,
                                           idx_left='(', idx_right=')', **kwargs)

        # define matlab-specific imports
        self._imports.pop(0)

        # create matlab session
        import matlab.engine
        self._constr = matlab
        self._matlab = matlab.engine.start_matlab()

    def add_var_update(self, lhs: ComputeVar, rhs: str, lhs_idx: Optional[str] = None, rhs_shape: Optional[tuple] = ()):

        super(JuliaBackend, self).add_var_update(lhs=lhs, rhs=rhs, lhs_idx=lhs_idx, rhs_shape=rhs_shape)
        line = self.code.pop()
        lhs, rhs = line.split(' = ')
        if rhs_shape or lhs_idx:
            rhs = self._matlab.vectorize(rhs)
        self.add_code_line(f"{lhs} = {rhs};")

    def generate_func_head(self, func_name: str, state_var: str = 'y', return_var: str = 'dy', func_args: list = None):

        helper_funcs = tuple(self._helper_funcs)
        self._helper_funcs = []
        fhead = super().generate_func_head(func_name, state_var, return_var, func_args)
        self._helper_funcs = list(helper_funcs)
        return fhead

    def generate_func_tail(self, rhs_var: str = None):

        self.remove_indent()
        self.add_code_line("end")
        self.add_code_line("")

        if self._helper_funcs:

            # add definitions of helper functions after the _imports
            for func in self._helper_funcs:
                self.add_code_line(func)
            self.add_linebreak()

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
            rhs_eval = eval(f"self._matlab.{self._fname}")

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

        # transform function arguments into matlab variables
        args_m = tuple([self._transform_to_mat(arg) for arg in args])

        # define wrapper function to ensure that python arrays are transformed into matlab variables
        def func_mat(t, y):
            y_m = self._transform_to_mat(y)
            t_m = self._transform_to_mat(t)
            return np.asarray(func(t_m, y_m, *args_m)).squeeze()

        # perform integration via scipy solver (mostly Runge-Kutta methods)
        if solver == 'euler':

            # solve ivp via forward euler method (fixed integration step-size)
            results = self._solve_euler(func_mat, (), T, dt, dts, y0, t0)

        elif solver == 'scipy':

            # solve ivp via scipy methods (solvers of various orders with adaptive step-size)
            from scipy.integrate import solve_ivp
            kwargs['t_eval'] = times

            # call scipy solver
            results = solve_ivp(fun=func_mat, t_span=(t0, T), y0=y0, first_step=dt, **kwargs)
            results = results['y'].T

        else:

            raise ValueError(f'Solver {solver} is not supported for this backend.')

        return results

    def _add_func_call(self, name: str, args: Iterable, return_var: str = 'dy'):
        self.add_code_line(f"function {return_var} = {name}({','.join(args)})")

    def _transform_to_mat(self, v: np.ndarray):
        if hasattr(v, 'shape') and sum(v.shape) > 0:
            if 'complex' in v.dtype.name:
                return self._constr.complex(v.real, v.imag)
            elif 'float' in v.dtype.name:
                return self._constr.double(v.tolist())
            else:
                return self._constr.int32(v.tolist())
        elif hasattr(v, 'shape'):
            if 'complex' in v.dtype.name:
                return self._constr.complex(v.real, v.imag)
            elif 'float' in v.dtype.name:
                return self._constr.double([v])
            else:
                return self._constr.int32([v])
        else:
            return v
