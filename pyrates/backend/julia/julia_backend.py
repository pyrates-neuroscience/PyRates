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

"""Wraps Julia such that its low-level functions can be used by PyRates to create and simulate a compute graph.
"""

# pyrates internal _imports
import sys

from ..base import BaseBackend
from ..computegraph import ComputeVar
from .julia_funcs import julia_funcs

# external _imports
from typing import Optional, Dict, List, Callable, Iterable, Union, Tuple
import numpy as np

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


# backend classes
#################


class JuliaBackend(BaseBackend):

    def __init__(self,
                 ops: Optional[Dict[str, str]] = None,
                 imports: Optional[List[str]] = None,
                 **kwargs
                 ) -> None:
        """Instantiates tensorflow backend, i.e. a tensorflow graph.
        """

        # add user-provided operations to function dict
        julia_ops = julia_funcs.copy()
        if ops:
            julia_ops.update(ops)

        # set default float precision to float64
        kwargs["float_precision"] = "float64"

        # call parent method
        super().__init__(ops=julia_ops, imports=imports, file_ending='.jl', start_idx=1, **kwargs)

        # define fortran-specific imports
        self._imports.pop(0)
        self._imports.append("using LinearAlgebra")

        # set up pyjulia
        from julia.api import Julia
        jl = Julia(runtime=kwargs.pop('julia_path'), compiled_modules=False)
        from julia import Main
        self._jl = Main
        self._no_vectorization = ["*(", "interp("]

    def get_var(self, v: ComputeVar):
        v = super().get_var(v)
        dtype = v.dtype.name
        s = sum(v.shape)
        if s > 0:
            return v
        if 'float' in dtype:
            return float(v)
        if 'complex' in dtype:
            return complex(np.real(v), np.imag(v))
        return int(v)

    def add_var_update(self, lhs: ComputeVar, rhs: str, lhs_idx: Optional[str] = None, rhs_shape: Optional[tuple] = ()):

        super().add_var_update(lhs=lhs, rhs=rhs, lhs_idx=lhs_idx, rhs_shape=rhs_shape)
        if rhs_shape or lhs_idx:
            line = self.code.pop()
            lhs, rhs = line.split(' = ')
            if not any([rhs[:len(expr)] == expr for expr in self._no_vectorization]):
                rhs = f"@. {rhs}"
            self.add_code_line(f"{lhs} = {rhs}")

    def create_index_str(self, idx: Union[str, int, tuple], separator: str = ',', apply: bool = True,
                         **kwargs) -> Tuple[str, dict]:

        if not apply:
            self._start_idx = 0
            idx, idx_dict = super().create_index_str(idx, separator, apply, **kwargs)
            self._start_idx = 1
            return idx, idx_dict
        else:
            return super().create_index_str(idx, separator, apply, **kwargs)

    def generate_func_tail(self, rhs_var: str = None):

        self.add_code_line(f"return {rhs_var}")
        self.remove_indent()
        self.add_code_line("end")

    def generate_func(self, func_name: str, to_file: bool = True, **kwargs):

        # generate the current function string via the code generator
        if kwargs.pop('julia_diffeq', False):
            self.add_linebreak()
            self.add_code_line(f"function {func_name}_julia(dy, y, p, t)")
            self.add_indent()
            self.add_code_line(f"return {func_name}(t, y, dy, p...)")
            self.remove_indent()
            self.add_code_line("end")
        func_str = self.generate()

        if to_file:

            # save rhs function to file
            file = f'{self._fdir}/{self._fname}{self._fend}' if self._fdir else f"{self._fname}{self._fend}"
            with open(file, 'w') as f:
                f.writelines(func_str)
                f.close()

            # import function from file
            rhs_eval = self._jl.include(file)

        else:

            # just execute the function string, without writing it to file
            rhs_eval = self._jl.eval(func_str)

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

    def _add_func_call(self, name: str, args: Iterable):
        self.add_code_line(f"function {name}({','.join(args)})")

    def _process_idx(self, idx: Union[Tuple[int, int], int, str, ComputeVar], **kwargs) -> str:

        if type(idx) is str and ':' in idx:
            idx0, idx1 = idx.split(':')
            self._start_idx = 0
            idx0 = int(self._process_idx(idx0))
            idx1 = int(self._process_idx(idx1))
            self._start_idx = 1
            return self._process_idx((idx0, idx1))
        return super()._process_idx(idx=idx, **kwargs)

    @staticmethod
    def expr_to_str(expr: str, args: tuple):

        # replace power operator
        func = '**'
        while func in expr:
            expr = expr.replace(func, '^')

        return expr
