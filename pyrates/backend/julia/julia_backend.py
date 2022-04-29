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
        """Instantiates Julia backend.
        """

        # add user-provided operations to function dict
        julia_ops = julia_funcs.copy()
        if ops:
            julia_ops.update(ops)

        # set default float precision to float64
        kwargs["float_precision"] = "float64"

        # call parent method
        super().__init__(ops=julia_ops, imports=imports, file_ending='.jl', start_idx=1, add_hist_arg=True, **kwargs)

        # define julia-specific imports
        self._imports.pop(0)
        self._imports.append("using LinearAlgebra")

        # set up pyjulia
        from julia.api import Julia
        jl = Julia(runtime=kwargs.pop('julia_path'), compiled_modules=False)
        from julia import Main
        self._jl = Main
        self._no_vectorization = ["*(", "interp("]
        self._fcall = None

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

    def add_var_hist(self, lhs: str, delay: Union[ComputeVar, float], state_idx: Union[int, tuple], **kwargs):
        idx = self._process_idx(state_idx)
        d = self._process_delay(delay)
        self.add_code_line(f"{lhs} = hist((), t-{d}; idxs={idx})")

    def get_hist_func(self, y: np.ndarray):
        self._jl.eval(f"y_init = {y.tolist()}")
        hist = """
        function hist(p, t; idxs=nothing)
            return idxs == nothing ? y_init : y_init[idxs]
        end
        """
        self._jl.eval(hist)
        return self._jl.hist

    def create_index_str(self, idx: Union[str, int, tuple], separator: str = ',', apply: bool = True,
                         **kwargs) -> Tuple[str, dict]:

        if not apply:
            self._start_idx = 0
            idx, idx_dict = super().create_index_str(idx, separator, apply, **kwargs)
            self._start_idx = 1
            return idx, idx_dict
        else:
            return super().create_index_str(idx, separator, apply, **kwargs)

    def generate_func_tail(self, rhs_var: str = 'dy'):

        self.add_code_line(f"return {rhs_var}")
        self.remove_indent()
        self.add_code_line("end")

    def generate_func(self, func_name: str, to_file: bool = True, **kwargs):

        self._fcall = func_name

        # generate the current function string via the code generator
        if kwargs.pop('julia_ode', False):
            self.add_linebreak()
            self.add_code_line(f"function {func_name}_julia(dy, y, p, t)")
            self.add_indent()
            self.add_code_line(f"return {func_name}(t, y, dy, p...)")
            self.remove_indent()
            self.add_code_line("end")
        if kwargs.pop('julia_dde', False):
            self.add_linebreak()
            self.add_code_line(f"function {func_name}_julia(dy, y, h, p, t)")
            self.add_indent()
            self.add_code_line(f"return {func_name}(t, y, h, dy, p...)")
            self.remove_indent()
            self.add_code_line("end")
        func_str = self.generate()

        if to_file:

            # save rhs function to file
            file = f'{self.fdir}/{self._fname}{self._fend}' if self.fdir else f"{self._fname}{self._fend}"
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

        if 'julia' in solver:

            # solve via DifferentialEquations.jl
            self._jl.eval('using DifferentialEquations')

            if 'dde' in solver:

                # define wrapper function and solver family
                jfunc = f"""
                function julia_dderun(du,u,h,p,t)
                    return {self._fcall}(t,u,h,du,p...)
                end
                """

                # solve ivp via DifferentialEquations.jl solver
                self._jl.eval(jfunc)
                model = self._jl.DDEProblem(self._jl.julia_dderun, y0, args[0], [0.0, T], args[2:])
                method = kwargs.pop('method', 'Tsit5')
                solver = getattr(self._jl, method)
                solver = self._jl.MethodOfSteps(solver())
                atol, rtol = kwargs.pop('atol', 1e-6), kwargs.pop('rtol', 1e-3)
                results = self._jl.solve(model, solver, saveat=times, atol=atol, rtol=rtol)

            else:

                # define wrapper function and solver family
                jfunc = f"""
                function julia_oderun(du,u,p,t)
                    return {self._fcall}(t,u,du,p...)
                end
                """

                # solve ivp via DifferentialEquations.jl solver
                self._jl.eval(jfunc)
                model = self._jl.ODEProblem(self._jl.julia_oderun, y0, [0.0, T], args[1:])
                method = kwargs.pop('method', 'Tsit5')
                solver = getattr(self._jl, method)
                atol, rtol = kwargs.pop('atol', 1e-6), kwargs.pop('rtol', 1e-3)
                results = self._jl.solve(model, solver(), saveat=times, atol=atol, rtol=rtol)

            results = np.asarray(results).T

        else:

            # non-julia solver
            results = super()._solve(solver=solver, func=func, args=args, T=T, dt=dt, dts=dts, y0=y0, t0=t0,
                                     times=times, **kwargs)

        return results

    def _add_func_call(self, name: str, args: Iterable, return_var: str = 'dy'):
        self.add_code_line(f"function {name}({','.join(args)})")

    def _process_idx(self, idx: Union[Tuple[int, int], int, str, ComputeVar], **kwargs) -> str:

        if type(idx) is str and idx != ':' and ':' in idx:
            idx0, idx1 = idx.split(':')
            self._start_idx = 0
            idx0 = int(self._process_idx(idx0))
            idx1 = int(self._process_idx(idx1))
            self._start_idx = 1
            return self._process_idx((idx0, idx1))
        if type(idx) is ComputeVar and idx.name == "t" and idx.value >= self._start_idx:
            self._start_idx = 0
            idx_processed = super()._process_idx(idx=idx, **kwargs)
            self._start_idx = 1
            return idx_processed
        return super()._process_idx(idx=idx, **kwargs)

    @staticmethod
    def expr_to_str(expr: str, args: tuple):

        # replace power operator
        func = '**'
        while func in expr:
            expr = expr.replace(func, '^')

        return expr
