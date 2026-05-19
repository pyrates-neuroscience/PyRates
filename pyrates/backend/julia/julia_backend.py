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

    # Adds two Julia-native paths on top of the inherited euler/heun/scipy:
    #   'julia_ode' → DifferentialEquations.jl ODEProblem
    #   'julia_dde' → DifferentialEquations.jl DDEProblem (MethodOfSteps)
    # ``_validate_solver`` is overridden below to accept arbitrary
    # "julia_*" identifiers (any string containing 'julia') so users can pick
    # specific algorithms via the existing ``method=`` keyword.
    SUPPORTED_SOLVERS = ('euler', 'heun', 'scipy', 'julia_ode', 'julia_dde')

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
        self._no_vectorization = ["*(", "interp(", "hist("]
        self._fcall = None
        self._is_dde = False
        self._lags = []

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
        self._is_dde = True
        if isinstance(delay, float):
            self._lags.append(delay)
        elif type(delay) is ComputeVar and np.ndim(delay.value) == 0:
            self._lags.append(float(delay.value))
        idx = self._process_idx(state_idx)
        d = self._process_delay(delay)
        self.add_code_line(f"{lhs} = hist((), t-{d}; idxs={idx})")

    def get_hist_func(self, y: np.ndarray, t0: float = 0.0):
        # Julia DDE pre-history: constant initial condition for t <= t0.
        # During integration DifferentialEquations.jl provides its own
        # interpolant as `h`; this function is only called for t < t0.
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
        kwargs.pop('julia_ode', None)  # legacy flags, now auto-generated
        kwargs.pop('julia_dde', None)

        # append a DifferentialEquations.jl-compatible wrapper function.
        # For DDEs the wrapper exposes `h` (the solver interpolant) so the
        # RHS can call hist(p, t-d; idxs=i).  For plain ODEs it is omitted.
        self.add_linebreak()
        if self._is_dde:
            self.add_code_line(f"function {func_name}_julia(dy, y, h, p, t)")
            self.add_indent()
            self.add_code_line(f"return {func_name}(t, y, h, dy, p...)")
            self.remove_indent()
            self.add_code_line("end")
        else:
            self.add_code_line(f"function {func_name}_julia(dy, y, p, t)")
            self.add_indent()
            self.add_code_line(f"return {func_name}(t, y, dy, p...)")
            self.remove_indent()
            self.add_code_line("end")

        func_str = self.generate()

        if to_file:

            # save rhs function to file
            file = f'{self.fdir}/{self._fname}{self._fend}' if self.fdir else f"{self._fname}{self._fend}"
            with open(file, 'w') as f:
                f.writelines(func_str)

            # import all functions from file into Julia Main
            self._jl.include(file)

        else:

            # execute the function string directly
            self._jl.eval(func_str)

        # return the main RHS function object (wrapper is accessed via _fcall in _solve)
        rhs_eval = getattr(self._jl, func_name)
        return self._apply_decorator(rhs_eval, **kwargs)

    def _validate_solver(self, solver: str) -> None:
        # Accept any "julia*" string (the dispatch below uses ``'julia' in
        # solver``); fall through to the base check otherwise so plain
        # ``euler`` / ``heun`` / ``scipy`` keep working.
        if 'julia' in solver:
            return
        super()._validate_solver(solver)

    def _solve(self, solver: str, func: Callable, args: tuple, T: float, dt: float, dts: float, y0: np.ndarray,
               t0: np.ndarray, times: np.ndarray, **kwargs) -> np.ndarray:

        self._validate_solver(solver)

        if 'julia' in solver:

            # solve via DifferentialEquations.jl
            self._jl.eval('using DifferentialEquations')

            # retrieve the pre-generated DifferentialEquations.jl-compatible wrapper
            wrapper = getattr(self._jl, f'{self._fcall}_julia')

            method = kwargs.pop('method', 'Tsit5')
            atol, rtol = kwargs.pop('atol', 1e-6), kwargs.pop('rtol', 1e-3)

            # auto-detect DDE: _is_dde is set when add_var_hist was called,
            # or the user can still explicitly request it with solver='julia_dde'
            is_dde = self._is_dde or 'dde' in solver

            if is_dde:

                # args layout: (hist_julia_func, dy_zeros, p1, p2, ...)
                # hist_julia_func   → h argument of DDEProblem
                # dy_zeros          → internal buffer, not passed to Julia
                # p1, p2, ...       → parameter tuple for DDEProblem
                hist_func = args[0]
                params = args[2:]

                # pass constant_lags for discontinuity tracking when available
                lags = kwargs.pop('constant_lags', self._lags if self._lags else None)
                solve_kwargs = dict(saveat=times, atol=atol, rtol=rtol)
                if lags:
                    model = self._jl.DDEProblem(wrapper, y0, hist_func, [float(t0), T], params,
                                                constant_lags=list(lags))
                else:
                    model = self._jl.DDEProblem(wrapper, y0, hist_func, [float(t0), T], params)
                jl_solver = self._jl.MethodOfSteps(getattr(self._jl, method)())
                results = self._jl.solve(model, jl_solver, **solve_kwargs)

            else:

                # args layout: (dy_zeros, p1, p2, ...)
                # dy_zeros is skipped; p1, p2, ... form the parameter tuple
                params = args[1:]
                model = self._jl.ODEProblem(wrapper, y0, [float(t0), T], params)
                jl_solver = getattr(self._jl, method)()
                results = self._jl.solve(model, jl_solver, saveat=times, atol=atol, rtol=rtol)

            results = np.asarray(results).T

        else:

            # non-julia solver — fall back to Python/scipy solvers
            results = super()._solve(solver=solver, func=func, args=args, T=T, dt=dt, dts=dts, y0=y0, t0=t0,
                                     times=times, **kwargs)

        return results

    def _add_func_call(self, name: str, args: Iterable, return_var: str = 'dy'):
        self.add_code_line(f"function {name}({','.join(args)})")

    def _process_idx(self, idx: Union[Tuple[int, int], int, str, ComputeVar], **kwargs) -> str:

        if type(idx) is str and idx != ':' and ':' in idx:
            # Parse "a:b" range bounds with start_idx=0 (raw ints), then format
            # the tuple at start_idx=1 (Julia's 1-based, inclusive range).
            idx0, idx1 = idx.split(':')
            self._start_idx = 0
            idx0 = int(self._process_idx(idx0))
            idx1 = int(self._process_idx(idx1))
            self._start_idx = 1
            return self._process_idx((idx0, idx1))
        # No longer need the `idx.name == "t" and idx.value >= self._start_idx`
        # double-offset guard: BaseBackend._process_idx is now idempotent for
        # ComputeVar inputs (review §4.3).
        return super()._process_idx(idx=idx, **kwargs)

    @staticmethod
    def expr_to_str(expr: str, args: tuple):

        # replace power operator
        func = '**'
        while func in expr:
            expr = expr.replace(func, '^')

        return expr
