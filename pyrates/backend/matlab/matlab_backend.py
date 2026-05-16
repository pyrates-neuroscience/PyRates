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
                                           idx_left='(', idx_right=')', add_hist_arg=True, **kwargs)

        # define matlab-specific imports
        self._imports.pop(0)

        # create matlab session
        import matlab.engine
        self._constr = matlab
        self._matlab = matlab.engine.start_matlab()

        self._nlags = 1
        self._is_dde = False
        self._lag_names = {}
        self._param_names = []
        self._fcall = None

    def add_var_update(self, lhs: ComputeVar, rhs: str, lhs_idx: Optional[str] = None, rhs_shape: Optional[tuple] = ()):

        super(JuliaBackend, self).add_var_update(lhs=lhs, rhs=rhs, lhs_idx=lhs_idx, rhs_shape=rhs_shape)
        line = self.code.pop()
        lhs, rhs = line.split(' = ')
        if rhs_shape or lhs_idx:
            rhs = self._matlab.vectorize(rhs)
        self.add_code_line(f"{lhs} = {rhs};")

    def add_var_hist(self, lhs: str, delay: Union[ComputeVar, float], state_idx: Union[int, tuple], **kwargs):
        self._is_dde = True
        idx = self._process_idx(state_idx)
        delay_name = None
        if type(delay) is ComputeVar:
            delay_name = delay.name
            delay_val = float(delay.value[0] if delay.shape else delay.value)
        else:
            delay_val = float(delay)
        if delay_val not in self.lags:
            self.lags[delay_val] = self._nlags
            self._nlags += 1
        delay_idx = self.lags[delay_val]
        if delay_name is not None and delay_idx not in self._lag_names:
            self._lag_names[delay_idx] = delay_name

        self.add_code_line(f"{lhs} = hist({idx}, {delay_idx});")

    def generate_func_head(self, func_name: str, state_var: str = 'y', return_var: str = 'dy', func_args: list = None,
                           add_hist_func: bool = True):

        helper_funcs = tuple(self._helper_funcs)
        self._helper_funcs = []
        fhead = super().generate_func_head(func_name, state_var, return_var, func_args, add_hist_func=add_hist_func)
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

        self._fcall = func_name

        # capture parameter names for DDE-BIFTOOL wrapper generation
        func_args = kwargs.pop('func_args', None)
        if func_args is not None:
            self._param_names = [a if isinstance(a, str) else a.name for a in func_args]
        kwargs.pop('state_vars', None)

        # generate the current function string via the code generator
        func_str = self.generate()

        if to_file:

            # save rhs function to file
            file = f'{self.fdir}/{self._fname}{self._fend}' if self.fdir else f"{self._fname}{self._fend}"
            with open(file, 'w') as f:
                f.writelines(func_str)

            # import function from file
            if self.fdir:
                self._matlab.addpath(self.fdir, nargout=0)
            rhs_eval = eval(f"self._matlab.{self._fname}")

            # generate DDE-BIFTOOL compatibility files when needed
            if self._is_dde:
                self.generate_biftool_funcs(func_name)

        else:

            # just execute the function string, without writing it to file
            rhs_eval = self._matlab.eval(func_str)

        # apply function decorator
        decorator = kwargs.pop('decorator', None)
        if decorator:
            decorator_kwargs = kwargs.pop('decorator_kwargs', dict())
            rhs_eval = decorator(rhs_eval, **decorator_kwargs)

        return rhs_eval

    def generate_biftool_funcs(self, func_name: str):
        """Generate DDE-BIFTOOL-compatible sys_rhs and sys_tau wrapper files.

        DDE-BIFTOOL sys_rhs convention: f(xx, par) where xx(:,1) = y(t) and
        xx(:,k+1) = y(t - tau_k).  sys_tau returns the 1-based indices of the
        delay parameters inside the par vector.
        """
        base = f'{self.fdir}/{func_name}' if self.fdir else func_name

        # 1-based index of each parameter in the par vector
        param_idx = {name: i + 1 for i, name in enumerate(self._param_names)}

        # delay parameter indices in par, ordered by lag index
        tau_indices = []
        for lag_idx in sorted(self._lag_names.keys()):
            dname = self._lag_names[lag_idx]
            if dname in param_idx:
                tau_indices.append(param_idx[dname])

        # extract parameters from par inside the biftool wrapper
        param_assigns = '\n'.join(
            f'    {name} = par({i + 1});'
            for i, name in enumerate(self._param_names)
        )
        param_call = ', '.join(self._param_names)

        biftool_rhs = (
            f"function dy = {func_name}_biftool(xx, par)\n"
            f"    % DDE-BIFTOOL sys_rhs wrapper: xx(:,1)=y(t), xx(:,k+1)=y(t-tau_k)\n"
            f"    y = xx(:, 1);\n"
            f"    hist = xx(:, 2:end);\n"
            f"{param_assigns}\n"
            f"    dy = zeros(size(y));\n"
            f"    dy = {func_name}(0.0, y, hist, dy, {param_call});\n"
            f"end\n"
        )

        tau_inds_str = ', '.join(str(i) for i in tau_indices)
        biftool_tau = (
            f"function tau_inds = {func_name}_tau()\n"
            f"    % 1-based indices of delay parameters in the par vector\n"
            f"    tau_inds = [{tau_inds_str}];\n"
            f"end\n"
        )

        with open(f'{base}_biftool.m', 'w') as f:
            f.write(biftool_rhs)
        with open(f'{base}_tau.m', 'w') as f:
            f.write(biftool_tau)

    @staticmethod
    def to_file(fn: str, **kwargs):
        from scipy.io import savemat
        savemat(f"{fn}.mat", mdict=kwargs)

    @staticmethod
    def get_hist_func(y: np.ndarray, t0: float = 0.0):
        return lambda t: y[:]

    def _solve(self, solver: str, func: Callable, args: tuple, T: float, dt: float, dts: float, y0: np.ndarray,
               t0: np.ndarray, times: np.ndarray, **kwargs) -> np.ndarray:

        # transform function arguments into matlab variables
        args_m = tuple([self._transform_to_mat(arg) for arg in args])

        # define wrapper function to ensure that python arrays are transformed into matlab variables
        def func_mat(t, y):
            y_m = self._transform_to_mat(y)
            t_m = self._transform_to_mat(t)
            return np.asarray(func(t_m, y_m, *args_m)).squeeze()

        return super()._solve(solver=solver, func=func_mat, args=(), T=T, dt=dt, dts=dts, y0=y0, t0=t0, times=times,
                              **kwargs)

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
