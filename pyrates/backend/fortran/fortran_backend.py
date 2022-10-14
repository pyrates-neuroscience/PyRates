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

"""Wraps fortran such that it's low-level functions can be used by PyRates to create and simulate a compute graph.
"""

# pyrates internal _imports
from ..base import BaseBackend
from ..computegraph import ComputeVar
from .fortran_funcs import fortran_funcs
from ..parser import replace

# external _imports
from typing import Optional, Dict, List, Callable, Iterable, Union, Tuple
import os
from numpy import f2py
import numpy as np

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


# backend classes
#################


class FortranBackend(BaseBackend):

    n1 = 62
    n2 = 72
    linebreak_start = "     & "
    linebreak_end = "&"

    def __init__(self,
                 ops: Optional[Dict[str, str]] = None,
                 imports: Optional[List[str]] = None,
                 **kwargs
                 ) -> None:
        """Instantiates Fortran backend.
        """

        # add user-provided operations to function dict
        fort_ops = fortran_funcs.copy()
        if ops:
            fort_ops.update(ops)

        # call parent method
        super().__init__(ops=fort_ops, imports=imports, file_ending='.f90', idx_left='(', idx_right=')', start_idx=1,
                         **kwargs)

        self._op_calls = {}
        self._var_declaration_info = {}

        # define fortran-specific imports
        self._imports.pop(0)
        self._imports.append("double precision :: PI = 4.0*atan(1.0)")

    def add_var_update(self, lhs: ComputeVar, rhs: str, lhs_idx: Optional[str] = None, rhs_shape: Optional[tuple] = ()):
        self.register_vars([lhs])
        super().add_var_update(lhs, rhs, lhs_idx, rhs_shape)

    def create_index_str(self, idx: Union[str, int, tuple], separator: str = ',', apply: bool = True,
                         **kwargs) -> Tuple[str, dict]:

        if not apply:
            self._start_idx = 0
            idx, idx_dict = super().create_index_str(idx, separator, apply, **kwargs)
            self._start_idx = 1
            return idx, idx_dict
        else:
            return super().create_index_str(idx, separator, apply, **kwargs)

    def generate_func_head(self, func_name: str, state_var: str = 'y', return_var: str = 'dy', func_args: list = None,
                           add_hist_func: bool = False):

        # finalize list of arguments for the function call
        if func_args:
            self.register_vars(func_args)
            func_args = [arg.name for arg in func_args]
        else:
            func_args = []
        state_vars = ['t', state_var]
        if add_hist_func:
            state_vars.append('hist')
        _, indices = np.unique(func_args, return_index=True)
        func_args = state_vars + [func_args[idx] for idx in np.sort(indices)]

        # define module
        self.add_code_line(f"module {self._fname}")
        self.add_linebreak()

        # add global variable definitions and other imports
        for imp in self._imports:
            self.add_code_line(imp)

        # start function definition section
        self.add_linebreak()
        self.add_code_line("contains")
        self.add_linebreak()

        # add function header
        self.add_linebreak()
        self._add_func_call(name=func_name, args=func_args, return_var=return_var)

        return func_args

    def generate_func_tail(self, rhs_var: str = None):

        # end the subroutine
        self.add_code_line(f"end subroutine")
        self.add_linebreak()
        self.add_linebreak()

        # add definitions of helper functions after the main run function
        for func in self._helper_funcs:
            self.add_code_line(func)
            self.add_linebreak()

        # end the module
        self.add_code_line(f"end module")

    def add_code_line(self, code_str):
        """Add code line string to code.
        """
        for code in code_str.split('\n'):
            if self.linebreak_end not in code:
                code = code.replace('\t', '')
                code = '\t' * self.lvl + code
            if self.break_line(code):
                idx = self._find_first_op(code, start=len(self.linebreak_start),
                                          stop=self.n2 - len(self.linebreak_end))
                self.add_code_line(f'{code[0:idx]}{self.linebreak_end}')
                code = f"{self.linebreak_start}{code[idx:]}"
                self.add_code_line(code)
            else:
                self.code.append(code)

    def break_line(self, code: str):
        n = len(code)
        if n > self.n2:
            return True
        if n > self.n1:
            if self.linebreak_start in code:
                if self.linebreak_end in code[len(self.linebreak_start):]:
                    if n - len(self.linebreak_start) - len(self.linebreak_end) < self.n2:
                        return False
                    return True
                if n - len(self.linebreak_start) < self.n2:
                    return False
                return True
            if self.linebreak_end in code:
                if n - len(self.linebreak_end) < self.n2:
                    return False
                return True
            return False
        return False

    def generate_func(self, func_name: str, to_file: bool = True, func_args: tuple = (), state_vars: tuple = (),
                      **kwargs):

        if to_file:
            file = f'{self.fdir}/{self._fname}{self._fend}' if self.fdir else f'{self._fname}{self._fend}'
        else:
            file = None

        # generate the final string representing the function file
        auto_compatible = kwargs.pop('auto', False)
        if auto_compatible:

            # case I: generate both the auto function and auto constants strings and write the constants string to file
            func_file, const_file = self._generate_auto_files(func_name=func_name, func_args=func_args,
                                                              state_vars=state_vars, **kwargs)

            # write auto constants to file
            build_dir = f"{self.fdir}/" if self.fdir else ""
            cname = kwargs.pop('auto_constants_file', 'ivp')
            try:
                with open(f'{build_dir}c.{cname}', 'wt') as cfile:
                    cfile.write(const_file)
            except FileNotFoundError:
                with open(f'{build_dir}c.{cname}', 'xt') as cfile:
                    cfile.write(const_file)

        else:

            # case II: generate a standard fortran function string
            func_file = self.generate()

        # compile fortran function and write it to file
        f2py.compile(func_file, modulename=self._fname, extension=self._fend, source_fn=file, verbose=False)

        # import function from temporary file
        exec(f"from {self._fname} import {self._fname}", globals())
        exec(f"rhs_eval = {self._fname}.{func_name}", globals())
        rhs_eval = globals().pop('rhs_eval')

        # apply function decorator
        decorator = kwargs.pop('decorator', None)
        if decorator:
            decorator_kwargs = kwargs.pop('decorator_kwargs', dict())
            rhs_eval = decorator(rhs_eval, **decorator_kwargs)

        return rhs_eval

    def register_vars(self, variables: list):
        for v in variables:
            if v.name not in self._var_declaration_info:
                self._var_declaration_info[v.name] = v

    def clear(self) -> None:
        """Removes all layers, variables and operations from graph. Deletes build directory.
        """

        # delete fortran-specific temporary files
        wdir = self.fdir if self.fdir else os.getcwd()
        for f in [f for f in os.listdir(wdir)]:
            if "cpython" in f and self._fname in f and f[-3:] == ".so":
                os.remove(f"{wdir}/{f}")
            elif f == 'c.ivp' or (f[:5] == 'fort.' and len(f) == 6):
                os.remove(f"{wdir}/{f}")
            elif f == f"{self._fname}.exe" or f == f"{self._fname}.mod" or f == f"{self._fname}.o":
                os.remove(f"{wdir}/{f}")

        # call parent method
        super().clear()

    @staticmethod
    def expr_to_str(expr: str, args: tuple):

        func = 'cshift('
        if func in expr:

            old_shift = f"{args[-1]}"
            new_shift = f"-{old_shift}"
            start = expr.find(func) + len(func)
            stop = expr[start:].find(')')
            old_expr = expr[start:start+stop]
            new_expr = replace(expr[start:start+stop], old_shift, new_shift)
            expr = replace(expr, old_expr, new_expr)

        return expr

    def _generate_auto_files(self, func_name: str, func_args: tuple = (), state_vars: tuple = (),
                             blocked_indices: tuple = (10, 15), **kwargs):

        # wrapper to the right-hand side evaluation function
        ####################################################

        # add function header that Auto-07p expects
        self.add_linebreak()
        self.add_linebreak()
        self.add_code_line("subroutine func(ndim,y,icp,args,ijac,dy,dfdu,dfdp)")
        self.add_linebreak()

        # load the module in which the pyrates function has been defined
        self.add_code_line(f"use {self._fname}")

        # declare auto-related variables
        dtype = self._get_dtype(self._var_declaration_info['y'].dtype)
        self.add_code_line("implicit none")
        self.add_code_line("integer, intent(in) :: ndim, icp(*), ijac")
        self.add_code_line(f"{dtype}, intent(in) :: y(ndim), args(*)")
        self.add_code_line(f"{dtype}, intent(out) :: dy(ndim)")
        self.add_code_line(f"{dtype}, intent(inout) :: dfdu(ndim,ndim), dfdp(ndim,*)")

        # extract parameters from args list
        increment = 1
        params = []
        for i, arg in enumerate(func_args):
            idx = i + increment
            if blocked_indices[0] <= idx <= blocked_indices[1]:
                idx -= increment
                increment += blocked_indices[1] - blocked_indices[0]
                idx += increment
            params.append(f'args({idx})')

        # call the pyrates subroutine
        additional_args = f", {', '.join(params)}" if params else ""
        self.add_linebreak()
        self.add_code_line(f"call {func_name}(args(14), y, dy{additional_args})")
        self.add_linebreak()
        self.add_code_line("end subroutine func")
        self.add_linebreak()

        # routine that sets up an initial value problem
        ###############################################

        # generate subroutine header
        self.add_linebreak()
        self.add_code_line("subroutine stpnt(ndim, y, args, t)")
        self.add_linebreak()
        self.add_code_line("implicit None")
        self.add_code_line("integer, intent(in) :: ndim")
        self.add_code_line(f"{dtype}, intent(inout) :: y(ndim), args(*)")
        self.add_code_line(f"{dtype}, intent(in) :: t")
        self.add_linebreak()

        # define parameter values
        increment = 1
        for i, arg in enumerate(func_args):
            idx = i + increment
            if blocked_indices[0] <= idx <= blocked_indices[1]:
                idx -= increment
                increment += blocked_indices[1] - blocked_indices[0]
                idx += increment
            p = self._var_declaration_info[arg]
            if sum(p.shape) > 1:
                raise ValueError(f'Vector-valued parameter detected ({p.name} with shape {p.shape}), which cannot be '
                                 f'handled by Auto-07p. Please change the definition of your network (e.g. remove '
                                 f'extrinsic inputs) such that no vectorized model parameters exist.')
            self.add_code_line(f"args({idx}) = {p.value}  ! {p.name}")

        # define initial state
        for i, var in enumerate(state_vars):
            v = self._var_declaration_info[var]
            self.add_code_line(f"y({i+1}) = {v.value}  ! {v.name}")

        # end subroutine
        self.add_linebreak()
        self.add_code_line("end subroutine stpnt")
        self.add_linebreak()

        # dummy routines (could be made available for more complex Auto-07p usages)
        ###########################################################################

        self.add_linebreak()
        for routine in ['bcnd', 'icnd', 'fopt', 'pvls']:
            self.add_linebreak()
            self.add_code_line(f"subroutine {routine}")
            self.add_code_line(f"end subroutine {routine}")
            self.add_linebreak()

        func_file = self.generate()
        self.code.clear()

        # create auto constants file
        ############################

        auto_constants = {'NDIM': 1, 'NPAR': 1, 'IPS': -2, 'ILP': 0, 'ICP': [14], 'NTST': 1, 'NCOL': 3, 'IAD': 0,
                          'ISP': 0, 'ISW': 1, 'IPLT': 0, 'NBC': 0, 'NINT': 0, 'NMX': 10000, 'NPR': 1, 'MXBF': 10,
                          'IID': 2, 'ITMX': 2, 'ITNW': 5, 'NWTN': 2, 'JAC': 0, 'EPSL': 1e-6, 'EPSU': 1e-6, 'EPSS': 1e-4,
                          'IRS': 0, 'DS': 1e-4, 'DSMIN': 1e-8, 'DSMAX': 1e-2, 'IADS': 1, 'THL': {}, 'THU': {},
                          'UZR': {}, 'STOP': {}}

        auto_constants['NDIM'] = len(state_vars)
        auto_constants['NPAR'] = len(func_args)
        for key in list(kwargs.keys()):
            if key in auto_constants:
                auto_constants[key] = kwargs.pop(key)

        # write auto constants to string
        for key, val in auto_constants.items():
            self.add_code_line(f"{key} = {val}")
        const_file = self.generate()
        self.code.clear()

        return func_file, const_file

    def _get_func_info(self, name: str, shape: tuple = (), dtype: str = 'float'):

        func_info = self._funcs[name]

        # case I: generate shape-specific fortran function call
        if callable(func_info['call']):

            # extract unique index for input variable shape
            try:
                shapes, indices = self._op_calls[name]
                try:
                    idx = shapes.index(shape)
                    idx = indices[idx]
                except IndexError:
                    idx = indices[-1]
                    shapes.append(shape)
                    indices.append(idx)
            except KeyError:
                idx = 1
                self._op_calls[name] = [shape], [idx]

            # generate function call and string
            func_call, func_str = func_info['call'](idx, self._get_shape(shape, var=''), self._get_dtype(dtype))
            func_info['call'] = func_call
            func_info['def'] = func_str

        return func_info

    def _add_func_call(self, name: str, args: Iterable, return_var: str = 'dy'):

        # add function header
        self.add_code_line(f"subroutine {name}({','.join(args)})")
        self.add_linebreak()
        self.add_code_line("implicit none")
        self.add_linebreak()

        # add variable declarations
        for arg in self._var_declaration_info:
            dtype, intent, shape = self._get_var_declaration_info(arg, args)
            intent = f", intent({intent})" if intent else ""
            self.add_code_line(f"{dtype}{intent} :: {arg}{shape}")

    def _get_var_declaration_info(self, var: str, args: Iterable) -> tuple:

        # extract variable definition
        v = self._var_declaration_info[var]

        # define data type
        dtype = self._get_dtype(v.dtype)

        # define intent of input arguments
        if v.name in args:
            intent = 'in' if v.is_constant or v.name in 'ty' else 'inout'
        else:
            intent = ""

        # define shape
        shape = self._get_shape(v.shape, var)

        return dtype, intent, shape

    def _solve(self, solver: str, func: Callable, args: tuple, T: float, dt: float, dts: float, y0: np.ndarray,
               t0: np.ndarray, times: np.ndarray, **kwargs) -> np.ndarray:

        # extract delta vector
        dy = args[0]

        # define wrapper function for fortran subroutine
        def fort_func(t, y, *args):
            func(t, y, *args)
            return dy

        return super()._solve(solver=solver, func=fort_func, args=args, T=T, dt=dt, dts=dts, y0=y0, t0=t0, times=times,
                              **kwargs)

    def _get_dtype(self, dtype: Union[str, np.dtype]):
        if dtype == 'float':
            dtype = self._float_precision
        if 'float' in dtype:
            dtype = 'double precision' if '64' in dtype else 'real'
        else:
            dtype = 'integer'
        return dtype

    def _process_idx(self, idx: Union[Tuple[int, int], int, str, ComputeVar], **kwargs) -> str:
        if idx == ':':
            return ''
        return super()._process_idx(idx=idx, **kwargs)

    @staticmethod
    def _get_shape(shape: tuple, var: str):
        shape = str(shape) if shape else ''
        if len(shape) < 3:
            shape = '(1)' if (var == 'dy' or var == 'y') else ''
        elif shape[-2] == ',':
            shape = f"{shape[:-2]})"
        return shape

    @staticmethod
    def _find_first_op(code, start, stop):
        if stop < len(code):
            code_tmp = code[start:stop]
            ops = ["+", "-", "*", "/", "**", "^", "%", "<", ">", "==", "!=", "<=", ">="]
            indices = [code_tmp.index(op) for op in ops if op in code_tmp]
            if indices and max(indices) > 0:
                return max(indices) + start
            idx = start
            for break_sign in [',', ')', ' ']:
                if break_sign in code_tmp:
                    idx_tmp = len(code_tmp) - code_tmp[::-1].index(break_sign)
                    if len(code_tmp) - idx_tmp < len(code_tmp) - idx:
                        idx = idx_tmp
            return idx + start
        return stop + start
