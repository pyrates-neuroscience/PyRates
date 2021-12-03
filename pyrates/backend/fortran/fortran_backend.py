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
        """Instantiates tensorflow backend, i.e. a tensorflow graph.
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

    def generate_func_head(self, func_name: str, state_var: str = None, func_args: list = None):

        # finalize list of arguments for the function call
        if func_args:
            self.register_vars(func_args)
            func_args = [arg.name for arg in func_args]
        else:
            func_args = []
        state_vars = ['t', state_var]
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
        self._add_func_call(name=func_name, args=func_args)

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
                    if len(code_tmp)-idx_tmp < len(code_tmp)-idx:
                        idx = idx_tmp
            return idx + start
        return stop + start

    def _get_func_info(self, name: str, shape: tuple = ()):

        func_info = self._funcs[name]

        # case I: generate shape-specific fortran function call
        if type(func_info['call']) is Callable:

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
            func_call, func_str = func_info['call'](idx, shape)
            func_info['call'] = func_call
            func_info['def'] = func_str

        return func_info

    def _add_func_call(self, name: str, args: Iterable):

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

    def generate_func(self, func_name: str, to_file: bool = True, **kwargs):

        if to_file:
            file = f'{self._fdir}/{self._fname}{self._fend}' if self._fdir else f'{self._fname}{self._fend}'
        else:
            file = None

        # compile fortran function
        f2py.compile(self.generate(), modulename=self._fname, extension=self._fend, source_fn=file, verbose=False)

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
        wdir = self._fdir if self._fdir else os.getcwd()
        for f in [f for f in os.listdir(wdir)
                  if "cpython" in f and self._fname in f and f[-3:] == ".so"]:
            os.remove(f"{wdir}/{f}")

        # call parent method
        super().clear()

    def _get_var_declaration_info(self, var: str, args: Iterable) -> tuple:

        # extract variable definition
        v = self._var_declaration_info[var]

        # define data type
        dtype = str(v.dtype)
        if 'float' in dtype:
            dtype = 'double precision' if '64' in dtype else 'real'
        else:
            dtype = 'integer'

        # define intent of input arguments
        if v.name in args:
            intent = 'in' if v.is_constant or v.name in 'ty' else 'inout'
        else:
            intent = ""

        # define shape
        shape = str(v.shape)
        if not shape and (var == 'dy' or var == 'y'):
            shape = '(1)'
        elif shape[-2] == ',':
            shape = f"{shape[:-2]})"
        else:
            shape = ""

        return dtype, intent, shape

    @staticmethod
    def _solve_euler(func: Callable, args: tuple, T: float, dt: float, dts: float, y: np.ndarray):

        # preparations for fixed step-size integration
        idx = 0
        steps = int(np.round(T / dt))
        store_steps = int(np.round(T / dts))
        store_step = int(np.round(dts / dt))
        state_rec = np.zeros((store_steps, y.shape[0]) if y.shape else (store_steps, 1))

        # solve ivp for forward Euler method
        dy = args[0]
        args = args[1:]
        for step in range(steps):
            if step % store_step == 0:
                state_rec[idx, :] = y
                idx += 1
            func(step, y, dy, *args)
            y += dt * dy

        return state_rec

# class PyAutoBackend(FortranBackend):
#
#     _blocked_indices = (10, 32)
#     _time_scale = 1e1
#
#     def __init__(self,
#                  ops: Optional[Dict[str, str]] = None,
#                  dtypes: Optional[Dict[str, object]] = None,
#                  name: str = 'net_0',
#                  float_default_type: str = 'float64',
#                  imports: Optional[List[str]] = None,
#                  build_dir: Optional[str] = None,
#                  **kwargs
#                  ) -> None:
#
#         # validate auto directory
#         try:
#             self._auto_dir = kwargs.pop('auto_dir')
#         except KeyError as e:
#             print("WARNING: No auto installation directory has been passed to the backend. If this directory has "
#                   "not been set as an environment variable, this will cause an error when attempting to perform "
#                   "simulations. The Auto-07p installation directory can be passed to the `run()` method as follows: "
#                   "apply_kwargs={'backend_kwargs': {'auto_dir': '<path>'}}.")
#
#         # set auto default constants
#         self.auto_constants = {'NDIM': 1, 'NPAR': 1,
#                               'IPS': -2, 'ILP': 0, 'ICP': [14], 'NTST': 1, 'NCOL': 3, 'IAD': 0, 'ISP': 0, 'ISW': 1,
#                               'IPLT': 0, 'NBC': 0, 'NINT': 0, 'NMX': 10000, 'NPR': 1, 'MXBF': 10, 'IID': 2, 'ITMX': 2,
#                               'ITNW': 5, 'NWTN': 2, 'JAC': 0, 'EPSL': 1e-6, 'EPSU': 1e-6, 'EPSS': 1e-4, 'IRS': 0,
#                               'DS': 1e-4, 'DSMIN': 1e-8, 'DSMAX': 1e-2, 'IADS': 1, 'THL': {},
#                               'THU': {}, 'UZR': {}, 'STOP': {}}
#
#         super().__init__(ops=ops, dtypes=dtypes, name=name, float_default_type=float_default_type, imports=imports,
#                          build_dir=build_dir, **kwargs)
#
#     def clear(self) -> None:
#         """Removes all layers, variables and operations from graph. Deletes build directory.
#         """
#
#         # call parent method
#         super().clear()
#
#         # delete fortran-specific temporary files
#         wdir = f"{self._orig_dir}/{self._build_dir}" if self._build_dir != self._orig_dir else self._orig_dir
#         fendings = ["so", "exe", "mod", "o"]
#         for f in [f for f in os.listdir(wdir)]:
#             fname = self._file_name
#             fsplit = f.split('.')
#             if fsplit[0] == fname and fsplit[-1] in fendings:
#                 os.remove(f"{wdir}/{f}")
#             elif fsplit[0] == 'fort':
#                 os.remove(f"{wdir}/{f}")
#
#     def _generate_func(self, func_str: str, func_name: str = None, file_name: str = None, build_dir: str = None,
#                        decorator: Any = None, **kwargs):
#
#         # preparations
#         if not file_name:
#             file_name = self._file_name
#         if not build_dir:
#             build_dir = f'{self._orig_dir}/{self._build_dir}' if self._build_dir else self._orig_dir
#
#         # add auto-related subroutines
#         code_gen = FortranGen()
#         code_gen.add_code_line(func_str)
#         code_gen.add_linebreak()
#         code_gen.add_linebreak()
#         code_gen, constants_gen = self._generate_auto_routines(code_gen, file_name, func_name, **kwargs)
#         func_str = code_gen.generate()
#         code_gen.clear()
#
#         # add global variables and module definition above the funcion definitions
#         code_gen = FortranGen()
#         code_gen.add_code_line(f"module {self._file_name}")
#         code_gen.add_linebreak()
#         for line in self._global_variables:
#             code_gen.add_code_line(line)
#         code_gen.add_linebreak()
#         code_gen.add_code_line("contains")
#         code_gen.add_linebreak()
#         code_gen.add_linebreak()
#         code_gen.add_code_line(func_str)
#
#         # write rhs function to file
#         f2py.compile(code_gen.generate(), modulename=file_name, extension=self._file_ending,
#                      source_fn=f'{build_dir}/{file_name}{self._file_ending}', verbose=False)
#
#         # import function from temporary file
#         exec(f"from {file_name} import func", globals())
#         rhs_eval = globals().pop('func')
#
#         # apply function decorator
#         if decorator:
#             rhs_eval = decorator(rhs_eval, **kwargs)
#
#         # write auto constants to file
#         try:
#             with open(f'{build_dir}/c.ivp', 'wt') as cfile:
#                 cfile.write(constants_gen.generate())
#         except FileNotFoundError:
#             with open(f'{build_dir}/c.ivp', 'xt') as cfile:
#                 cfile.write(constants_gen.generate())
#
#         return rhs_eval
#
#     def _generate_auto_routines(self, code_gen: FortranGen, module_name: str, func_name: str, **kwargs) -> tuple:
#
#         # wrapper to the right-hand side evaluation function
#         ####################################################
#
#         # add function header that Auto-07p expects
#         code_gen.add_code_line("subroutine func(ndim,state_vec,icp,args,ijac,state_vec_update,dfdu,dfdp)")
#         code_gen.add_linebreak()
#
#         # load the module in which the pyrates function has been defined
#         code_gen.add_code_line(f"use {module_name}")
#
#         # declare auto-related variables
#         code_gen.add_code_line("implicit none")
#         code_gen.add_code_line("integer, intent(in) :: ndim, icp(*), ijac")
#         code_gen.add_code_line("double precision, intent(in) :: state_vec(ndim), args(*)")
#         code_gen.add_code_line("double precision, intent(out) :: state_vec_update(ndim)")
#         code_gen.add_code_line("double precision, intent(inout) :: dfdu(ndim,ndim), dfdp(ndim,*)")
#
#         # declare variables that need to be extracted from args array
#         for arg in self._func_args.copy():
#             dtype, _, shape = self._get_var_declaration_info(arg)
#             try:
#                 s = [int(s) for s in shape[1:-1].split(',')]
#                 if len(s) == 1:
#                     val = self.graph.eval_node(var)
#                     val_init = f"{dtype}, parameter :: {var}{shape} = (/{','.join(val.tolist())}/)"
#                     self._global_variables.append(val_init)
#                 else:
#                     raise NotImplementedError('The PyAutoBackend cannot generate run functions that include parameters '
#                                               'of dimensionality 2 or higher.')
#
#                 idx = self._func_args.index(arg)
#                 self._func_args.pop(idx)
#             except ValueError:
#                 code_gen.add_code_line(f"{dtype} :: {arg}{shape}")
#         code_gen.add_linebreak()
#
#         # extract variables from args array
#         increment = 1
#         for i, arg in enumerate(self._func_args):
#             idx = i + increment
#             if idx >= self._blocked_indices[0] and idx <= self._blocked_indices[1]:
#                 increment += self._blocked_indices[1] - self._blocked_indices[0]
#                 idx += increment
#             code_gen.add_code_line(f"{arg} = args({idx})")
#
#         # call the pyrates subroutine
#         additional_args = f", {', '.join(self._func_args)}" if self._func_args else ""
#         code_gen.add_code_line(f"call {func_name}(args(14), state_vec, state_vec_update{additional_args})")
#         code_gen.add_linebreak()
#         code_gen.add_code_line("end subroutine func")
#
#         # routine that sets up an initial value problem
#         ###############################################
#
#         # generate subroutine header
#         code_gen.add_linebreak()
#         code_gen.add_code_line("subroutine stpnt(ndim, state_vec, args, t)")
#         code_gen.add_linebreak()
#         code_gen.add_code_line("implicit None")
#         code_gen.add_code_line("integer, intent(in) :: ndim")
#         code_gen.add_code_line("double precision, intent(inout) :: state_vec(ndim), args(*)")
#         code_gen.add_code_line("double precision, intent(in) :: t")
#         code_gen.add_linebreak()
#
#         # define parameter values
#         code_gen.add_linebreak()
#         increment = 1
#         idx = 0
#         for i, arg in enumerate(self._func_args):
#             idx = i + increment
#             if idx >= self._blocked_indices[0] and idx <= self._blocked_indices[1]:
#                 increment += self._blocked_indices[1] - self._blocked_indices[0]
#                 idx += increment
#             val = self.graph.eval_node(arg)
#             code_gen.add_code_line(f"args({idx}) = {val}")
#         npar = idx
#
#         # define initial state
#         code_gen.add_linebreak()
#         for key, info in self._state_vars.items():
#             idx = info['index']
#             v_init = info['init']
#             code_gen.add_code_line(f"state_vec({idx}) = {v_init}")
#         code_gen.add_linebreak()
#
#         # end subroutine
#         code_gen.add_linebreak()
#         code_gen.add_code_line("end subroutine stpnt")
#         code_gen.add_linebreak()
#
#         # dummy routines (could be made available for more complex Auto-07p usages)
#         ###########################################################################
#
#         code_gen.add_linebreak()
#         for routine in ['bcnd', 'icnd', 'fopt', 'pvls']:
#             code_gen.add_linebreak()
#             code_gen.add_code_line(f"subroutine {routine}")
#             code_gen.add_code_line(f"end subroutine {routine}")
#             code_gen.add_linebreak()
#
#         # create auto constants file
#         ############################
#
#         self.auto_constants['NDIM'] = len(self.graph.eval_node('state_vec'))
#         self.auto_constants['NPAR'] = npar
#         for key, val in kwargs:
#             if key in self.auto_constants:
#                 self.auto_constants[key] = kwargs.pop(key)
#
#         # write auto constants to string
#         const_gen = FortranGen()
#         for key, val in self.auto_constants.items():
#             const_gen.add_code_line(f"{key} = {val}")
#
#         return code_gen, const_gen
#
#     def _solve_ivp(self, solver: str, T: float, state_vec: np.ndarray, rhs: np.ndarray, dt: float,
#                    eval_times: np.ndarray, dts: float, rhs_func: Callable, *args, **kwargs) -> np.ndarray:
#         """
#
#         Parameters
#         ----------
#         rhs_func
#         func_args
#         state_vars
#         T
#         dt
#         dts
#         t
#         solver
#         output_indices
#         kwargs
#
#         Returns
#         -------
#
#         """
#
#         from pyrates.utility.pyauto import PyAuto
#         from scipy.interpolate import interp1d
#
#         # preparations
#         pyauto = PyAuto(working_dir=self._build_dir, auto_dir=self._auto_dir)
#         ds = dt * self._time_scale
#         dsmin = ds*1e-2
#         auto_defs = {'DSMIN': dsmin, 'DSMAX': ds*1e1, 'NMX': int(T/dsmin)}
#         for key, val in auto_defs.items():
#             if key not in kwargs:
#                 kwargs[key] = val
#         ndim = len(state_vec)
#
#         # solve ivp
#         kwargs_tmp = {key: val for key, val in kwargs.items() if key in self.auto_constants}
#         pyauto.run(e=self._file_name, c='ivp', DS=ds, name='t', UZR={14: T}, STOP={'UZ1'},
#                    **kwargs_tmp)
#
#         # extract results
#         extract = [f'U({i+1})' for i in range(ndim)]
#         extract.append('PAR(14)')
#         results_tmp = pyauto.extract(keys=extract, cont='t')
#         times = results_tmp.pop('PAR(14)')
#         results = []
#         for i in range(ndim):
#             y_inter = interp1d(times, np.squeeze(results_tmp.pop(f'U({i+1})')))
#             results.append(y_inter(eval_times))
#
#         return np.asarray(results).T
#
#     def _idx_to_str(self, idx, var: str):
#         index = f"{self._idx[0]}{idx}{self._idx[1]}"
#         value = self.graph.eval_node(var)
#         self._state_vars[var] = {'index': index, 'init': value}
#         return index
