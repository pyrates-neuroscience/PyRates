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

"""Contains wrapper classes for different backends that are needed by the parser module.

A new backend needs to implement the following methods:
- __init__
- run
- add_var
- add_op
- add_layer

Currently supported backends:
- Numpy: BaseBackend.
- Tensorflow: TensorflowBackend.
- Fortran: FortranBackend (experimental).

"""

# pyrates internal _imports
from ..computegraph import ComputeVar
from .base_funcs import base_funcs
from .. import PyRatesException

# external _imports
from typing import Optional, Dict, List, Union, Tuple, Callable, Iterable
import numpy as np
import os, sys
from shutil import rmtree


# Helper Functions and Classes
##############################


class CodeGen:
    """Generates python code. Can add code lines, line-breaks, indents and remove indents.
    """

    def __init__(self):
        self.code = []
        self.lvl = 0

    def generate(self):
        """Generates a single code string from its history of code additions.
        """
        return '\n'.join(self.code)

    def add_code_line(self, code_str):
        """Add code line string to code.
        """
        code_str = code_str.split('\n')
        for code in code_str:
            self.code.append("\t" * self.lvl + code)

    def add_linebreak(self):
        """Add a line-break to the code.
        """
        self.code.append("")

    def add_indent(self):
        """Add an indent to the code.
        """
        self.lvl += 1

    def remove_indent(self):
        """Remove an indent to the code.
        """
        if self.lvl == 0:
            raise(SyntaxError("Error in generation of network function file: A net negative indentation was requested.")
                  )
        self.lvl -= 1

    def clear(self):
        """Deletes all code lines from the memory of the generator.
        """
        self.code.clear()


#######################################
# classes for backend functionalities #
#######################################


class BaseBackend(CodeGen):
    """Default backend class. Transforms all network equations into their numpy equivalents. Based on a Python code
    generator.
    """

    def __init__(self,
                 ops: Optional[Dict[str, str]] = None,
                 imports: Optional[List[str]] = None,
                 **kwargs
                 ) -> None:
        """Instantiates the standard, numpy-based backend.
        """

        # call to super method (initializes code generator)
        super().__init__()

        # definition of usable math operations
        self._funcs = base_funcs.copy()
        if ops:
            self._funcs.update(ops)
        self._helper_funcs = []

        # definition of extrinsic function _imports
        self._imports = ["from numpy import pi, sqrt"]
        if imports:
            for imp in imports:
                self.add_import(imp)

        # public attributes
        self.add_hist_arg = kwargs.pop('add_hist_arg', True)
        self.lags = {}
        self.idx_dummy_var = "temporary_pyrates_var_index"

        # private attributes
        self._float_precision = kwargs.pop('float_precision', 'float32')
        self._int_precision = kwargs.pop('int_precision', 'int32')
        self._idx_left = kwargs.pop('idx_left', '[')
        self._idx_right = kwargs.pop('idx_right', ']')
        self._start_idx = kwargs.pop('start_idx', 0)
        self._no_funcs = ["identity", "index_1d", "index_2d", "index_range", "index_axis"]

        # file-creation-related attributes
        fdir, *fname = self.get_fname(kwargs.pop('file_name', 'pyrates_run'))
        if fdir:
            sys.path.append(fdir)
        sys.path.append(os.getcwd())
        self.fdir = fdir
        self._fname = fname[0]
        self._fend = f".{fname[1]}" if len(fname) > 1 else kwargs.pop('file_ending', '.py')

    def get_var(self, v: ComputeVar):
        if v.is_float or v.is_complex:
            dtype = self._float_precision
            if 'complex' in dtype and v.name in ['t', 'time']:
                dtype = f'float{dtype[7:]}'
        else:
            dtype = self._int_precision
        return np.asarray(v.value, dtype=dtype)

    def get_op(self, name: str, **kwargs) -> dict:

        # retrieve function information from backend definitions
        func_info = self._get_func_info(name, **kwargs)
        func_name = func_info['call']

        # add extrinsic function imports if necessary
        if 'imports' in func_info:
            for imp in func_info['imports']:
                *in_path, in_func = imp.split('.')
                self.add_import(f"from {'.'.join(in_path)} import {in_func}")

        if 'def' in func_info:

            # extract the provided function definition
            func_str = func_info['def']

            # remember the function definition string for file creation
            if func_str not in self._helper_funcs and func_name not in self._no_funcs:
                self._helper_funcs.append(func_str)

        if 'func' in func_info:

            # extract the provided callable
            func = func_info['func']

        else:

            # extract the provided function definition
            func_str = func_info['def']

            # make _imports available to function
            for imp in self._imports:
                try:
                    exec(imp, globals())
                except SyntaxError:
                    pass

            # evaluate the function string to receive a callable
            exec(func_str, globals())
            func = globals().pop(func_name)

        return {'func': func, 'call': func_name}

    def add_var_update(self, lhs: ComputeVar, rhs: str, lhs_idx: Optional[str] = None, rhs_shape: Optional[tuple] = ()):

        lhs = lhs.name
        if lhs_idx:
            idx, _ = self.create_index_str(lhs_idx, apply=True)
            lhs = f"{lhs}{idx}"
        self.add_code_line(f"{lhs} = {rhs}")

    def add_var_hist(self, lhs: str, delay: Union[ComputeVar, float], state_idx: str, **kwargs):
        idx = self._process_idx(state_idx)
        d = self._process_delay(delay)
        self.add_code_line(f"{lhs} = hist(t-{d})[{idx}]")

    def add_import(self, line: str):
        if line not in self._imports:
            self._imports.append(line)

    def create_index_str(self, idx: Union[str, int, tuple], separator: str = ',', apply: bool = True,
                         **kwargs) -> Tuple[str, dict]:

        # preprocess idx
        if type(idx) is str and separator in idx:
            idx = tuple(idx.split(separator))

        # case: multiple indices
        if type(idx) is tuple:
            idx = list(idx)
            for i in range(len(idx)):
                idx[i] = self._process_idx(idx[i], **kwargs)
            idx = tuple([f"{i}" for i in idx])
            idx_str = f"{self._idx_left}{separator.join(idx)}{self._idx_right}" if apply else separator.join(idx)
            return idx_str, dict()

        # case: single index
        idx = self._process_idx(idx, **kwargs)
        return f"{self._idx_left}{idx}{self._idx_right}" if apply else idx, dict()

    def get_fname(self, f: str) -> tuple:

        f_split = f.split('.')
        if len(f_split) > 2:
            raise ValueError(f'File name {f} has wrong format. Only one `.` can be used to separate file name from '
                             f'file ending.')
        if len(f_split) == 2:
            *path, file = f_split[0].split('/')
            return '/'.join(path), file, f_split[1]
        else:
            *path, file = f.split('/')
            return '/'.join(path), file

    def generate_func_head(self, func_name: str, state_var: str = 'y', return_var: str = 'dy', func_args: list = None,
                           add_hist_func: bool = False):

        imports = self._imports
        helper_funcs = self._helper_funcs
        if func_args:
            func_args = [arg.name for arg in func_args]
        else:
            func_args = []
        state_vars = ['t', state_var]
        if add_hist_func:
            state_vars.append('hist')
        _, indices = np.unique(func_args, return_index=True)
        func_args = state_vars + [func_args[idx] for idx in np.sort(indices)]

        if imports:

            # add _imports at beginning of file
            for imp in imports:
                self.add_code_line(imp)
            self.add_linebreak()

        if helper_funcs:

            # add definitions of helper functions after the _imports
            for func in helper_funcs:
                self.add_code_line(func)
            self.add_linebreak()

        # add function header
        self.add_linebreak()
        self._add_func_call(name=func_name, args=func_args, return_var=return_var)
        self.add_indent()

        return func_args

    def generate_func_tail(self, rhs_var: str = 'dy'):

        self.add_code_line(f"return {rhs_var}")
        self.remove_indent()

    def generate_func(self, func_name: str, to_file: bool = True, **kwargs):

        # generate the current function string via the code generator
        func_str = self.generate()

        if to_file:

            # save rhs function to file
            file = f'{self.fdir}/{self._fname}' if self.fdir else self._fname
            with open(f'{file}{self._fend}', 'w') as f:
                f.writelines(func_str)
                f.close()

            # import function from file
            exec(f"from {self._fname} import {func_name}", globals())

        else:

            # just execute the function string, without writing it to file
            exec(func_str, globals())

        rhs_eval = globals().pop(func_name)

        # apply function decorator
        decorator = kwargs.pop('decorator', None)
        if decorator:
            decorator_kwargs = kwargs.pop('decorator_kwargs', dict())
            rhs_eval = decorator(rhs_eval, **decorator_kwargs)

        return rhs_eval

    def run(self, func: Callable, func_args: tuple, T: float, dt: float, dts: float, outputs: dict,
            solver: str, **kwargs) -> dict:

        # initial values
        t0 = func_args[0]
        times = np.arange(0.0, T, dts) if dts else np.arange(0.0, T, dt)
        y0 = func_args[1]

        # perform simulation
        results = self._solve(solver=solver, func=func, args=func_args[2:], T=T, dt=dt, dts=dts, y0=y0, t0=t0,
                              times=times, **kwargs)

        # reduce state recordings to requested state variables
        for key, idx in outputs.items():
            if type(idx) is tuple and idx[1] - idx[0] == 1:
                idx = (idx[0],)
            elif type(idx) is int:
                idx = (idx,)
            outputs[key] = results[:, idx] if len(idx) == 1 else results[:, idx[0]:idx[1]]
        outputs['time'] = times

        return outputs

    def clear(self):

        # clear code generator
        super().clear()

        # remove files and directories that have been created during simulation process
        if self.fdir:
            rmtree(self.fdir)
        else:
            try:
                os.remove(f"{self._fname}{self._fend}")
            except FileNotFoundError:
                pass

        # delete loaded modules from the system
        if self._fname in sys.modules:
            del sys.modules[self._fname]

    @staticmethod
    def to_file(fn: str, **kwargs):
        np.savez(fn, **kwargs)

    @staticmethod
    def register_vars(variables: list):
        pass

    @staticmethod
    def finalize_idx_str(var: ComputeVar, idx: str):
        return f"{var.name}{idx}"

    @staticmethod
    def expr_to_str(expr: str, args: tuple):
        return expr

    @staticmethod
    def get_hist_func(y: np.ndarray):
        return lambda t: y

    def _get_func_info(self, name: str, **kwargs):
        return self._funcs[name]

    def _process_idx(self, idx: Union[Tuple[int, int], int, str, ComputeVar], **kwargs) -> str:
        if type(idx) is ComputeVar:
            idx.set_value(idx.value + self._start_idx)
            return idx.name
        if type(idx) is tuple:
            return f"{idx[0] + self._start_idx}:{idx[1]}"
        if type(idx) is int:
            return f"{idx + self._start_idx}"
        try:
            return self._process_idx(int(idx), **kwargs)
        except (TypeError, ValueError):
            return idx

    def _process_delay(self, delay: Union[ComputeVar, float]) -> str:
        return f"{delay}[{self._start_idx}]" if type(delay) is ComputeVar and delay.shape else f"{delay}"

    def _solve(self, solver: str, func: Callable, args: tuple, T: float, dt: float, dts: float, y0: np.ndarray,
               t0: np.ndarray, times: np.ndarray, **kwargs) -> np.ndarray:

        # perform integration via scipy solver (mostly Runge-Kutta methods)
        if solver == 'euler':

            # solve ivp via forward euler method (fixed integration step-size)
            results = self._solve_euler(func, args, T, dt, dts, y0, t0)

        elif solver == 'heun':

            # solve ivp via forward Heun's method (fixed integration step-size)
            results = self._solve_heun(func, args, T, dt, dts, y0, t0)

        elif solver == 'scipy':

            results = self._solve_scipy(func, args, T, dt, y0, t0, times, **kwargs)

        else:

            raise PyRatesException('Invalid option for keyword `solver`. Please check the documentation of the '
                                   '`CircuitTemplate.run` method for valid options.')

        return results

    def _add_func_call(self, name: str, args: Iterable, return_var: str = 'dy'):
        self.add_code_line(f"def {name}({','.join(args)}):")

    @staticmethod
    def _solve_euler(func: Callable, args: tuple, T: float, dt: float, dts: float, y: np.ndarray, t0: np.ndarray):

        # preparations for fixed step-size integration
        idx = 0
        steps = int(np.round(T / dt))
        store_steps = int(np.round(T / dts))
        store_step = int(np.round(dts / dt))
        state_rec = np.zeros((store_steps, y.shape[0]) if y.shape else (store_steps, 1), dtype=y.dtype)

        # solve ivp for forward Euler method
        for step in range(t0, steps+t0):
            if step % store_step == t0:
                state_rec[idx, :] = y
                idx += 1
            rhs = func(step, y, *args)
            y += dt * rhs

        return state_rec

    @staticmethod
    def _solve_heun(func: Callable, args: tuple, T: float, dt: float, dts: float, y: np.ndarray, t0: np.ndarray):

        # preparations for fixed step-size integration
        idx = 0
        steps = int(np.round(T / dt))
        store_steps = int(np.round(T / dts))
        store_step = int(np.round(dts / dt))
        state_rec = np.zeros((store_steps, y.shape[0]) if y.shape else (store_steps, 1), dtype=y.dtype)

        # solve ivp for forward Euler method
        for step in range(t0, steps + t0):
            if step % store_step == t0:
                state_rec[idx, :] = y
                idx += 1
            rhs = func(step, y, *args)
            y_0 = y + dt * rhs
            y += dt/2 * (rhs + func(step, y_0, *args))

        return state_rec

    @staticmethod
    def _solve_scipy(func: Callable, args: tuple, T: float, dt: float, y: np.ndarray, t0: np.ndarray, times: np.ndarray,
                     **kwargs):

        # solve ivp via scipy methods (solvers of various orders with adaptive step-size)
        from scipy.integrate import solve_ivp
        kwargs['t_eval'] = times

        # call scipy solver
        results = solve_ivp(fun=func, t_span=(t0, T), y0=y, first_step=dt, args=args, **kwargs)
        return results['y'].T
