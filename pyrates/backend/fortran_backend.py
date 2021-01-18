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

"""Wraps tensorflow such that it's low-level functions can be used by PyRates to create and simulate a compute graph.
"""

# external imports
from typing import Optional, Dict, Callable, List, Any, Union
import os
import sys
from shutil import rmtree
import numpy as np
from numpy import f2py

# pyrates internal imports
from .numpy_backend import NumpyBackend, PyRatesAssignOp, PyRatesIndexOp, PyRatesOp, CodeGen, extract_lhs_var

# meta infos
__author__ = "Richard Gast"
__status__ = "development"

module_counter = 0


class FortranOp(PyRatesOp):

    def __init__(self, op: str, short_name: str, name, *args, **kwargs) -> None:
        self.build_dir = kwargs.pop('build_dir', '')
        super().__init__(op, short_name, name, *args, **kwargs)

    def _generate_func(self):
        return generate_func(self)

    @classmethod
    def _process_args(cls, args, results, constants_to_num=False):
        return super()._process_args(args, results, constants_to_num=constants_to_num)


class FortranIndexOp(PyRatesIndexOp):

    def __init__(self, op: str, short_name: str, name, *args, **kwargs) -> None:
        self.build_dir = kwargs.pop('build_dir', '')
        super().__init__(op, short_name, name, *args, **kwargs)

    def _generate_func(self):
        return generate_func(self)


class FortranAssignOp(PyRatesAssignOp):

    def __init__(self, op: str, short_name: str, name, *args, **kwargs) -> None:
        self.build_dir = kwargs.pop('build_dir', '')
        super().__init__(op, short_name, name, *args, **kwargs)

    def _generate_func(self, return_key='y_delta'):
        try:
            idx = self._op_dict['arg_names'].index(return_key)
        except ValueError:
            return_key = extract_lhs_var(self._op_dict['value'])
            idx = self._op_dict['arg_names'].index(return_key)
        ndim = self._op_dict['args'][idx].shape
        return generate_func(self, return_key=return_key, omit_assign=True, return_dim=ndim, return_intent='out')

    def numpy(self):
        result = self._callable(*self.args[1:])
        self._check_numerics(result, self.name)
        return result


class FortranBackend(NumpyBackend):

    idx_l, idx_r = "(", ")"
    idx_start = 1

    def __init__(self,
                 ops: Optional[Dict[str, str]] = None,
                 dtypes: Optional[Dict[str, object]] = None,
                 name: str = 'net_0',
                 float_default_type: str = 'float32',
                 imports: Optional[List[str]] = None,
                 build_dir: Optional[str] = None,
                 auto_compat: bool = False
                 ) -> None:
        """Instantiates numpy backend, i.e. a compute graph with numpy operations.
        """

        # define operations and datatypes of the backend
        ################################################

        # base math operations
        ops_f = {"+": {'name': "fortran_add", 'call': "+"},
                 "-": {'name': "fortran_subtract", 'call': "-"},
                 "*": {'name': "fortran_multiply", 'call': "*"},
                 "/": {'name': "fortran_divide", 'call': "/"},
                 "%": {'name': "fortran_modulo", 'call': "MODULO"},
                 "^": {'name': "fortran_power", 'call': "**"},
                 "**": {'name': "fortran_power_float", 'call': "**"},
                 "@": {'name': "fortrandot", 'call': ""},
                 ".T": {'name': "fortrantranspose", 'call': ""},
                 ".I": {'name': "fortraninvert", 'call': ""},
                 ">": {'name': "fortrangreater", 'call': ">"},
                 "<": {'name': "fortranless", 'call': "<"},
                 "==": {'name': "fortranequal", 'call': "=="},
                 "!=": {'name': "fortran_not_equal", 'call': "!="},
                 ">=": {'name': "fortran_greater_equal", 'call': ">="},
                 "<=": {'name': "fortran_less_equal", 'call': "<="},
                 "=": {'name': "assign", 'call': "="},
                 "+=": {'name': "assign_add", 'call': ""},
                 "-=": {'name': "assign_subtract", 'call': ""},
                 "*=": {'name': "assign_multiply", 'call': ""},
                 "/=": {'name': "assign_divide", 'call': ""},
                 "neg": {'name': "negative", 'call': "-"},
                 "sin": {'name': "fortran_sin", 'call': "SIN"},
                 "cos": {'name': "fortran_cos", 'call': "COS"},
                 "tan": {'name': "fortran_tan", 'call': "TAN"},
                 "atan": {'name': "fortran_atan", 'call': "ATAN"},
                 "abs": {'name': "fortran_abs", 'call': "ABS"},
                 "sqrt": {'name': "fortran_sqrt", 'call': "SQRT"},
                 "sq": {'name': "fortran_square", 'call': ""},
                 "exp": {'name': "fortran_exp", 'call': "EXP"},
                 "max": {'name': "fortran_max", 'call': "MAX"},
                 "min": {'name': "fortran_min", 'call': "MIN"},
                 "argmax": {'name': "fortran_transpose", 'call': "ARGMAX"},
                 "argmin": {'name': "fortran_argmin", 'call': "ARGMIN"},
                 "round": {'name': "fortran_round", 'call': ""},
                 "sum": {'name': "fortran_sum", 'call': "SUM"},
                 "mean": {'name': "fortran_mean", 'call': ""},
                 "concat": {'name': "fortran_concatenate", 'call': ""},
                 "reshape": {'name': "fortran_reshape", 'call': ""},
                 "append": {'name': "fortran_append", 'call': ""},
                 "shape": {'name': "fortran_shape", 'call': ""},
                 "dtype": {'name': "fortran_dtype", 'call': ""},
                 'squeeze': {'name': "fortran_squeeze", 'call': ""},
                 'expand': {'name': 'fortran_expand', 'call': ""},
                 "roll": {'name': "fortran_roll", 'call': ""},
                 "cast": {'name': "fortran_cast", 'call': ""},
                 "randn": {'name': "fortran_randn", 'call': ""},
                 "ones": {'name': "fortran_ones", 'call': ""},
                 "zeros": {'name': "fortran_zeros", 'call': ""},
                 "range": {'name': "fortran_arange", 'call': ""},
                 "softmax": {'name': "pyrates_softmax", 'call': ""},
                 "sigmoid": {'name': "pyrates_sigmoid", 'call': ""},
                 "tanh": {'name': "fortran_tanh", 'call': "TANH"},
                 "index": {'name': "pyrates_index", 'call': "pyrates_index"},
                 "mask": {'name': "pyrates_mask", 'call': ""},
                 "group": {'name': "pyrates_group", 'call': ""},
                 "asarray": {'name': "fortran_asarray", 'call': ""},
                 "no_op": {'name': "pyrates_identity", 'call': "no_op"},
                 "interpolate": {'name': "pyrates_interpolate", 'call': ""},
                 "interpolate_1d": {'name': "pyrates_interpolate_1d", 'call': ""},
                 "interpolate_nd": {'name': "pyrates_interpolate_nd", 'call': ""},
                 }
        if ops:
            ops_f.update(ops)
        self.pyauto_compat = auto_compat
        super().__init__(ops=ops_f, dtypes=dtypes, name=name, float_default_type=float_default_type,
                         imports=imports, build_dir=build_dir)
        self._imports = []
        self.npar = 0
        self.ndim = 0
        self._auto_files_generated = False

    def compile(self, build_dir: Optional[str] = None, decorator: Optional[Callable] = None, **kwargs) -> tuple:
        """Compile the graph layers/operations. Creates python files containing the functions in each layer.

        Parameters
        ----------
        build_dir
            Directory in which to create the file structure for the simulation.
        decorator
            Decorator function that should be applied to the right-hand side evaluation function.
        kwargs
            decorator keyword arguments

        Returns
        -------
        tuple
            Contains tuples of layer run functions and their respective arguments.

        """

        # preparations
        ##############

        # remove empty layers and operators
        new_layer_idx = 0
        for layer_idx, layer in enumerate(self.layers.copy()):
            for op in layer.copy():
                if op is None:
                    layer.pop(layer.index(op))
            if len(layer) == 0:
                self.layers.pop(new_layer_idx)
            else:
                new_layer_idx += 1

        # remove previously imported rhs_funcs from system
        if 'rhs_func' in sys.modules:
            del sys.modules['rhs_func']

        # collect state variable and parameter vectors
        state_vars, params, var_map = self._process_vars()

        # update state variable getter operations to include the full state variable vector as first argument
        y = self.get_var('y')
        for key, (vtype, idx) in var_map.items():
            var = self.get_var(key)
            if vtype == 'state_var' and var.short_name != 'y':
                var.args[0] = y
                var.build_op(var.args)

        # create rhs evaluation function
        ################################

        # set up file header
        func_gen = FortranGen()
        for import_line in self._imports:
            func_gen.add_code_line(import_line)
            func_gen.add_linebreak()
        func_gen.add_linebreak()

        # define function head
        func_gen.add_indent()
        if self.pyauto_compat:
            func_gen.add_code_line("subroutine func(ndim,y,icp,args,ijac,y_delta,dfdu,dfdp)")
            func_gen.add_linebreak()
            func_gen.add_code_line("implicit none")
            func_gen.add_linebreak()
            func_gen.add_code_line("integer, intent(in) :: ndim, icp(*), ijac")
            func_gen.add_linebreak()
            func_gen.add_code_line("double precision, intent(in) :: y(ndim), args(*)")
            func_gen.add_linebreak()
            func_gen.add_code_line("double precision, intent(out) :: y_delta(ndim)")
            func_gen.add_linebreak()
            func_gen.add_code_line("double precision, intent(inout) :: dfdu(ndim,ndim), dfdp(ndim,*)")
            func_gen.add_linebreak()
        else:
            func_gen.add_code_line("subroutine func(ndim,t,y,args,y_delta)")
            func_gen.add_linebreak()
            func_gen.add_code_line("implicit none")
            func_gen.add_linebreak()
            func_gen.add_code_line("integer, intent(in) :: ndim")
            func_gen.add_linebreak()
            func_gen.add_code_line("double precision, intent(in) :: t, y(ndim)")
            func_gen.add_linebreak()
            func_gen.add_code_line("double precision, intent(in) :: args(*)")
            func_gen.add_linebreak()
            func_gen.add_code_line("double precision, intent(out) :: y_delta(ndim)")
            func_gen.add_linebreak()

        # declare variable types
        func_gen.add_code_line("double precision ")
        for key in var_map:
            var = self.get_var(key)
            if "float" in str(var.dtype) and var.short_name != 'y_delta':
                func_gen.add_code_line(f"{var.short_name},")
        if "," in func_gen.code[-1]:
            func_gen.code[-1] = func_gen.code[-1][:-1]
        else:
            func_gen.code.pop(-1)
        func_gen.add_linebreak()
        func_gen.add_code_line("integer ")
        for key in var_map:
            var = self.get_var(key)
            if "int" in str(var.dtype):
                func_gen.add_code_line(f"{var.short_name},")
        if "," in func_gen.code[-1]:
            func_gen.code[-1] = func_gen.code[-1][:-1]
        else:
            func_gen.code.pop(-1)
        func_gen.add_linebreak()

        # declare constants
        args = [None for _ in range(len(params))]
        func_gen.add_code_line("! declare constants")
        func_gen.add_linebreak()
        updates, indices = [], []
        i = 0
        for key, (vtype, idx) in var_map.items():
            if vtype == 'constant':
                var = params[idx-self.idx_start][1]
                if var.short_name != 'y_delta':
                    func_gen.add_code_line(f"{var.short_name} = args({idx})")
                    func_gen.add_linebreak()
                    args[idx-1] = var
                    updates.append(f"{var.short_name}")
                    indices.append(i)
                    i += 1
        func_gen.add_linebreak()

        # extract state variables from input vector y
        func_gen.add_code_line("! extract state variables from input vector")
        func_gen.add_linebreak()
        for key, (vtype, idx) in var_map.items():
            var = self.get_var(key)
            if vtype == 'state_var':
                func_gen.add_code_line(f"{var.short_name} = {var.value}")
                func_gen.add_linebreak()
        func_gen.add_linebreak()

        # add equations
        func_gen.add_code_line("! calculate right-hand side update of equation system")
        func_gen.add_linebreak()
        arg_updates = []
        for i, layer in enumerate(self.layers):
            for j, op in enumerate(layer):
                lhs = op.value.split("=")[0]
                lhs = lhs.replace(" ", "")
                find_arg = [arg == lhs for arg in updates]
                if any(find_arg):
                    idx = find_arg.index(True)
                    arg_updates.append((updates[idx], indices[idx]))
                func_gen.add_code_line(op.value)
                func_gen.add_linebreak()
        func_gen.add_linebreak()

        # update parameters where necessary
        # func_gen.add_code_line("! update system parameters")
        # func_gen.add_linebreak()
        # for upd, idx in arg_updates:
        #     update_str = f"args{self.idx_l}{idx}{self.idx_r} = {upd}"
        #     if f"    {update_str}" not in func_gen.code:
        #         func_gen.add_code_line(update_str)
        #         func_gen.add_linebreak()

        # end function
        func_gen.add_code_line(f"end subroutine func")
        func_gen.add_linebreak()
        func_gen.remove_indent()

        # save rhs function to file
        fname = f'{self._build_dir}/rhs_func'
        f2py.compile(func_gen.generate(), modulename='rhs_func', extension='.f', source_fn=f'{fname}.f', verbose=False)

        # create additional subroutines in pyauto compatibility mode
        gen_def = kwargs.pop('generate_auto_def', True)
        if self.pyauto_compat and gen_def:
            self.generate_auto_def(self._build_dir)

        # import function from file
        fname_import = 'func' if self.pyauto_compat else 'rhs_eval'
        exec(f"from rhs_func import {fname_import}", globals())
        rhs_eval = globals().pop('func')

        # apply function decorator
        if decorator:
            rhs_eval = decorator(rhs_eval, **kwargs)

        return rhs_eval, args, state_vars, var_map

    def generate_auto_def(self, directory):
        """

        Parameters
        ----------
        directory

        Returns
        -------

        """

        if not self.pyauto_compat:
            raise ValueError('This method can only be called in pyauto compatible mode. Please set `pyauto_compat` to '
                             'True upon calling the `CircuitIR.compile` method.')

        # read file
        ###########

        try:

            # read file from excisting system compilation
            if directory is None:
                directory = self._build_dir
            fn = f"{directory}/rhs_func.f" if "rhs_func.f" not in directory else directory
            with open(fn, 'rt') as f:
                func_str = f.read()

        except FileNotFoundError:

            # compile system and then read the build files
            self.compile(build_dir=directory, generate_auto_def=False)
            fn = f"{self._build_dir}/rhs_func.f"
            with open(fn, 'r') as f:
                func_str = f.read()

        end_str = "end subroutine func\n"
        idx = func_str.index(end_str)
        func_str = func_str[:idx+len(end_str)]

        # generate additional subroutines
        #################################

        func_gen = FortranGen()

        # generate subroutine header
        func_gen.add_linebreak()
        func_gen.add_indent()
        func_gen.add_code_line("subroutine stpnt(ndim, y, args, t)")
        func_gen.add_linebreak()
        func_gen.add_code_line("implicit None")
        func_gen.add_linebreak()
        func_gen.add_code_line("integer, intent(in) :: ndim")
        func_gen.add_linebreak()
        func_gen.add_code_line("double precision, intent(inout) :: y(ndim), args(*)")
        func_gen.add_linebreak()
        func_gen.add_code_line("double precision, intent(in) :: T")
        func_gen.add_linebreak()

        # declare variable types
        func_gen.add_code_line("double precision ")
        for key in self.vars:
            var = self.get_var(key)
            name = var.short_name
            if "float" in str(var.dtype) and name != 'y_delta' and name != 'y' and name != 't':
                func_gen.add_code_line(f"{var.short_name},")
        if "," in func_gen.code[-1]:
            func_gen.code[-1] = func_gen.code[-1][:-1]
        else:
            func_gen.code.pop(-1)
        func_gen.add_linebreak()
        func_gen.add_code_line("integer ")
        for key in self.vars:
            var = self.get_var(key)
            if "int" in str(var.dtype):
                func_gen.add_code_line(f"{var.short_name},")
        if "," in func_gen.code[-1]:
            func_gen.code[-1] = func_gen.code[-1][:-1]
        else:
            func_gen.code.pop(-1)
        func_gen.add_linebreak()

        _, params, var_map = self._process_vars()

        # define parameter values
        func_gen.add_linebreak()
        for key, (vtype, idx) in var_map.items():
            if vtype == 'constant':
                var = params[idx-self.idx_start][1]
                if var.short_name != 'y_delta' and var.short_name != 'y':
                    func_gen.add_code_line(f"{var.short_name} = {var}")
                    func_gen.add_linebreak()
        func_gen.add_linebreak()

        # define initial state
        func_gen.add_linebreak()
        npar = 0
        for key, (vtype, idx) in var_map.items():
            if vtype == 'constant':
                var = params[idx-self.idx_start][1]
                if var.short_name != 'y_delta' and var.short_name != 'y':
                    func_gen.add_code_line(f"args({idx}) = {var.short_name}")
                    func_gen.add_linebreak()
                    if idx > npar:
                        npar = idx
        func_gen.add_linebreak()

        func_gen.add_linebreak()
        for key, (vtype, idx) in var_map.items():
            var = self.get_var(key)
            if vtype == 'state_var':
                func_gen.add_code_line(f"{var.value} = {var.numpy()}")
                func_gen.add_linebreak()
        func_gen.add_linebreak()

        # end subroutine
        func_gen.add_linebreak()
        func_gen.add_code_line("end subroutine stpnt")
        func_gen.add_linebreak()

        # add dummy subroutines
        for routine in ['bcnd', 'icnd', 'fopt', 'pvls']:
            func_gen.add_linebreak()
            func_gen.add_code_line(f"subroutine {routine}")
            func_gen.add_linebreak()
            func_gen.add_code_line(f"end subroutine {routine}")
            func_gen.add_linebreak()
        func_gen.add_linebreak()
        func_gen.remove_indent()

        func_combined = f"{func_str} \n {func_gen.generate()}"
        f2py.compile(func_combined, source_fn=fn, modulename='rhs_func', extension='.f', verbose=False)

        self.npar = npar
        self.ndim = self.get_var('y').shape[0]

        # generate constants file
        #########################

        # declare auto constants and their values
        auto_constants = {'NDIM': self.ndim, 'NPAR': self.npar, 'IPS': -2, 'ILP': 0, 'ICP': [14], 'NTST': 1, 'NCOL': 4,
                          'IAD': 3, 'ISP': 0, 'ISW': 1, 'IPLT': 0, 'NBC': 0, 'NINT': 0, 'NMX': 10000, 'NPR': 100,
                          'MXBF': 10, 'IID': 2, 'ITMX': 8, 'ITNW': 5, 'NWTN': 3, 'JAC': 0, 'EPSL': 1e-7, 'EPSU': 1e-7,
                          'EPSS': 1e-5, 'IRS': 0, 'DS': 1e-4, 'DSMIN': 1e-8, 'DSMAX': 1e-2, 'IADS': 1, 'THL': {},
                          'THU': {}, 'UZR': {}, 'STOP': {}}

        # write auto constants to string
        cgen = FortranGen()
        for key, val in auto_constants.items():
            cgen.add_code_line(f"{key} = {val}")
            cgen.add_linebreak()

        # write auto constants to file
        try:
            with open(f'{directory}/c.ivp', 'wt') as cfile:
                cfile.write(cgen.generate())
        except FileNotFoundError:
            with open(f'{directory}/c.ivp', 'xt') as cfile:
                cfile.write(cgen.generate())

        self._auto_files_generated = True

        return fn

    def to_pyauto(self, directory=None, generate_auto_def=True, **kwargs):
        """

        Parameters
        ----------
        directory
        generate_auto_def

        Returns
        -------

        """
        from pyrates.utility.pyauto import PyAuto
        directory = directory if directory else self._build_dir
        if generate_auto_def:
            self.generate_auto_def(directory)
        return PyAuto(directory, **kwargs)

    def clear(self) -> None:
        """Removes all layers, variables and operations from graph. Deletes build directory.
        """

        wdir = super().clear()
        for f in [f for f in os.listdir(wdir)
                  if "cpython" in f and ("rhs_func" in f or "pyrates_func" in f) and f[-3:] == ".so" ]:
            os.remove(f"{wdir}/{f}")

    def _solve(self, rhs_func, func_args, T, dt, dts, t, solver, output_indices, **kwargs):
        """

        Parameters
        ----------
        rhs_func
        func_args
        state_vars
        T
        dt
        dts
        t
        solver
        output_indices
        kwargs

        Returns
        -------

        """

        if self.pyauto_compat:

            from pyrates.utility.pyauto import PyAuto
            pyauto = PyAuto(working_dir=self._build_dir)
            dsmin = dt*1e-2
            auto_defs = {'DSMIN': dsmin, 'DSMAX': dt*1e2, 'NMX': int(T/dsmin), 'NPR': int(dts/dt)}
            for key, val in auto_defs.items():
                if key not in kwargs:
                    kwargs[key] = val
            pyauto.run(e='rhs_func', c='ivp', DS=dt, name='t', UZR={14: T}, STOP={'UZ1'}, **kwargs)

            extract = [f'U({i+1})' for i in range(self.ndim)]
            out_vars = [f'U({i[0]+self.idx_start if type(i) is list else i+self.idx_start})' for i in output_indices]
            extract.append('PAR(14)')

            results = pyauto.extract(keys=extract, cont='t')
            times = results.pop('PAR(14)')

            state_vars = self.get_var('y')
            for key, val in results.items():
                start, stop = key.index('('), key.index(')')
                idx = int(key[start+1:stop])
                state_vars[idx-self.idx_start] = val[-1]

            return times, [results[v] for v in out_vars]

        return super()._solve(rhs_func, func_args, T, dt, dts, t, solver, output_indices, **kwargs)

    def _create_op(self, op, name, *args):
        if not self.ops[op]['call']:
            raise NotImplementedError(f"The operator `{op}` is not implemented for this backend ({self.name}). "
                                      f"Please consider passing the required operation to the backend initialization "
                                      f"or choose another backend.")
        if op in ["=", "+=", "-=", "*=", "/="]:
            if len(args) > 2 and hasattr(args[2], 'shape') and len(args[2].shape) > 1 and \
                    'bool' not in str(type(args[2])):
                args_tmp = list(args)
                if not hasattr(args_tmp[2], 'short_name'):
                    idx = self.add_var(vtype='state_var', name='idx', value=args_tmp[2])
                else:
                    idx = args_tmp[2]
                var, upd, idx = self._process_update_args_old(args[0], args[1], idx)
                idx_str = ",".join([f"{idx.short_name}[:,{i}]" for i in range(idx.shape[1])])
                args = (var, upd, idx_str, idx)
            return FortranAssignOp(self.ops[op]['call'], self.ops[op]['name'], name, *args, idx_l=self.idx_l,
                                   idx_r=self.idx_r, build_dir=self._build_dir)
        elif op is "index":
            return FortranIndexOp(self.ops[op]['call'], self.ops[op]['name'], name, *args, idx_l=self.idx_l,
                                  idx_r=self.idx_r, build_dir=self._build_dir)
        else:
            if op is "cast":
                args = list(args)
                for dtype in self.dtypes:
                    if dtype in str(args[1]):
                        args[1] = f"np.{dtype}"
                        break
                args = tuple(args)
            return FortranOp(self.ops[op]['call'], self.ops[op]['name'], name, *args, build_dir=self._build_dir)

    def _process_vars(self):
        """

        Returns
        -------

        """
        state_vars, constants, var_map = [], [], {}
        s_idx, c_idx, s_len = 0, 0, 0
        for key, var in self.vars.items():
            key, state_var = self._is_state_var(key)
            if state_var:
                state_vars.append((key, var))
                var_map[key] = ('state_var', (s_idx, s_len))
                s_idx += 1
                s_len += 1
            elif var.vtype == 'constant' or (var.short_name not in self.lhs_vars and key != 'y'):
                if c_idx == 10:
                    for _ in range(4):
                        constants.append(())
                    c_idx = 14
                constants.append((key, var))
                var_map[key] = ('constant', c_idx+self.idx_start)
                c_idx += 1
        return state_vars, constants, var_map

    @staticmethod
    def _compare_shapes(op1: Any, op2: Any, index=False) -> bool:
        """Checks whether the shapes of op1 and op2 are compatible with each other.

        Parameters
        ----------
        op1
            First operator.
        op2
            Second operator.

        Returns
        -------
        bool
            If true, the shapes of op1 and op2 are compatible.

        """

        if hasattr(op1, 'shape') and hasattr(op2, 'shape'):
            if op1.shape == op2.shape:
                return True
            elif len(op1.shape) > 1 and len(op2.shape) > 1:
                return True
            elif len(op1.shape) == 0 and len(op2.shape) == 0:
                return True
            else:
                return False
        elif hasattr(op1, 'shape'):
            if sum(op1.shape) > 0:
                return False
            else:
                return True
        else:
            return True


class FortranGen(CodeGen):

    def add_code_line(self, code_str):
        """Add code line string to code.
        """
        if self.code:
            idx = -1
            while self.code[idx] != '\n' and abs(idx) < len(self.code):
                idx -= 1
            code_line = ''.join([self.code.pop(i) for i in range(idx+1, 0)]) + code_str
        else:
            code_line = code_str
        if "&" not in code_line:
            code_line = code_line.replace('\t', '')
            code_line = "\t" * self.lvl + code_line
        n = 60
        if len(code_line) > n:
            idx = self._find_first_op(code_line, start=0, stop=n)
            self.code.append(f"{code_line[0:idx]}")
            while idx < len(code_line):
                self.add_linebreak()
                idx_new = self._find_first_op(code_line, start=idx, stop=idx+n)
                self.code.append(f"     & {code_line[idx:idx+idx_new]}")
                idx += idx_new
        else:
            self.code.append(code_line)

    @staticmethod
    def _find_first_op(code, start, stop):
        if stop < len(code):
            code_tmp = code[start:stop]
            ops = ["+", "-", "*", "/", "**", "^", "%", "<", ">", "==", "!=", "<=", ">="]
            indices = [code_tmp.index(op) for op in ops if op in code_tmp]
            if indices and max(indices) > 0:
                return max(indices)
            idx = start
            for break_sign in [',', ')', ' ']:
                if break_sign in code_tmp:
                    idx_tmp = len(code_tmp) - code_tmp[::-1].index(break_sign)
                    if len(code_tmp)-idx_tmp < len(code_tmp)-idx:
                        idx = idx_tmp
            return idx
        return stop


def generate_func(self, return_key='f', omit_assign=False, return_dim=None, return_intent='out'):
    """Generates a function from operator value and arguments"""

    global module_counter

    # function head
    func_dict = {}
    func = FortranGen()
    func.add_linebreak()
    func.add_indent()
    fname = self.short_name.lower()
    func.add_code_line(f"subroutine {fname}({return_key}")
    for arg in self._op_dict['arg_names']:
        if arg != return_key:
            func.add_code_line(f",{arg}")
    func.add_code_line(")")
    func.add_linebreak()

    # argument type definition
    for arg, name in zip(self._op_dict['args'], self._op_dict['arg_names']):
        if name != return_key:
            dtype = "integer" if "int" in str(arg.vtype) else "double precision"
            dim = f"dimension({','.join([str(s) for s in arg.shape])}), " if arg.shape else ""
            func.add_code_line(f"{dtype}, {dim}intent(in) :: {name}")
            func.add_linebreak()
    out_dim = f"({','.join([str(s) for s in return_dim])})" if return_dim else ""
    func.add_code_line(f"double precision, intent({return_intent}) :: {return_key}{out_dim}")
    func.add_linebreak()

    func.add_code_line(f"{self._op_dict['value']}" if omit_assign else f"{return_key} = {self._op_dict['value']}")
    func.add_linebreak()
    func.add_code_line("end")
    func.add_linebreak()
    func.remove_indent()
    module_counter += 1
    fn = f"pyrates_func_{module_counter}"
    f2py.compile(func.generate(), modulename=fn, extension=".f", verbose=False,
                 source_fn=f"{self.build_dir}/{fn}.f" if self.build_dir else f"{fn}.f")
    exec(f"from pyrates_func_{module_counter} import {fname}", globals())
    func_dict[self.short_name] = globals().pop(fname)
    return func_dict
