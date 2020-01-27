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
from numpy import f2py

# pyrates internal imports
from .numpy_backend import NumpyBackend, PyRatesAssignOp, PyRatesOp, CodeGen
from .parser import replace

# meta infos
__author__ = "Richard Gast"
__status__ = "development"

module_counter = 0


class FortranOp(PyRatesOp):

    def _generate_func(self):
        """Generates a function from operator value and arguments"""

        global module_counter

        # function head
        func_dict = {}
        func = FortranGen()
        func.add_linebreak()
        func.add_indent()
        func.add_code_line(f"subroutine {self.short_name}(f")
        for arg in self._op_dict['arg_names']:
            func.add_code_line(f",{arg}")
        func.add_code_line(")")
        func.add_linebreak()

        # argument type definition
        for arg, name in zip(self._op_dict['args'], self._op_dict['arg_names']):
            dtype = "integer" if "int" in str(arg.vtype) else "double precision"
            dim = f"dimension({','.join([str(s) for s in arg.shape])}), " if arg.shape else ""
            func.add_code_line(f"{dtype}, {dim}intent(in) :: {name}")
            func.add_linebreak()
        func.add_code_line("double precision, intent(out) :: f")
        func.add_linebreak()

        func.add_code_line(f"f = {self._op_dict['value']}")
        func.add_linebreak()
        func.add_code_line("end")
        func.add_linebreak()
        func.remove_indent()
        module_counter += 1
        f2py.compile(func.generate(), modulename=f"pyrates_func_{module_counter}", extension=".f",
                     source_fn=f"/tmp/pyrates_func_{module_counter}.f", verbose=False)
        exec(f"from pyrates_func_{module_counter} import {self.short_name}", globals(), func_dict)
        return func_dict


class FortranIndexOp(FortranOp):

    @classmethod
    def generate_op(cls, op, args):
        """Generates the index function string, call signature etc.
        """

        # initialization
        results = {'value': None, 'args': [], 'arg_names': [], 'is_constant': False, 'shape': (), 'dtype': 'float32',
                   'input_ops': []}

        # extract variable
        ##################

        var_tmp = args[0]
        n_vars = 0

        if PyRatesOp.__subclasscheck__(type(var_tmp)):

            # nest pyrates operations and their arguments into the indexing operation
            var = var_tmp.lhs if hasattr(var_tmp, 'lhs') else var_tmp.value
            pop_indices = []
            for arg_tmp in var_tmp.arg_names:
                if arg_tmp in results['arg_names']:
                    pop_indices.append(var_tmp.arg_names.index(arg_tmp))
            new_args = var_tmp.args.copy()
            new_arg_names = var_tmp.arg_names.copy()
            for i, pop_idx in enumerate(pop_indices):
                new_args.pop(pop_idx - i)
                new_arg_names.pop(pop_idx - i)
            results['input_ops'].append(var_tmp.name)
            results['args'] += new_args
            results['arg_names'] += new_arg_names
            n_vars += 1

        elif hasattr(var_tmp, 'vtype'):

            # parse pyrates variables into the indexing operation
            if var_tmp.vtype != 'constant' or var_tmp.shape:
                var = var_tmp.short_name
                results['args'].append(var_tmp)
                results['arg_names'].append(var)
                n_vars += 1
            else:
                var = var_tmp.value

        else:

            raise ValueError(f'Variable type for indexing not understood: {var_tmp}. '
                             f'Please consider another form of variable declaration or indexing.')

        # extract idx
        #############

        idx = args[1]

        if PyRatesOp.__subclasscheck__(type(idx)):

            # nest pyrates operations and their arguments into the indexing operation
            var_idx = idx.lhs if hasattr(idx, 'lhs') else idx.value
            pop_indices = []
            for arg_tmp in idx.arg_names:
                if arg_tmp in results['arg_names']:
                    pop_indices.append(idx.arg_names.index(arg_tmp))
            new_args = idx.args.copy()
            new_arg_names = idx.arg_names.copy()
            for i, pop_idx in enumerate(pop_indices):
                new_args.pop(pop_idx - i)
                new_arg_names.pop(pop_idx - i)
            results['input_ops'].append(idx.name)
            results['args'] += new_args
            results['arg_names'] += new_arg_names
            n_vars += 1

        elif hasattr(idx, 'vtype'):

            # parse pyrates variables into the indexing operation
            key = idx.short_name
            if idx.vtype != 'constant' or idx.shape:
                var_idx = key
                results['args'].append(idx)
                results['arg_names'].append(key)
                n_vars += 1
            else:
                var_idx = idx.value

        elif type(idx) is str or "int" in str(type(idx)) or "float" in str(type(idx)):

            # parse constant numpy-like indices into the indexing operation
            pop_indices = []
            for i, arg in enumerate(args[2:]):
                if hasattr(arg, 'short_name') and arg.short_name in idx:
                    if hasattr(arg, 'vtype') and arg.vtype == 'constant' and sum(arg.shape) < 2:
                        idx = replace(idx, arg.short_name, f"{int(arg)}")
                        pop_indices.append(i + 2)
                    elif hasattr(arg, 'value'):
                        idx = replace(idx, arg.short_name, arg.value)
                        pop_indices.append(i + 2)
                        results['args'] += arg.args
                        results['arg_names'] += arg.arg_names
            var_idx = idx
            args = list(args)
            for idx in pop_indices[::-1]:
                args.pop(idx)
            args = tuple(args)

        elif type(idx) in (list, tuple):

            # parse list of constant indices into the operation
            var_idx = f"{','.join(i for i in idx)}"

        else:
            raise ValueError(f'Index type not understood: {idx}. Please consider another form of variable indexing.')

        # setup function head
        #####################

        # remaining arguments
        results, results_args, results_arg_names, n_vars_tmp = cls._process_args(args[2:], results)
        n_vars += n_vars_tmp

        # apply index to variable and return variable
        #############################################

        # generate op
        var_idx = f"({','.join([str(int(idx) + 1) for idx in var_idx.split(',')])})"
        results['value'] = f"{var}{var_idx}"

        # check whether any state variables are part of the indexing operation or not
        if n_vars == 0:
            results['is_constant'] = True

        return results


class FortranBackend(NumpyBackend):
    def __init__(self,
                 ops: Optional[Dict[str, str]] = None,
                 dtypes: Optional[Dict[str, object]] = None,
                 name: str = 'net_0',
                 float_default_type: str = 'float32',
                 imports: Optional[List[str]] = None,
                 build_dir: Optional[str] = None,
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
                 "sum": {'name': "fortran_sum", 'call': ""},
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
                 "index": {'name': "pyrates_index", 'call': "fortran_index"},
                 "mask": {'name': "pyrates_mask", 'call': ""},
                 "group": {'name': "pyrates_group", 'call': ""},
                 "asarray": {'name': "fortran_asarray", 'call': ""},
                 "no_op": {'name': "pyrates_identity", 'call': ""},
                 "interpolate": {'name': "pyrates_interpolate", 'call': ""},
                 "interpolate_1d": {'name': "pyrates_interpolate_1d", 'call': ""},
                 "interpolate_nd": {'name': "pyrates_interpolate_nd", 'call': ""},
                 }
        if ops:
            ops_f.update(ops)
        super().__init__(ops=ops_f, dtypes=dtypes, name=name, float_default_type=float_default_type,
                         imports=imports, build_dir=build_dir)

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

        # create directory in which to store rhs function
        orig_path = os.getcwd()
        if build_dir:
            os.makedirs(build_dir, exist_ok=True)
        dir_name = f"{build_dir}/pyrates_build" if build_dir else "pyrates_build"
        try:
            os.mkdir(dir_name)
        except FileExistsError:
            pass
        os.chdir(dir_name)
        try:
            os.mkdir(self.name)
        except FileExistsError:
            rmtree(self.name)
            os.mkdir(self.name)
        for key in sys.modules.copy():
            if self.name in key:
                del sys.modules[key]
        os.chdir(self.name)
        net_dir = os.getcwd()
        self._build_dir = net_dir
        sys.path.append(net_dir)

        # remove previously imported rhs_funcs from system
        if 'rhs_func' in sys.modules:
            del sys.modules['rhs_func']

        # collect state variable and parameter vectors
        state_vars, params, var_map = self._process_vars()

        # create rhs evaluation function
        ################################

        # set up file header
        func_gen = CodeGen()
        for import_line in self._imports:
            func_gen.add_code_line(import_line)
            func_gen.add_linebreak()
        func_gen.add_linebreak()

        # define function head
        func_gen.add_code_line("SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)")
        func_gen.add_linebreak()
        func_gen.add_code_line("IMPLICIT NONE")
        func_gen.add_linebreak()
        func_gen.add_code_line("INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC")
        func_gen.add_linebreak()
        func_gen.add_code_line("DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)")
        func_gen.add_linebreak()
        func_gen.add_code_line("DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)")
        func_gen.add_linebreak()
        func_gen.add_code_line("DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)")
        func_gen.add_linebreak()

        # declare variable types
        func_gen.add_code_line("DOUBLE PRECISION ")
        for key in var_map:
            var = self.get_var(key)
            if "float" in str(var.vtype):
                func_gen.add_code_line(f"{var},")
        if "," in func_gen.code[-1]:
            func_gen.code[-1] = func_gen.code[-1][:-1]
        else:
            func_gen.code.pop(-1)
        func_gen.add_linebreak()
        func_gen.add_code_line("INTEGER ")
        for key in var_map:
            var = self.get_var(key)
            if "int" in str(var.vtype):
                func_gen.add_code_line(f"{var},")
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
                var = params[idx][1]
                func_gen.add_code_line(f"{var.short_name} = PAR({idx})")
                func_gen.add_linebreak()
                args[idx] = var
                if var.short_name != "y_delta":
                    updates.append(f"{var.short_name}")
                    indices.append(i)
                    i += 1
        func_gen.add_linebreak()

        # extract state variables from input vector y
        func_gen.add_code_line("# extract state variables from input vector")
        func_gen.add_linebreak()
        for key, (vtype, idx) in var_map.items():
            var = self.get_var(key)
            if vtype == 'state_var':
                func_gen.add_code_line(f"{var.short_name} = U({idx})")
                func_gen.add_linebreak()
        func_gen.add_linebreak()

        # add equations
        func_gen.add_code_line("# calculate right-hand side update of equation system")
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
        func_gen.add_code_line("# update system parameters")
        func_gen.add_linebreak()
        for upd, idx in arg_updates:
            update_str = f"params[{idx}] = {upd}"
            if f"    {update_str}" not in func_gen.code:
                func_gen.add_code_line(update_str)
                func_gen.add_linebreak()

        # add return line
        func_gen.add_code_line(f"return {self.vars['y_delta'].short_name}")
        func_gen.add_linebreak()

        # save rhs function to file
        with open('rhs_func.py', 'w') as f:
            f.writelines(func_gen.code)
            f.close()

        # import function from file
        exec("from rhs_func import rhs_eval", globals())
        rhs_eval = globals().pop('rhs_eval')
        os.chdir(orig_path)

        # apply function decorator
        if decorator:
            rhs_eval = decorator(rhs_eval, **kwargs)

        return rhs_eval, args, state_vars, var_map

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
            return PyRatesAssignOp(self.ops[op]['call'], self.ops[op]['name'], name, *args)
        elif op is "index":
            return FortranIndexOp(self.ops[op]['call'], self.ops[op]['name'], name, *args)
        else:
            if op is "cast":
                args = list(args)
                for dtype in self.dtypes:
                    if dtype in str(args[1]):
                        args[1] = f"np.{dtype}"
                        break
                args = tuple(args)
            return FortranOp(self.ops[op]['call'], self.ops[op]['name'], name, *args)

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
        code_line = "\t" * self.lvl + code_str if self.code[-1] == '\n' else code_str
        n = 72
        if len(code_line) > n:
            idx = self._find_first_op(code_line, start=0, stop=n)
            self.code.append(f"{code_line[0:idx]}")
            while idx < n:
                self.add_linebreak()
                idx_new = self._find_first_op(code_line, start=idx, stop=idx+n)
                self.code.append("     " f"& {code_line[idx:idx+idx_new]}")
                idx += idx_new
        else:
            self.code.append(code_line)

    def _find_first_op(self, code, start, stop):
        if stop < len(code):
            code_tmp = code[start:stop]
            ops = ["+", "-", "*", "/", "**", "^", "%", "<", ">", "==", "!=", "<=", ">="]
            indices = [code_tmp.index(op) for op in ops if op in code_tmp]
            if indices and max(indices) > 0:
                return max(indices)
        return stop
