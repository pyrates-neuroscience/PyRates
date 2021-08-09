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
- Numpy: NumpyBackend.
- Tensorflow: TensorflowBackend.
- Fortran: FortranBackend (experimental).

"""

# external imports
import time
from typing import Optional, Dict, List, Union, Any, Callable
import numpy as np
from copy import deepcopy
import os
import sys
from shutil import rmtree
import warnings

# pyrates internal imports
from .funcs import *
from .parser import replace, CodeGen, sympify
from .computegraph import ComputeGraph

#################################
# classes for backend variables #
#################################


class BaseVar(np.ndarray):
    """Base class for adding variables to the PyRates compute graph. Creates a numpy array with additional attributes
    for variable identification/retrieval from graph. Should be used as parent class for custom variable classes.

    Parameters
    ----------
    vtype
        Type of the variable. Can be either `constant` or `state_variable`. Constant variables are necessary to perform
        certain graph optimizations previous to run time.
    dtype
        Data-type of the variable. For valid data-types, check the documentation of the backend in use.
    shape
        Shape of the variable.
    value
        Value of the variable. If scalar, please provide the shape in addition.
    name
        Full name of the variable (including the node and operator it belongs to).
    short_name
        Name of the variable excluding its node and operator.

    Returns
    -------
    BaseVar
        Instance of BaseVar.
    """

    def __new__(cls, vtype: str, backend: Any, name: str, dtype: Optional[str] = None,
                shape: Optional[tuple] = None, value: Optional[Any] = None, squeeze: bool = True):
        """Creates new instance of BaseVar.
        """

        # check whether necessary arguments were provided
        if all([arg is None for arg in [shape, value, dtype]]):
            raise ValueError('Either `value` or `shape` and `dtype` need to be provided')

        # get shape
        if not shape:
            shape = value.shape if hasattr(value, 'shape') else np.shape(value)

        # get data type
        if not dtype:
            if hasattr(value, 'dtype'):
                dtype = value.dtype
            else:
                try:
                    dtype = np.dtype(value)
                except TypeError:
                    dtype = type(value)
        dtype = dtype.name if hasattr(dtype, 'name') else str(dtype)
        if dtype in backend.dtypes:
            dtype = backend.dtypes[dtype]
        else:
            for dtype_tmp in backend.dtypes:
                if dtype_tmp in dtype:
                    dtype = backend.dtypes[dtype_tmp]
                    break
            else:
                dtype = backend._float_def
                warnings.warn(f'WARNING! Unknown data type of variable {name}: {dtype}. '
                              f'Datatype will be set to default type: {dtype}.')

        # create variable
        if vtype == 'state_var' and 1 in tuple(shape) and squeeze:
            idx = tuple(shape).index(1)
            shape = list(shape)
            shape.pop(idx)
            shape = tuple(shape)
        if callable(value):
            obj = value
        else:
            value = cls._get_value(value, dtype, shape)
            if squeeze:
                value = cls.squeeze(value)
            obj = cls._get_var(value, name, dtype)

        # store additional attributes on variable object
        obj.short_name = name.split('/')[-1]
        if not hasattr(obj, 'name'):
            obj.name = name
        else:
            name = obj.name
        obj.vtype = vtype

        return obj, name

    def numpy(self):
        """Returns current value of BaseVar.
        """
        try:
            return self[:]
        except IndexError:
            return self

    def reshape(self, shape: tuple, **kwargs):

        obj = super().reshape(shape, **kwargs)
        if hasattr(self, 'name'):
            obj.name = self.name
        if hasattr(self, 'short_name'):
            obj.short_name = self.short_name
        if hasattr(self, 'vtype'):
            obj.vtype = self.vtype
        return obj

    @staticmethod
    def _get_value(value, dtype, shape):
        """Defines initial value of variable.
        """
        if value is None:
            return np.zeros(shape=shape, dtype=dtype)
        elif not hasattr(value, 'shape'):
            if type(value) is list:
                return np.zeros(shape=shape, dtype=dtype) + np.asarray(value, dtype=dtype).reshape(shape)
            else:
                return np.zeros(shape=shape, dtype=dtype) + value
        elif shape:
            if value.shape == shape:
                return value
            elif sum(shape) < sum(value.shape):
                return value.squeeze()
            else:
                idx = ",".join("None" if s == 1 else ":" for s in shape)
                return eval(f'value[{idx}]')
        else:
            return np.asarray(value, dtype=dtype)

    @staticmethod
    def squeeze(var, axis=None):
        return var.squeeze(axis)

    def _is_equal_to(self, v: np.ndarray):
        if self.short_name == v.short_name and self.shape == v.shape:
            if sum(v.shape) > 1:
                return all(self.flatten() != v.flatten())
            else:
                return self != v

    @classmethod
    def _get_var(cls, value, name, dtype):
        """Creates new numpy array from BaseVar.
        """
        return np.array(value).view(cls)

    def __deepcopy__(self, memodict={}):
        obj = super().__deepcopy__(memodict)
        if not hasattr(obj, 'name') and hasattr(self, 'name'):
            obj.name = self.name
        if not hasattr(obj, 'short_name') and hasattr(self, 'short_name'):
            obj.short_name = self.short_name
        if not hasattr(obj, 'vtype') and hasattr(self, 'vtype'):
            obj.vtype = self.vtype
        return obj

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(str(self))

    @staticmethod
    def __subclasscheck__(subclass):
        if np.ndarray.__subclasscheck__(subclass):
            return True
        else:
            return interp1d.__subclasscheck__(subclass)


#######################################
# classes for backend functionalities #
#######################################


class BaseBackend(object):
    """

    """

    def __init__(self,
                 ops: Optional[Dict[str, str]] = None,
                 dtypes: Optional[Dict[str, object]] = None,
                 name: str = 'net_0',
                 float_default_type: str = 'float32',
                 imports: Optional[List[str]] = None,
                 build_dir: str = None,
                 compute_graph: ComputeGraph = None,
                 **kwargs
                 ) -> None:
        """Instantiates the standard, numpy-based backend, i.e. a compute graph with numpy operations.
        """

        super().__init__()

        # define operations and datatypes of the backend
        ################################################

        # base math operations
        self.ops = {
                    "max": {'func': np.maximum, 'str': "np.maximum"},
                    "min": {'func': np.minimum, 'str': "np.minimum"},
                    "argmax": {'func': np.argmax, 'str': "np.argmax"},
                    "argmin": {'func': np.argmin, 'str': "np.argmin"},
                    "round": {'func': np.round, 'str': "np.round"},
                    "sum": {'func': np.sum, 'str': "np.sum"},
                    "mean": {'func': np.mean, 'str': "np.mean"},
                    "matmul": {'func': np.matmul, 'str': "np.matmul"},
                    "concat": {'func': np.concatenate, 'str': "np.concatenate"},
                    "reshape": {'func': np.reshape, 'str': "np.reshape"},
                    "append": {'func': np.append, 'str': "np.append"},
                    "shape": {'func': np.shape, 'str': "np.shape"},
                    "dtype": {'func': np.dtype, 'str': "np.dtype"},
                    'squeeze': {'func': np.squeeze, 'str': "np.squeeze"},
                    'expand': {'func': np.expand_dims, 'str': "np.expand_dims"},
                    "roll": {'func': np.roll, 'str': "np.roll"},
                    "cast": {'func': np.asarray, 'str': "np.asarray"},
                    "randn": {'func': np.random.randn, 'str': "np.randn"},
                    "ones": {'func': np.ones, 'str': "np.ones"},
                    "zeros": {'func': np.zeros, 'str': "np.zeros"},
                    "range": {'func': np.arange, 'str': "np.arange"},
                    "softmax": {'func': pr_softmax, 'str': "pr_softmax"},
                    "sigmoid": {'func': pr_sigmoid, 'str': "pr_sigmoid"},
                    "tanh": {'func': np.tanh, 'str': "np.tanh"},
                    "no_op": {'func': pr_identity, 'str': "pr_identity"},
                    "interp": {'func': pr_interp, 'str': "pr_interp"},
                    "interpolate_1d": {'func': pr_interp_1d, 'str': "pr_interp_1d"},
                    "interpolate_nd": {'func': pr_interp_nd, 'str': "pr_interp_nd"},
                    "index": {'func': pr_base_index, 'str': "pr_base_index"},
                    }
        if ops:
            self.ops.update(ops)

        # base data-types
        self.dtypes = {"float16": np.float16,
                       "float32": np.float32,
                       "float64": np.float64,
                       "int16": np.int16,
                       "int32": np.int32,
                       "int64": np.int64,
                       "uint16": np.uint16,
                       "uint32": np.uint32,
                       "uint64": np.uint64,
                       "complex64": np.complex64,
                       "complex128": np.complex128,
                       "bool": np.bool
                       }
        if dtypes:
            self.dtypes.update(dtypes)

        # initialize compute graph
        self.graph = compute_graph if compute_graph else ComputeGraph(**kwargs)

        # further attributes
        self._var_map = dict()
        self.state_vars = []
        self.non_state_vars = []
        self._float_def = self.dtypes[float_default_type]
        self.name = name
        self._imports = ["from numpy import *", "from pyrates.backend.funcs import *"]
        if imports:
            for imp in imports:
                if imp not in self._imports:
                    self._imports.append(imp)
        self._input_names = []
        self.type = 'numpy'
        self._orig_dir = None
        self._build_dir = None

        # create build dir
        self._orig_dir = os.getcwd()
        self._build_dir = f"{build_dir}/{self.name}" if build_dir else ""
        if build_dir:
            os.makedirs(build_dir, exist_ok=True)
            try:
                os.mkdir(build_dir)
            except FileExistsError:
                pass
            except FileNotFoundError as e:
                # for debugging
                raise e
            try:
                os.mkdir(self._build_dir)
            except FileExistsError:
                rmtree(self._build_dir)
                os.mkdir(self._build_dir)
            sys.path.append(self._build_dir)

    def add_var(self,
                name: str,
                vtype: str,
                value: Optional[Any] = None,
                shape: Optional[Union[tuple, list, np.shape]] = None,
                dtype: Optional[Union[str, np.dtype]] = None,
                **kwargs
                ) -> tuple:
        """Adds a variable to the backend.

        Parameters
        ----------
        vtype
            Variable type. Can be
                - `state_var` for variables that can change over time.
                - `constant` for non-changing variables.
        name
            Name of the variable.
        value
            Value of the variable. Not needed for placeholders.
        shape
            Shape of the variable.
        dtype
            Datatype of the variable.
        kwargs
            Additional keyword arguments passed to `computegraph.ComputeGraph.add_var`.

        Returns
        -------
        tuple
            (1) variable name, (2) dictionary with all variable information.

        """

        # extract variable scope
        scope = kwargs.pop('scope', None)
        if scope:
            name = f'{scope}/{name}'

        # if variable already exists, return it
        try:
            return name, self.get_var(name, get_key=False)
        except KeyError:
            pass

        # create variable
        var, name = BaseVar(vtype=vtype, dtype=dtype, shape=shape, value=value, name=name, backend=self,
                            squeeze=kwargs.pop('squeeze', True))

        # add variable to compute graph
        label, cg_var = self.graph.add_var(label=var.short_name, value=var, vtype=vtype, **kwargs)

        # save to dict
        self._var_map[name] = label

        return label, cg_var

    def add_op(self,
               inputs: Union[list, tuple],
               name: str,
               **kwargs
               ) -> tuple:
        """Add operation to the backend.

        Parameters
        ----------
        inputs
            List with the names of all compute graph nodes that should enter as input in this operation.
        name
            Key of the operation. If it is a key of `backend.ops`, the function call at `backend.ops[name]['func']` will
             be used.
        kwargs
            Additional keyword arguments passed to `computegraph.ComputeGraph.add_op`.

        Returns
        -------
        tuple
            (1) The key for extracting the operator from the compute graph,
            (2) the dictionary of the compute graph operator node.

        """

        # extract operator scope
        scope = kwargs.pop('scope', None)
        if scope:
            name = f'{scope}/{name}'

        # add operator to compute graph
        label, cg_var = self.graph.add_op(inputs, label=name, **kwargs)

        # save to dict
        self._var_map[name] = label

        return label, cg_var

    def get_var(self, name: str, get_key: bool = False, **kwargs) -> Union[dict, str]:
        """Retrieve variable from graph.

        Parameters
        ----------
        name
            Identifier of the variable.
        get_key
            If true, only the name of the variable in the compute graph is returned.

        Returns
        -------
        Union[dict, str]
            Variable dictionary from graph.

        """

        # extract operator scope
        scope = kwargs.pop('scope', None)
        if scope:
            name = f'{scope}/{name}'
        try:
            return self._var_map[name] if get_key else self.graph.nodes[self._var_map[name]]
        except KeyError:
            return name if get_key else self.graph.nodes[name]

    def run(self, T: float, dt: float, dts: float = None, outputs: dict = None, solver: str = None,
            in_place: bool = True, func_name: str = None, file_name: str = None, compile_kwargs: dict = None, **kwargs) -> dict:

        # TODO: Implement input/output functionalities

        # preparations
        ##############

        if not compile_kwargs:
            compile_kwargs = dict()
        if not func_name and solver:
            func_name = 'rhs_func'
        if not file_name and solver:
            file_name = 'pyrates_rhs_func'

        # network specs
        run_info = self._compile(in_place=in_place, func_name=func_name, file_name=file_name, **compile_kwargs)
        state_vec = self.get_var(run_info['state_vec'])['value']
        rhs = np.zeros_like(state_vec)

        # simulation specs
        steps = int(np.round(T / dt))
        state_rec = np.zeros((steps, state_vec.shape[0]))
        if dts:
            times = np.arange(0, T, dts)
            kwargs['t_eval'] = times
        else:
            times = np.arange(0, T, dt)

        # perform simulation
        ####################

        if solver:

            # numerical integration via a system update function generated from the compute graph
            rhs_func = run_info['func']
            func_args = run_info['func_args'][3:]
            args = tuple([self.get_var(arg)['value'] for arg in func_args])

            if solver == 'euler':

                # perform integration via standard Euler method
                t = 0.0
                for step in range(steps):
                    state_vec += dt * rhs_func(t, state_vec, rhs, *args)
                    t += dt
                    state_rec[step, :] = state_vec

            else:

                from scipy.integrate import solve_ivp

                # solve via scipy's ode integration function
                fun = lambda t, y: rhs_func(t, y, rhs, *args)
                t = 0.0
                results = solve_ivp(fun=fun, t_span=(t, T), y0=state_vec, first_step=dt, **kwargs)
                state_rec = results['y'].T

        else:

            # numerical integration by directly traversing the compute graph at each integration step
            for step in range(steps):
                state_vec += dt * np.concatenate(self.graph._out_nodes())
                state_rec[step, :] = state_vec

        # reduce state recordings to requested state variables
        final_results = {}
        for key, var in outputs.items():
            idx = run_info['old_state_vars'].index(var)
            idx = run_info['vec_indices'][idx]
            if type(idx) is tuple and idx[1] - idx[0] == 1:
                idx = (idx[0],)
            elif type(idx) is int:
                idx = (idx,)
            final_results[key] = state_rec[:, idx]
        final_results['time'] = times

        return final_results

    def _compile(self, in_place: bool = True, func_name: str = None, file_name: str = None, **kwargs) -> dict:

        # finalize compute graph
        self.graph = self.graph.compile(in_place=in_place)

        # create state variable vector
        vars, indices = [], []
        idx = 0
        for var in self.state_vars:
            v = self.get_var(var)['value']
            if not v.shape:
                v = np.reshape(v, (1,))
            shape = v.shape[0]
            vars.append(v)
            indices.append((idx, idx+shape))
            idx += shape
        state_vec = np.concatenate(vars)
        state_var_key, state_var = self.add_var(vtype='state_var', name='state_vec', value=state_vec)

        # store new state-var vector in graph
        old_state_vars = []
        for var, (idx_l, idx_r) in zip(self.state_vars, indices):
            v = self.get_var(var)
            v['value'] = state_var['value'][idx_l:idx_r]
            v['index'] = (idx_l, idx_r)
            old_state_vars.append(self.get_var(var, get_key=True))

        # create right-hand side update vector (also serves as test-run of network equations)
        rhs_vec = np.concatenate([self.graph.eval_node(self.get_var(v, get_key=True)) for v in self.state_vars])
        rhs_var_key, rhs_var = self.add_var(vtype='state_var', name='state_vec_update', value=np.zeros_like(rhs_vec))

        # check consistency of state var and rhs vector
        if state_vec.shape != rhs_vec.shape:
            raise ValueError(
                'Shapes of state variable vector and right-hand side updates of these state variables do'
                'not match. Please check the definition of variable types in the model definition.')

        return_dict = {'old_state_vars': old_state_vars, 'state_vec': state_var_key, 'vec_indices': indices}

        if func_name or file_name:

            if not func_name:
                func_name = 'rhs_func'

            code_gen = CodeGen()

            # generate function body with all equations and assignments
            func_args, code_gen_tmp = self._graph_to_str(rhs_indices=indices, state_var=state_var_key,
                                                         rhs_var=rhs_var_key)
            func_body = code_gen_tmp.generate()

            # generate function head
            func_args, code_gen = self._generate_func_head(code_gen=code_gen, state_var=state_var_key,
                                                           func_args=func_args, func_name=func_name,
                                                           imports=self._imports)

            # add lines from function body after function head
            code_gen.add_code_line(func_body)

            # generate function tail
            code_gen = self._generate_func_tail(code_gen=code_gen, rhs_var=rhs_var_key)

            # finalize function string
            func_str = code_gen.generate()

            # write function string to file
            func = self._generate_func(func_name=func_name, file_name=file_name, func_str=func_str,
                                       build_dir=self._build_dir, **kwargs)
            return_dict['func'] = func
            return_dict['func_args'] = func_args

        return return_dict

    def _graph_to_str(self, code_gen: CodeGen = None, rhs_indices: list = None, state_var: str = None,
                      rhs_var: str = None):

        # preparations
        if code_gen is None:
            code_gen = CodeGen()
        backend_funcs = {key: val['str'] for key, val in self.ops.items()}
        code_gen.add_linebreak()

        # extract state variable from state vector if necessary
        if rhs_indices:

            for (idx_l, idx_r), var in zip(rhs_indices, self.state_vars):
                code_gen.add_code_line(f"{self.get_var(var, get_key=True)} = {state_var}[{idx_l}:{idx_r}]")

            code_gen.add_linebreak()

        # get equation string and argument list for each non-DE node at the end of the compute graph hierarchy
        func_args2, code_gen = self._generate_update_equations(code_gen, nodes=self.non_state_vars, funcs=backend_funcs)

        # get equation string and argument list for each DE node at the end of the compute graph hierarchy
        func_args1, code_gen = self._generate_update_equations(code_gen, nodes=self.state_vars, rhs_var=rhs_var,
                                                                  indices=rhs_indices, funcs=backend_funcs)

        return func_args1 + func_args2, code_gen

    def _generate_update_equations(self, code_gen: CodeGen, nodes: list, rhs_var: str = None, indices: list = None,
                                   funcs: dict = None):

        # collect right-hand side expression and all input variables to these expressions
        func_args, expressions, var_names = [], [], []
        for node in nodes:
            node_dict = self.get_var(node)
            expr_args, expr = self.graph.node_to_expr(node_dict['update'], **funcs)
            func_args.extend(expr_args)
            expressions.append(expr)
            var_names.append(node_dict['symbol'].name)

        # add the left-hand side assignments of the collected right-hand side expressions to the code generator
        if rhs_var:

            # differential equation (DE) update
            if indices:

                # DE updates stored in a state-vector
                for (idx_l, idx_r), expr in zip(indices, expressions):
                    expr_str = self._expr_to_str(expr)
                    code_gen.add_code_line(f"{rhs_var}[{idx_l}:{idx_r}] = {expr_str}")

            else:

                if len(nodes) > 1:
                    raise ValueError('Received a request to update a variable via multiple right-hand side expressions.'
                                     )

                # DE update stored in a single variable
                expr_str = self._expr_to_str(expressions[0])
                code_gen.add_code_line(f"{rhs_var} = {expr_str}")

            # add rhs var to function arguments
            func_args = [rhs_var] + func_args

        else:

            # non-differential equation update
            if indices:

                # non-DE update stored in a variable slice
                for (idx_l, idx_r), expr, target_var in zip(indices, expressions, var_names):
                    expr_str = self._expr_to_str(expr)
                    code_gen.add_code_line(f"{target_var}[{idx_l}:{idx_r}] = {expr_str}")

            else:

                # non-DE update stored in a single variable
                for target_var, expr in zip(var_names, expressions):
                    expr_str = self._expr_to_str(expr)
                    code_gen.add_code_line(f"{target_var} = {expr_str}")

        code_gen.add_linebreak()

        return func_args, code_gen

    @staticmethod
    def _generate_func_head(func_name: str, code_gen: CodeGen = None, state_var: str = None, func_args: list = None,
                            imports: list = None):

        if code_gen is None:
            code_gen = CodeGen()
        if not func_args:
            func_args = []
        state_vars = ['t', state_var]
        _, indices = np.unique(func_args, return_index=True)
        func_args = state_vars + [func_args[idx] for idx in np.sort(indices)]

        if imports:

            # add imports at beginning of file
            for imp in imports:
                code_gen.add_code_line(imp)
            code_gen.add_linebreak()

        # add function header
        code_gen.add_linebreak()
        code_gen.add_code_line(f"def {func_name}({','.join(func_args)}):")
        code_gen.add_indent()

        return func_args, code_gen

    @staticmethod
    def _generate_func_tail(code_gen: CodeGen = None, rhs_var: str = None):

        if code_gen is None:
            code_gen = CodeGen()

        code_gen.add_code_line(f"return {rhs_var}")
        code_gen.remove_indent()

        return code_gen

    @staticmethod
    def _generate_func(file_name: str, func_name: str, func_str: str, build_dir: str, decorator: Any = None,
                       **kwargs):

        if file_name:

            # save rhs function to file
            if build_dir:
                file_name = f'{build_dir}/{file_name}'
            with open(f'{file_name}.py', 'w') as f:
                f.writelines(func_str)
                f.close()

            # import function from file
            exec(f"from {file_name} import {func_name}", globals())
            rhs_eval = globals().pop(func_name)

        else:

            # generate function from string
            exec(func_str, globals())
            rhs_eval = globals().pop(func_name)

        # apply function decorator
        if decorator:
            rhs_eval = decorator(rhs_eval, **kwargs)

        return rhs_eval

    @staticmethod
    def _expr_to_str(expr: Any) -> str:
        expr_str = str(expr)
        while 'apply_index' in expr_str:
            # replace `apply_index` calls with brackets-based indexing
            start = expr_str.find('apply_index')
            end = expr_str.find(')', start=start)
            expr_str = expr_str.replace(expr_str[start:end], f"{expr.args[0]}[{expr.args[1]}]")
        return expr_str


# class PyRatesOp:
#     """Base class for adding operations on variables to the PyRates compute graph. Should be used as parent class for
#     custom operator classes.
#
#     Parameters
#     ----------
#     op
#         Call-signature of the operation/function.
#     short_name
#         Name of the operation within the equations.
#     name
#         Name of the operation/function.
#     args
#         Arguments to the function.
#
#     """
#
#     var_class = BaseVar
#
#     def __init__(self, op: str, short_name: str, name, *args, **kwargs) -> None:
#         """Instantiates PyRates operator.
#         """
#
#         # set identifiers of operator
#         self.op = op
#         self.short_name = short_name if short_name in name else name.split('/')[-1]
#         self.name = name
#
#         # generate function string
#         self._op_dict = self.generate_op_str(op, args, **kwargs)
#
#         # extract information from parsing results
#         self.value = self._op_dict['value']
#         self.arg_names, args = self._get_unique_args(self._op_dict['arg_names'], self._op_dict['args'])
#         self.input_ops = self._op_dict['input_ops']
#         self.is_constant = self._op_dict['is_constant']
#
#         # build final op
#         self.shape = ()
#         self.dtype = 'float'
#         self._callable = lambda x: x
#         self.args = ()
#         self.build_op(args)
#
#     def numpy(self):
#         """Evaluates the return values of the PyRates operation.
#         """
#         result = self._callable(*self.args)
#         self._check_numerics(result, self.name)
#         return result
#
#     def _generate_func(self):
#         """Generates a function from operator value and arguments"""
#         func_dict = {}
#         func = CodeGen()
#         func.add_code_line(f"def {self.short_name}({','.join([arg for arg in self._op_dict['arg_names']])}):")
#         func.add_indent()
#         func.add_code_line(f"return {self._op_dict['value']}")
#         exec(func.generate(), globals(), func_dict)
#         return func_dict
#
#     def build_op(self, args):
#         """Builds the function of the operator, tests its functionality and stores shape and data-type of the result
#         on the instance.
#         """
#
#         func_dict = self._generate_func()
#         self._callable = func_dict.pop(self.short_name)
#
#         # test function
#         self.args = self._deepcopy(args)
#         result = self.numpy() if 'no_op' not in self.short_name else self.args[0]
#         self.args = args
#
#         # remember output shape and data-type
#         self.shape = result.shape if hasattr(result, 'shape') else ()
#         self.dtype = result.dtype if hasattr(result, 'dtype') else type(result)
#
#     @classmethod
#     def generate_op_str(cls, op, args, **kwargs):
#         """Generates the function string, call signature etc.
#         """
#
#         # collect operator arguments
#         results = {'value': None, 'args': [], 'arg_names': [], 'is_constant': False, 'input_ops': []}
#
#         if op == 'no_op':
#             if hasattr(args[0], 'short_name'):
#                 val = args[0].short_name
#                 arg = [args[0]]
#                 name = [args[0].short_name]
#             else:
#                 val = str(args[0])
#                 arg = []
#                 name = []
#             results['value'] = val
#             results['args'] = arg
#             results['arg_names'] = name
#             return results
#
#         results, results_args, results_arg_names, n_vars = cls._process_args(args, results)
#
#         # setup operator call
#         #####################
#
#         eval_gen = CodeGen()
#
#         # check whether operator is a simple mathematical symbol or involves a function call
#         if op in "+-**/*^%<>==>=<=" and len(args) == 2:
#             arg1 = cls._arg_to_str(op, args[0], results_args[0], results_arg_names[0])
#             arg2 = cls._arg_to_str(op, args[1], results_args[1], results_arg_names[1])
#             eval_gen.add_code_line(f"{arg1} {op} {arg2}")
#             func_call = False
#         else:
#             func_call = True
#             eval_gen.add_code_line(f"{op}(")
#
#         # add operator arguments
#         for i, (key, arg) in enumerate(zip(results_arg_names, results_args)):
#             if key != '__no_name__':
#                 if type(arg) is str:
#                     if func_call:
#                         if arg == "(":
#                             eval_gen.add_code_line(f"{arg}")
#                         else:
#                             eval_gen.add_code_line(f"{arg},")
#                     idx = cls._index(results['args'], arg)
#                     results['args'].pop(idx)
#                     results['arg_names'].pop(idx)
#                 elif type(arg) is dict and func_call:
#                     eval_gen.add_code_line(f"{arg['value']},")
#                 elif func_call:
#                     eval_gen.add_code_line(f"{key},")
#             else:
#                 if func_call:
#                     eval_gen.add_code_line(f"{arg},")
#                 if key in results['arg_names']:
#                     idx = cls._index(results['arg_names'], key)
#                     results['args'].pop(idx)
#                     results['arg_names'].pop(idx)
#
#         # add function end
#         if func_call:
#             eval_gen.code[-1] = eval_gen.code[-1][:-1]
#             eval_gen.add_code_line(")")
#         results['value'] = eval_gen.generate()
#
#         # check whether operation arguments contain merely constants
#         if n_vars == 0:
#             results['is_constant'] = True
#
#         return results
#
#     @classmethod
#     def _process_args(cls, args, results, constants_to_num=True):
#         """Parses arguments to function into argument names that are added to the function call and argument values
#         that can used to set up the return line of the function.
#         """
#
#         results_args = []
#         results_arg_names = []
#         n_vars = 0
#
#         for idx, arg in enumerate(args):
#
#             if PyRatesOp.__subclasscheck__(type(arg)):
#
#                 # extract arguments from other operations and nest them into this operation
#                 results['input_ops'].append(arg.name)
#                 pop_indices = []
#                 for arg_tmp in arg.arg_names:
#                     if arg_tmp in results['arg_names']:
#                         pop_indices.append(arg.arg_names.index(arg_tmp))
#                 new_args = arg.args.copy()
#                 new_arg_names = arg.arg_names.copy()
#                 for i, pop_idx in enumerate(pop_indices):
#                     new_args.pop(pop_idx - i)
#                     new_arg_names.pop(pop_idx - i)
#                 results_args.append({'args': new_args, 'value': arg.lhs if 'assign' in arg.short_name else arg.value,
#                                      'arg_names': new_arg_names})
#                 results_arg_names.append(arg.short_name)
#                 results['args'] += new_args
#                 results['arg_names'] += new_arg_names
#                 n_vars += 1
#
#             elif cls.var_class.__subclasscheck__(type(arg)):
#
#                 # parse PyRates variable into the function
#                 if arg.vtype is 'constant' and not tuple(arg.shape) and constants_to_num:
#                     arg_name = '__no_name__'
#                     arg_value = arg.numpy()  #if hasattr(arg, 'numpy') else arg
#                 else:
#                     arg_name = arg.short_name
#                     arg_value = arg
#                     n_vars += 1
#                 results_args.append(arg_value)
#                 results_arg_names.append(arg_name)
#                 results['args'].append(arg_value)
#                 results['arg_names'].append(arg_name)
#
#             elif type(arg) is tuple or type(arg) is list:
#
#                 # parse list of arguments
#                 tuple_begin = '('
#                 results_args.append(tuple_begin)
#                 results_arg_names.append(tuple_begin)
#                 results['args'].append(tuple_begin)
#                 results['arg_names'].append(tuple_begin)
#                 results, results_args_tmp, results_arg_names_tmp, n_vars_tmp = cls._process_args(arg, results)
#                 results_args += results_args_tmp
#                 results_arg_names += results_arg_names_tmp
#                 n_vars += n_vars_tmp
#                 tuple_end = ')'
#                 results_args.append(tuple_end)
#                 results_arg_names.append(tuple_end)
#                 results['args'].append(tuple_end)
#                 results['arg_names'].append(tuple_end)
#
#             else:
#
#                 # parse strings and constant arguments into the function
#                 arg_name = "__no_name__"
#                 results_args.append(arg)
#                 results_arg_names.append(arg_name)
#                 results['args'].append(arg)
#                 results['arg_names'].append(arg_name)
#
#         return results, results_args, results_arg_names, n_vars
#
#     @staticmethod
#     def _get_unique_args(arg_names, args):
#         arg_names_unique = np.unique(arg_names)
#         for name in arg_names_unique:
#             n = arg_names.count(name)
#             while n > 1:
#                 i0 = arg_names.index(name)+1
#                 i1 = arg_names[i0:].index(name)
#                 arg_names.pop(i0+i1)
#                 args.pop(i0+i1)
#                 n -= 1
#         return arg_names, args
#
#     @staticmethod
#     def _check_numerics(vals: BaseVar, name: str):
#         """Checks whether function evaluation leads to any NaNs or infinite values.
#         """
#         check = []
#         try:
#             if hasattr(vals, 'shape'):
#                 vals = vals.numpy().flatten() if hasattr(vals, 'numpy') else vals.flatten()
#             else:
#                 vals = [vals]
#             for val in vals:
#                 check.append(np.isnan(val) or np.isneginf(val))
#         except TypeError:
#             check.append(np.isnan(vals) or np.isneginf(vals))
#         if any(check):
#             raise ValueError(f'Result of operation ({name}) contains NaNs or infinite values.')
#
#     @staticmethod
#     def _get_arg(arg: Any, arg_reduced: Any, key: str) -> str:
#         if key == '__no_name__':
#             return f"{arg_reduced}" if arg_reduced > 0 else f"({arg_reduced})"
#         elif PyRatesOp.__subclasscheck__(type(arg)):
#             return f"{arg.value}"
#         else:
#             return key
#
#     @classmethod
#     def _arg_to_str(cls, op: str, arg, arg_reduced, key: str) -> str:
#         op_orders = [['+', '-'],
#                      ['*'],
#                      ['/', '%', '>', '<', '==', '<=', '>=', '!='],
#                      ['^', '**']]
#         op_order = 0
#         for i, ops in enumerate(op_orders):
#             if op in ops:
#                 op_order = i+1
#                 break
#         arg_order = 5
#         if hasattr(arg, 'op'):
#             for i, ops in enumerate(op_orders):
#                 if arg.op in ops:
#                     arg_order = i+1
#                     break
#         arg_str = cls._get_arg(arg, arg_reduced, key)
#         if (op_order > arg_order or (op_order == arg_order and op_order >= 3)) and \
#                 not (arg_str[0] == "(" and arg_str[-1] == ")"):
#             return f"({arg_str})"
#         return arg_str
#
#     @staticmethod
#     def _deepcopy(x):
#         return deepcopy(x)
#
#     @staticmethod
#     def _index(x, y):
#         return x.index(y)
#
#
# class PyRatesAssignOp(PyRatesOp):
#     """Sub-class for assign operations. Typically, an update (2. entry in args) is assigned to a PyRates variable
#     (1. entry in args). An index can be provided as third argument to target certain entries of the variable.
#     """
#
#     def __init__(self, op: str, short_name: str, name, *args, **kwargs) -> None:
#         super().__init__(op, short_name, name, *args, **kwargs)
#         self.lhs = self._op_dict.pop('lhs')
#         self.rhs = self._op_dict.pop('rhs')
#
#     def _generate_func(self):
#         func_dict = {}
#         func = CodeGen()
#         func.add_code_line(f"def {self.short_name}(")
#         for arg in self._op_dict['arg_names']:
#             func.add_code_line(f"{arg},")
#         func.code[-1] = func.code[-1][:-1]
#         func.add_code_line("):")
#         func.add_linebreak()
#         func.add_indent()
#         func.add_code_line(f"{self._op_dict['value']}")
#         func.add_linebreak()
#         func.add_code_line(f"return {self._op_dict['lhs']}")
#         exec(func.generate(), globals(), func_dict)
#         return func_dict
#
#     @classmethod
#     def generate_op_str(cls, op, args, **kwargs):
#         """Generates the assign function string, call signature etc.
#         """
#
#         # initialization
#         results = {'value': None, 'args': [], 'arg_names': [], 'is_constant': False, 'input_ops': [], 'lhs': None}
#         results_args = []
#         results_arg_names = []
#
#         # extract variables
#         ###################
#
#         var, upd = args[0:2]
#
#         if len(args) > 2:
#             var_idx, results_args, results_arg_names = cls._extract_var_idx(op, args, results_args, results_arg_names,
#                                                                             **kwargs)
#         else:
#             var_idx = ""
#
#         # add variable to the function arguments
#         var_key = var.short_name if 'assign' not in var.short_name else '__no_name__'
#         var_pos = len(results_args)
#         results_args.append(var if 'assign' not in var.short_name else {'value': var.lhs})
#         results_arg_names.append(var_key)
#
#         # add variable update to the function arguments
#         if hasattr(upd, 'vtype') and upd.vtype == 'constant' and not upd.shape:
#             upd_key = '__no_name__'
#             if not PyRatesOp.__subclasscheck__(type(upd)):
#                 upd = {'value': upd}
#         else:
#             upd_key = upd.short_name
#         upd_pos = len(results_args)
#         results_args.append(upd)
#         results_arg_names.append(upd_key)
#
#         # setup function head
#         #####################
#
#         # add arguments
#         for idx, (key, arg) in enumerate(zip(results_arg_names, results_args)):
#             if PyRatesOp.__subclasscheck__(type(arg)):
#                 pop_indices = []
#                 for arg_tmp in arg.arg_names:
#                     if arg_tmp in results['arg_names']:
#                         pop_indices.append(arg.arg_names.index(arg_tmp))
#                 new_args = arg.args.copy()
#                 new_arg_names = arg.arg_names.copy()
#                 for i, pop_idx in enumerate(pop_indices):
#                     new_args.pop(pop_idx - i)
#                     new_arg_names.pop(pop_idx - i)
#                 results_args[idx] = {'args': new_args, 'value': arg.value, 'arg_names': new_arg_names}
#                 results['input_ops'].append(arg.name)
#                 results['args'] += new_args
#                 results['arg_names'] += new_arg_names
#             elif key != '__no_name__':
#                 results['args'].append(arg)
#                 results['arg_names'].append(key)
#
#         # assign update to variable and return variable
#         ###############################################
#
#         var_str = results_args[var_pos]['value'] if type(results_args[var_pos]) is dict else var_key
#         upd_str = results_args[upd_pos]['value'] if type(results_args[upd_pos]) is dict else upd_key
#         if op in ["=", "+=", "-=", "*=", "/="]:
#             assign_line = f"{var_str}{var_idx} {op} {upd_str}"
#         elif "scatter" in op:
#             scatter_into_first_dim = args[-1]
#             if len(args) > 3 and scatter_into_first_dim:
#                 assign_line = f"{var_str}.{op}([{var_idx}], [{upd_str}])"
#             else:
#                 assign_line = f"{var_str}.{op}({var_idx}{upd_str})"
#         else:
#             assign_line = f"{var_str}{var_idx}.{op}({upd_str})"
#         results['value'] = assign_line
#         results['lhs'] = var_str
#         results['rhs'] = upd_str
#
#         return results
#
#     @classmethod
#     def _extract_var_idx(cls, op, args, results_args, results_arg_names, idx_l='[', idx_r=']'):
#
#         if hasattr(args[2], 'vtype'):
#
#             # for indexing via PyRates variables
#             key = args[2].short_name
#             if hasattr(args[2], 'value'):
#                 var_idx = f"{idx_l}{args[2].value}{idx_r}"
#             else:
#                 var_idx = f"{idx_l}{key}{idx_r}"
#
#         elif type(args[2]) is str and len(args) > 3:
#
#             # for indexing via string-based indices
#             key = args[3].short_name
#             var_idx = f"{idx_l}{args[2]}{idx_r}"
#             var_idx = var_idx.replace(args[3].short_name, key)
#
#         else:
#
#             # for indexing via integers
#             var_idx = f"{idx_l}{args[2]}{idx_r}"
#             key = "__no_name__"
#
#         if type(args[2]) is str and len(args) > 3:
#             results_args.append(args[3])
#         else:
#             results_args.append(args[2])
#         results_arg_names.append(key)
#
#         return var_idx, results_args, results_arg_names
#
#
# class PyRatesIndexOp(PyRatesOp):
#     """Sub-class for indexing operations. Typically, an index is provided (2. entry in args) indicating which values to
#      extract from a variable (1. entry in args).
#     """
#
#     @classmethod
#     def generate_op_str(cls, op, args, idx_l='[', idx_r=']'):
#         """Generates the index function string, call signature etc.
#         """
#
#         # initialization
#         results = {'value': None, 'args': [], 'arg_names': [], 'is_constant': False, 'shape': (), 'dtype': 'float32',
#                    'input_ops': []}
#
#         # extract variable
#         ##################
#
#         var_tmp = args[0]
#         n_vars = 0
#
#         if PyRatesOp.__subclasscheck__(type(var_tmp)):
#
#             # nest pyrates operations and their arguments into the indexing operation
#             var = var_tmp.lhs if hasattr(var_tmp, 'lhs') else var_tmp.value
#             pop_indices = []
#             for arg_tmp in var_tmp.arg_names:
#                 if arg_tmp in results['arg_names']:
#                     pop_indices.append(var_tmp.arg_names.index(arg_tmp))
#             new_args = var_tmp.args.copy()
#             new_arg_names = var_tmp.arg_names.copy()
#             for i, pop_idx in enumerate(pop_indices):
#                 new_args.pop(pop_idx - i)
#                 new_arg_names.pop(pop_idx - i)
#             results['input_ops'].append(var_tmp.name)
#             results['args'] += new_args
#             results['arg_names'] += new_arg_names
#             n_vars += 1
#
#         elif cls.var_class.__subclasscheck__(type(var_tmp)):
#
#             # parse pyrates variables into the indexing operation
#             if var_tmp.vtype != 'constant' or tuple(var_tmp.shape):
#                 var = var_tmp.short_name
#                 results['args'].append(var_tmp)
#                 results['arg_names'].append(var)
#                 n_vars += 1
#             else:
#                 var = var_tmp.numpy()
#
#         else:
#
#             raise ValueError(f'Variable type for indexing not understood: {var_tmp}. '
#                              f'Please consider another form of variable declaration or indexing.')
#
#         # extract idx
#         #############
#
#         idx = args[1]
#
#         if PyRatesOp.__subclasscheck__(type(idx)):
#
#             # nest pyrates operations and their arguments into the indexing operation
#             var_idx = f"{idx_l}{idx.lhs if hasattr(idx, 'lhs') else idx.value}{idx_r}"
#             pop_indices = []
#             for arg_tmp in idx.arg_names:
#                 if arg_tmp in results['arg_names']:
#                     pop_indices.append(idx.arg_names.index(arg_tmp))
#             new_args = idx.args.copy()
#             new_arg_names = idx.arg_names.copy()
#             for i, pop_idx in enumerate(pop_indices):
#                 new_args.pop(pop_idx - i)
#                 new_arg_names.pop(pop_idx - i)
#             results['input_ops'].append(idx.name)
#             results['args'] += new_args
#             results['arg_names'] += new_arg_names
#             n_vars += 1
#
#         elif cls.var_class.__subclasscheck__(type(idx)):
#
#             # parse pyrates variables into the indexing operation
#             key = idx.short_name
#             if idx.vtype != 'constant' or tuple(idx.shape):
#                 var_idx = f"{idx_l}{key}{idx_r}"
#                 results['args'].append(idx)
#                 results['arg_names'].append(key)
#                 n_vars += 1
#             else:
#                 var_idx = f"{idx_l}{idx.numpy()}{idx_r}"
#
#         elif type(idx) is str or "int" in str(type(idx)) or "float" in str(type(idx)):
#
#             # parse constant numpy-like indices into the indexing operation
#             pop_indices = []
#             for i, arg in enumerate(args[2:]):
#                 if hasattr(arg, 'short_name') and arg.short_name in idx:
#                     if hasattr(arg, 'vtype') and arg.vtype == 'constant' and sum(arg.shape) < 2:
#                         idx = replace(idx, arg.short_name, f"{int(arg)}")
#                         pop_indices.append(i+2)
#                     elif hasattr(arg, 'value'):
#                         idx = replace(idx, arg.short_name, arg.value)
#                         pop_indices.append(i+2)
#                         results['args'] += arg.args
#                         results['arg_names'] += arg.arg_names
#             var_idx = f"{idx_l}{idx}{idx_r}"
#             args = list(args)
#             for idx in pop_indices[::-1]:
#                 args.pop(idx)
#             args = tuple(args)
#
#         elif type(idx) in (list, tuple):
#
#             # parse list of constant indices into the operation
#             var_idx = f"{list(idx)}"
#
#         else:
#             raise ValueError(f'Index type not understood: {idx}. Please consider another form of variable indexing.')
#
#         # setup function head
#         #####################
#
#         # remaining arguments
#         results, results_args, results_arg_names, n_vars_tmp = cls._process_args(args[2:], results)
#         n_vars += n_vars_tmp
#
#         # apply index to variable and return variable
#         #############################################
#
#         # generate op
#         results['value'] = f"{var}{var_idx}"
#
#         # check whether any state variables are part of the indexing operation or not
#         if n_vars == 0:
#             results['is_constant'] = True
#
#         return results
#
#
# class NumpyBackend(object):
#     """Wrapper to numpy. This class provides an interface to all numpy functionalities that may be accessed via pyrates.
#     All numpy variables and operations will be stored in a layered compute graph that can be executed to evaluate the
#     (dynamic) behavior of the graph.
#
#     Parameters
#     ----------
#     ops
#         Additional operations this backend instance can perform, defined as key-value pairs. The key can then be used in
#         every equation parsed into this backend. The value is a dictionary again, with two keys:
#             1) name - the name of the function/operation (used during code generation)
#             2) call - the call signature of the function including import abbreviations (i.e. `np.add` for numpy's add
#             function)
#     dtypes
#         Additional data-types this backend instance can use, defined as key-value pairs.
#     name
#         Name of the backend instance. Used during code generation to create a recognizable file structure.
#     float_default_type
#         Default float precision. If no data-type is indicated for a particular variable, this will be used.
#     imports
#         Can be used to pass additional import statements that are needed for code generation of the custom functions
#         provided via `ops`. Will be added to the top of each generated code file.
#     build_dir
#         Directory in which pyrates builds will be stored.
#
#     """
#
#     idx_l, idx_r = "[", "]"
#     idx_start = 0
#
#     def __init__(self,
#                  ops: Optional[Dict[str, str]] = None,
#                  dtypes: Optional[Dict[str, object]] = None,
#                  name: str = 'net_0',
#                  float_default_type: str = 'float32',
#                  imports: Optional[List[str]] = None,
#                  build_dir: str = None,
#                  ) -> None:
#         """Instantiates numpy backend, i.e. a compute graph with numpy operations.
#         """
#
#         super().__init__()
#
#         # define operations and datatypes of the backend
#         ################################################
#
#         # base math operations
#         self.ops = {"+": {'func': np.add, 'str': "np.add"},
#                     "-": {'func': np.subtract, 'str': "np.subtract"},
#                     "*": {'func': np.multiply, 'str': "np.multiply"},
#                     "/": {'func': np.divide, 'str': "np.divide"},
#                     "%": {'func': np.mod, 'str': "np.mod"},
#                     "^": {'func': np.power, 'str': "np.power"},
#                     "**": {'func': np.float_power, 'str': "np.float_power"},
#                     "@": {'func': np.dot, 'str': "np.dot"},
#                     ".T": {'func': np.transpose, 'str': "np.transpose"},
#                     ".I": {'func': np.invert, 'str': "np.invert"},
#                     ">": {'func': np.greater, 'str': "np.greater"},
#                     "<": {'func': np.less, 'str': "np.less"},
#                     "==": {'func': np.equal, 'str': "np.equal"},
#                     "!=": {'func': np.not_equal, 'str': "np.not_equal"},
#                     ">=": {'func': np.greater_equal, 'str': "np.greater_equal"},
#                     "<=": {'func': np.less_equal, 'str': "np.less_equal"},
#                     "neg": {'func': neg_one, 'str': "neg_one"},
#                     "sin": {'func': np.sin, 'str': "np.sin"},
#                     "cos": {'func': np.cos, 'str': "np.cos"},
#                     "tan": {'func': np.tan, 'str': "np.tan"},
#                     "atan": {'func': np.arctan, 'str': "np.arctan"},
#                     "abs": {'func': np.abs, 'str': "np.abs"},
#                     "sqrt": {'func': np.sqrt, 'str': "np.sqrt"},
#                     "sq": {'func': np.square, 'str': "np.square"},
#                     "exp": {'func': np.exp, 'str': "np.exp"},
#                     "max": {'func': np.maximum, 'str': "np.maximum"},
#                     "min": {'func': np.minimum, 'str': "np.minimum"},
#                     "argmax": {'func': np.argmax, 'str': "np.argmax"},
#                     "argmin": {'func': np.argmin, 'str': "np.argmin"},
#                     "round": {'func': np.round, 'str': "np.round"},
#                     "sum": {'func': np.sum, 'str': "np.sum"},
#                     "mean": {'func': np.mean, 'str': "np.mean"},
#                     "matmul": {'func': np.matmul, 'str': "np.matmul"},
#                     "concat": {'func': np.concatenate, 'str': "np.concatenate"},
#                     "reshape": {'func': np.reshape, 'str': "np.reshape"},
#                     "append": {'func': np.append, 'str': "np.append"},
#                     "shape": {'func': np.shape, 'str': "np.shape"},
#                     "dtype": {'func': np.dtype, 'str': "np.dtype"},
#                     'squeeze': {'func': np.squeeze, 'str': "np.squeeze"},
#                     'expand': {'func': np.expand_dims, 'str': "np.expand_dims"},
#                     "roll": {'func': np.roll, 'str': "np.roll"},
#                     "cast": {'func': np.asarray, 'str': "np.asarray"},
#                     "randn": {'func': np.random.randn, 'str': "np.random.randn"},
#                     "ones": {'func': np.ones, 'str': "np.ones"},
#                     "zeros": {'func': np.zeros, 'str': "np.zeros"},
#                     "range": {'func': np.arange, 'str': "np.arange"},
#                     "softmax": {'func': pr_softmax, 'str': "pr_softmax"},
#                     "sigmoid": {'func': pr_sigmoid, 'str': "pr_sigmoid"},
#                     "tanh": {'func': np.tanh, 'str': "np.tanh"},
#                     #"index": {'func': pyrates_index, 'str': "pyrates_index"},
#                     #"mask": {'func': pr_mask, 'str': "pr_mask"},
#                     #"group": {'func': pr_group, 'str': "pr_group"},
#                     "no_op": {'func': pr_identity, 'str': "pr_identity"},
#                     "interp": {'func': pr_interp, 'str': "pr_interp"},
#                     "interpolate_1d": {'func': pr_interp_1d, 'str': "pr_interp_1d"},
#                     "interpolate_nd": {'func': pr_interp_nd, 'str': "pr_interp_nd"},
#                     }
#         if ops:
#             self.ops.update(ops)
#
#         # base data-types
#         self.dtypes = {"float16": np.float16,
#                        "float32": np.float32,
#                        "float64": np.float64,
#                        "int16": np.int16,
#                        "int32": np.int32,
#                        "int64": np.int64,
#                        "uint16": np.uint16,
#                        "uint32": np.uint32,
#                        "uint64": np.uint64,
#                        "complex64": np.complex64,
#                        "complex128": np.complex128,
#                        "bool": np.bool
#                        }
#         if dtypes:
#             self.dtypes.update(dtypes)
#
#         # further attributes
#         self.vars = dict()
#         self.state_vars = []
#         self.lhs_vars = []
#         self.var_counter = {}
#         self.op_counter = {}
#         self.op_indices = {}
#         self._float_def = self.dtypes[float_default_type]
#         self.name = name
#         self._input_layer_added = False
#         self._imports = ["import numpy as np", "from pyrates.backend.funcs import *"]
#         if imports:
#             for imp in imports:
#                 if imp not in self._imports:
#                     self._imports.append(imp)
#         self._input_names = []
#         self._type = 'numpy'
#
#         # create build dir
#         self._orig_dir = os.getcwd()
#         if build_dir:
#             os.makedirs(build_dir, exist_ok=True)
#         dir_name = f"{build_dir}/pyrates_build" if build_dir else "pyrates_build"
#         try:
#             os.mkdir(dir_name)
#         except FileExistsError:
#             pass
#         except FileNotFoundError as e:
#             # for debugging
#             raise e
#         self._build_dir = f"{dir_name}/{self.name}"
#         try:
#             os.mkdir(self._build_dir)
#         except FileExistsError:
#             rmtree(self._build_dir)
#             os.mkdir(self._build_dir)
#         sys.path.append(self._build_dir)
#
#     def run(self,
#             T: float,
#             dt: float,
#             outputs: Optional[dict] = None,
#             dts: Optional[int] = None,
#             solver: str = 'euler',
#             out_dir: Optional[str] = None,
#             profile: bool = False,
#             verbose: bool = True,
#             **kwargs
#             ) -> tuple:
#         """Executes all operations in the backend graph for a given number of steps.
#
#         Parameters
#         ----------
#         T
#             Simulation time.
#         dt
#             Simulation step size.
#         outputs
#             Variables in the graph to store the history from.
#         dts
#             Sampling step-size.
#         solver
#             Type of the numerical solver to use.
#         out_dir
#             Directory to write the session log into.
#         profile
#             If true, the total graph execution time will be printed and returned.
#         verbose
#             If true, updates about the simulation process will be displayed in the terminal.
#
#         Returns
#         -------
#         Union[Tuple[Dict[str, tf.Variable]], Tuple[dict, float]]
#             If `profile` was requested, a tuple is returned that contains
#                 1) the results dictionary
#                 2) the simulation time.
#             If not, a tuple containing only the results dictionary is returned which contains a numpy array with results
#              for each output key that was provided via `outputs`. This allows to call the run functioon like this:
#              `output, *time = run(*args, **kwargs)`
#
#         """
#
#         # preparations
#         ##############
#
#         if not dts:
#             dts = dt
#
#         # initialize session log
#         if out_dir:
#             # TODO : implement log files
#             pass
#
#         # initialize profiler
#         if profile:
#             t0 = time.time()
#
#         # initialize time
#         discrete_time = kwargs.pop('discrete_time', True)
#         if discrete_time:
#             t = self.add_var('state_var', name='t', value=0, dtype='int32', shape=())
#         else:
#             t = self.add_var('state_var', name='t', value=0.0, dtype=self._float_def, shape=())
#
#         # graph execution
#         #################
#
#         if verbose:
#             print("\t(2) Generating the model run function...")
#
#         # map layers that need to be executed to compiled network structure
#         decorator = kwargs.pop('decorator', None)
#         decorator_kwargs = kwargs.pop('decorator_kwargs', {})
#         rhs_func, args, state_vars, var_map = self.compile(self._build_dir, decorator=decorator, **decorator_kwargs)
#
#         if verbose:
#             print("\t\t...finished.")
#
#         # create output indices
#         output_indices = []
#         if outputs:
#             for out_key, out_vars in outputs.items():
#                 for n, (idx, _) in enumerate(out_vars):
#                     output_indices.append(
#                         [i - self.idx_start for i in idx] if type(idx) is list else idx - self.idx_start)
#                     outputs[out_key][n][0] = len(output_indices) - 1
#         else:
#             n = 0
#             for i in range(len(self.state_vars)):
#                 out_key = self.state_vars[i]
#                 if len(self.vars[out_key].shape) > 0:
#                     for j in range(self.vars[out_key].shape[0]):
#                         new_key = f"{out_key}_{j}"
#                         output_indices.append(n)
#                         outputs[new_key] = [(n, [new_key])]
#                         n += 1
#                 else:
#                     output_indices.append(n)
#                     outputs[out_key] = [(n, [out_key])]
#                     n += 1
#
#         # simulate backend behavior for each time-step
#         func_args = self._process_func_args(args, var_map, dt)
#
#         if verbose:
#             print("\t (3) Running the simulation...")
#
#         times, results = self._solve(rhs_func=rhs_func, func_args=func_args, T=T, dt=dt, dts=dts, t=t, solver=solver,
#                                      output_indices=output_indices, **kwargs)
#
#         if verbose:
#             print("\t\t finished.\n")
#
#         # output storage and clean-up
#         #############################
#
#         # store output variables in output dictionary
#         for i, (out_key, out_vars) in enumerate(outputs.items()):
#             node_col = []
#             for _, node_keys in out_vars:
#                 node_col += node_keys
#             outputs[out_key] = (np.asarray(results[i]), node_col)
#
#         # store profiling results
#         if profile:
#             sim_time = time.time() - t0
#             return outputs, times, sim_time
#
#         return outputs, times
#
#     def add_var(self,
#                 vtype: str,
#                 name: Optional[str] = None,
#                 value: Optional[Any] = None,
#                 shape: Optional[Union[tuple, list, np.shape]] = None,
#                 dtype: Optional[Union[str, np.dtype]] = None,
#                 **kwargs
#                 ) -> BaseVar:
#         """Adds a variable to the backend.
#
#         Parameters
#         ----------
#         vtype
#             Variable type. Can be
#                 - `state_var` for variables that can change over time.
#                 - `constant` for non-changing variables.
#         name
#             Name of the variable.
#         value
#             Value of the variable. Not needed for placeholders.
#         shape
#             Shape of the variable.
#         dtype
#             Datatype of the variable.
#         kwargs
#             Additional keyword arguments.
#
#         Returns
#         -------
#         PyRatesVar
#             Handle for the numpy variable.
#
#         """
#
#         # extract variable scope
#         scope = kwargs.pop('scope', None)
#         if scope:
#             name = f'{scope}/{name}'
#
#         if name in self.vars:
#             return self.get_var(name)
#
#         # create variable
#         var, name = self._create_var(vtype=vtype, dtype=dtype, shape=shape, value=value, name=name,
#                                      squeeze=kwargs.pop('squeeze', True))
#
#         # ensure uniqueness of variable names
#         if var.short_name in self.var_counter and name not in self.vars:
#             rename = True if var.short_name not in ["y",  "y_delta", "t",  "times"] else False
#             if var.vtype == 'constant' and rename:
#                 for var_tmp in self.vars.values():
#                     rename = not self._compare_vars(var, var_tmp)
#                     if rename:
#                         break
#             if rename:
#                 name_old = var.short_name
#                 name_new = f"{name_old}_{self.var_counter[name_old]}"
#                 var.short_name = name_new
#                 self.var_counter[name_old] += 1
#         else:
#             self.var_counter[var.short_name] = 0
#
#         self.vars[name] = var
#         return var
#
#     def remove_var(self, name: str) -> Union[BaseVar, None]:
#         """Removes variable from backend and returns it.
#
#         Parameters
#         ----------
#         name
#             Variable name.
#
#         Returns
#         -------
#         BaseVar
#
#         """
#         if name in self.var_counter:
#             self.var_counter[name] -= 1
#             if self.var_counter[name] < 0:
#                 self.var_counter.pop(name)
#         if name in self.vars:
#             return self.vars.pop(name)
#         else:
#             return None
#
#     def add_op(self,
#                op_name: str,
#                *args,
#                **kwargs
#                ) -> Union[PyRatesOp, BaseVar]:
#         """Add operation to the backend.
#
#         Parameters
#         ----------
#         op_name
#             Key of the operation. Needs to be a key of `backend.ops`.
#         args
#             Positional arguments to be passed to the operation.
#         kwargs
#             Keyword arguments to be passed to the operation, except for scope and dependencies, which are
#             extracted before.
#
#         Returns
#         -------
#         Union[PyRatesOp, PyRatesVar]
#             Handle for the lambda-numpy function
#
#         """
#
#         # process input arguments
#         #########################
#
#         # extract scope
#         scope = kwargs.pop('scope', None)
#
#         # generate operator identifier
#         name = kwargs.pop('name', None)
#         if name and scope:
#             name = f'{scope}/{name}'
#         elif scope:
#             name = f'{scope}/assign' if '=' in op_name else f"{scope}/{self.ops[op_name]['func']}"
#         else:
#             name = f'assign' if '=' in op_name else f"{self.ops[op_name]['func']}"
#         if name in self.op_counter:
#             name_old = name
#             name = f"{name}_{self.op_counter[name]}"
#             self.op_counter[name_old] += 1
#         else:
#             self.op_counter[name] = 0
#
#         # extract and process graph dependencies
#         dependencies = kwargs.pop('dependencies', [])
#         if dependencies:
#             found = False
#             for dep in dependencies:
#                 if dep.name == name:
#                     found = True
#                     break
#             if found:
#                 self.add_layer()
#
#         # create operation
#         ##################
#
#         try:
#
#             # try generating the operator
#             op = self._create_op(op_name, name, *args)
#
#             # make sure that the output shape of the operation matches the expectations
#             in_shape = []
#             expand_ops = ('@', 'index', 'concat', 'expand', 'stack', 'group', 'asarray', 'interpolate', 'interpolate_nd'
#                           )
#             if op_name not in expand_ops:
#                 for arg in args:
#                     if hasattr(arg, 'shape'):
#                         in_shape.append(sum(tuple(arg.shape)))
#                     elif type(arg) in (tuple, list):
#                         if hasattr(arg[0], 'shape'):
#                             in_shape.append(sum(tuple(arg[0].shape)))
#                         else:
#                             in_shape.append(sum(arg))
#                     else:
#                         in_shape.append(1)
#                 if hasattr(op, 'shape') and (sum(tuple(op.shape)) > max(in_shape)):
#                     arg1, arg2 = self.broadcast(args[0], args[1])
#                     op = self._create_op(op_name, name, arg1, arg2, *args[2:])
#
#         except Exception:
#
#             if type(args[0]) in (tuple, list) and len(args[0]) > 1:
#
#                 # broadcast entries of argument tuples
#                 args_tmp = self.broadcast(args[0][0], args[0][1])
#                 args_tmp = (args_tmp[0], args_tmp[1]) + tuple(args[0][2:])
#                 args_new = args_tmp + args[1:]
#
#             elif '=' in op_name and len(args) > 2:
#
#                 # for assign operations with index, apply index temporarily and broadcast rhs to match indexed lhs
#                 arg1_tmp = self._create_op('index', 'test_idx', args[0], args[2])
#                 args_tmp = self.broadcast(arg1_tmp, args[1])
#                 args_new = (args[0], args_tmp[1], args[2])
#
#             else:
#
#                 # broadcast leading 2 arguments to operation
#                 args_tmp = self.broadcast(args[0], args[1])
#                 args_new = args_tmp + args[2:]
#
#             # try to generate operation again
#             op = self._create_op(op_name, name, *args_new)
#
#         # remove op inputs from layers (need not be evaluated anymore)
#         for op_name_tmp in op.input_ops:
#             if 'assign' not in op_name_tmp:
#                 idx_1, idx_2 = self.op_indices[op_name_tmp]
#                 self._set_op(None, idx_1, idx_2)
#
#         # for constant ops, add a constant to the graph, containing the op evaluation
#         if op.is_constant and op_name != 'no_op':
#             new_var = op.numpy()
#             if hasattr(new_var, 'shape'):
#                 name = f'{name}_evaluated'
#                 return self.add_var(vtype='constant', name=name, value=new_var)
#             else:
#                 return new_var
#
#         # add op to the graph
#         self._add_op(op)
#         self.op_indices[op.name] = (self.layer, len(self.get_current_layer())-1)
#         return op
#
#     def add_layer(self, to_beginning=False) -> None:
#         """Adds a new layer to the backend and sets cursor to that layer.
#
#         Parameters
#         ----------
#         to_beginning
#             If true, layer is appended to the beginning of the layer stack instead of the end
#
#         Returns
#         -------
#         None
#
#         """
#
#         if to_beginning:
#             self.layers = [[]] + self.layers
#             self._base_layer += 1
#             self.layer = -self._base_layer
#         else:
#             self.layer = len(self.layers) - self._base_layer
#             self.layers.append([])
#
#     def add_output_layer(self, outputs, sampling_steps, out_shapes) -> dict:
#         """
#
#         Parameters
#         ----------
#         outputs
#         sampling_steps
#         out_shapes
#
#         Returns
#         -------
#
#         """
#
#         output_col = {}
#
#         # add output storage layer to the graph
#         if self._output_layer_added:
#
#             out_idx = self.get_var("output_collection/out_var_idx")
#
#             # jump to output layer
#             self.top_layer()
#
#         else:
#
#             # add output layer
#             self.add_layer()
#             self._output_layer_added = True
#
#             # create counting index for collector variables
#             out_idx = self.add_var(vtype='state_var', name='out_var_idx', dtype='int32', shape=(1,), value=0,
#                                    scope="output_collection")
#
#             # create increment operator for counting index
#             self.next_layer()
#             self.add_op('+=', out_idx, np.ones((1,), dtype='int32'), scope="output_collection")
#             self.previous_layer()
#
#         # add collector variables to the graph
#         for i, (var_col) in enumerate(outputs):
#
#             shape = (sampling_steps + 1, len(var_col)) + out_shapes[i]
#             key = f"output_col_{i}"
#             output_col[key] = self.add_var(vtype='state_var', name=f"out_col_{i}", scope="output_collection",
#                                            value=np.zeros(shape, dtype=self._float_def))
#             var_stack = self.stack_vars(var_col)
#
#             # add collect operation to the graph
#             self.add_op('=', output_col[key], var_stack, out_idx, scope="output_collection")
#
#         return output_col
#
#     def next_layer(self) -> None:
#         """Jump to next layer in stack. If we are already at end of layer stack, add new layer to the stack and jump to
#         that.
#         """
#         if self._base_layer+self.layer == len(self.layers)-1:
#             self.add_layer()
#         else:
#             self.layer += 1
#
#     def previous_layer(self) -> None:
#         """Jump to previous layer in stack. If we are already at beginning of layer stack, add new layer to beginning of
#         the stack and jump to that.
#         """
#         if self.layer == -self._base_layer:
#             self.add_layer(to_beginning=True)
#         else:
#             self.layer -= 1
#
#     def goto_layer(self, idx: int) -> None:
#         """Jump to layer indicated by index.
#
#         Parameters
#         ----------
#         idx
#             Position of layer to jump towards.
#
#
#         Returns
#         -------
#         None
#
#         """
#         self.layer = idx
#
#     def remove_layer(self, idx) -> None:
#         """Removes layer at index from stack.
#
#         Parameters
#         ----------
#         idx
#             Position of layer in graph.
#
#         """
#         self.layers.pop(self._base_layer + idx)
#         if idx <= self._base_layer:
#             self._base_layer -= 1
#
#     def top_layer(self) -> int:
#         """Jump to top layer of the stack.
#         """
#         self.layer = len(self.layers)-1-self._base_layer
#         return self.layer
#
#     def bottom_layer(self) -> int:
#         """Jump to bottom layer of the stack.
#         """
#         self.layer = -self._base_layer
#         return self.layer
#
#     def get_current_layer(self) -> list:
#         """Get operations in current layer.
#         """
#         return self.layers[self._base_layer + self.layer]
#
#     def clear(self) -> None:
#         """Removes all layers, variables and operations from graph. Deletes build directory.
#         """
#         self.vars.clear()
#         self.layers = [[]]
#         self.op_counter = 0
#         self.var_counter = 0
#         self.layer = 0
#         rmtree(f"{self._orig_dir}/{self._build_dir}")
#         if 'rhs_func' in sys.modules:
#             del sys.modules['rhs_func']
#         return self._orig_dir
#
#     def get_layer(self, idx) -> list:
#         """Retrieve layer from graph.
#
#         Parameters
#         ----------
#         idx
#             Position of layer in stack.
#
#         """
#         return self.layers[self._base_layer + idx]
#
#     def get_var(self, name: str) -> Union[BaseVar, PyRatesOp]:
#         """Retrieve variable from graph.
#
#         Parameters
#         ----------
#         name
#             Identifier of the variable.
#
#         Returns
#         -------
#         BaseVar
#             Variable from graph.
#
#         """
#         return self.vars[name]
#
#     def eval_var(self, var: str) -> np.ndarray:
#         """Get value of variable.
#
#         Parameters
#         ----------
#         var
#             Identifier of variable in graph.
#
#         Returns
#         -------
#         np.ndarray
#             Current value of the variable.
#
#         """
#         return self.vars[var].numpy()
#
#     def compile(self, build_dir: Optional[str] = None, decorator: Optional[Callable] = None, **kwargs) -> tuple:
#         """Compile the graph layers/operations. Creates python files containing the functions in each layer.
#
#         Parameters
#         ----------
#         build_dir
#             Directory in which to create the file structure for the simulation.
#         decorator
#             Decorator function that should be applied to the right-hand side evaluation function.
#         kwargs
#             decorator keyword arguments
#
#         Returns
#         -------
#         tuple
#             Contains tuples of layer run functions and their respective arguments.
#
#         """
#
#         # preparations
#         ##############
#
#         # remove empty layers and operators
#         new_layer_idx = 0
#         for layer_idx, layer in enumerate(self.layers.copy()):
#             for op in layer.copy():
#                 if op is None:
#                     layer.pop(layer.index(op))
#             if len(layer) == 0:
#                 self.layers.pop(new_layer_idx)
#             else:
#                 new_layer_idx += 1
#
#         # remove previously imported rhs_funcs from system
#         if 'rhs_func' in sys.modules:
#             del sys.modules['rhs_func']
#
#         # collect state variable and parameter vectors
#         state_vars, params, var_map = self._process_vars()
#
#         # update state variable getter operations to include the full state variable vector as first argument
#         y = self.get_var('y')
#         for key, (vtype, idx) in var_map.items():
#             var = self.get_var(key)
#             if vtype == 'state_var' and var.short_name != 'y':
#                 var.args[0] = y
#                 var.build_op(var.args)
#
#         # create rhs evaluation function
#         ################################
#
#         # set up file header
#         func_gen = CodeGen()
#         for import_line in self._imports:
#             func_gen.add_code_line(import_line)
#             func_gen.add_linebreak()
#         func_gen.add_linebreak()
#
#         # define function head
#         func_gen.add_code_line("def rhs_eval(t, y, params):")
#         func_gen.add_linebreak()
#         func_gen.add_indent()
#         func_gen.add_linebreak()
#
#         # declare constants
#         args = [None for _ in range(len(params))]
#         func_gen.add_code_line("# declare constants")
#         func_gen.add_linebreak()
#         updates, indices = [], []
#         i = 0
#         for key, (vtype, idx) in var_map.items():
#             if vtype == 'constant':
#                 var = params[idx][1]
#                 func_gen.add_code_line(f"{var.short_name} = params[{idx}]")
#                 func_gen.add_linebreak()
#                 args[idx] = var
#                 if var.short_name != "y_delta":
#                     updates.append(f"{var.short_name}")
#                     indices.append(i)
#                     i += 1
#         func_gen.add_linebreak()
#
#         # extract state variables from input vector y
#         func_gen.add_code_line("# extract state variables from input vector")
#         func_gen.add_linebreak()
#         for key, (vtype, idx) in var_map.items():
#             var = self.get_var(key)
#             if vtype == 'state_var':
#                 func_gen.add_code_line(f"{var.short_name} = {var.value}")
#                 func_gen.add_linebreak()
#         func_gen.add_linebreak()
#
#         # add equations
#         func_gen.add_code_line("# calculate right-hand side update of equation system")
#         func_gen.add_linebreak()
#         arg_updates = []
#         for i, layer in enumerate(self.layers):
#             for j, op in enumerate(layer):
#                 lhs = extract_lhs_var(op.value)
#                 find_arg = [arg == lhs for arg in updates]
#                 if any(find_arg):
#                     idx = find_arg.index(True)
#                     arg_updates.append((updates[idx], indices[idx]))
#                 func_gen.add_code_line(op.value)
#                 func_gen.add_linebreak()
#         func_gen.add_linebreak()
#
#         # update parameters where necessary
#         func_gen.add_code_line("# update system parameters")
#         func_gen.add_linebreak()
#         for upd, idx in arg_updates:
#             update_str = f"params[{idx}] = {upd}"
#             if f"    {update_str}" not in func_gen.code:
#                 func_gen.add_code_line(update_str)
#                 func_gen.add_linebreak()
#
#         # add return line
#         func_gen.add_code_line(f"return {self.vars['y_delta'].short_name}")
#         func_gen.add_linebreak()
#
#         # save rhs function to file
#         fname = f'{self._build_dir}/rhs_func'
#         with open(f'{fname}.py', 'w') as f:
#             f.writelines(func_gen.code)
#             f.close()
#
#         # import function from file
#         exec(f"from rhs_func import rhs_eval", globals())
#         rhs_eval = globals().pop('rhs_eval')
#
#         # apply function decorator
#         if decorator:
#             rhs_eval = decorator(rhs_eval, **kwargs)
#
#         return rhs_eval, args, state_vars, var_map
#
#     def broadcast(self, op1: Any, op2: Any, **kwargs) -> tuple:
#         """Tries to match the shapes of op1 and op2 such that op can be applied.
#
#         Parameters
#         ----------
#         op1
#             First argument to the operation.
#         op2
#             Second argument to the operation.
#         kwargs
#             Additional keyword arguments to be passed to the backend.
#
#         Returns
#         -------
#         tuple
#             Broadcasted op1 and op2.
#
#         """
#
#         # try to match shapes
#         if not self._compare_shapes(op1, op2):
#
#             # try adjusting op2 to match shape of op1
#             adjust_first = True if hasattr(op2, 'short_name') else False
#             op1_tmp, op2_tmp = self._match_shapes(op1, op2, adjust_second=adjust_first)
#
#             if not self._compare_shapes(op1_tmp, op2_tmp):
#
#                 # try adjusting op1 to match shape of op2
#                 op1_tmp, op2_tmp = self._match_shapes(op1_tmp, op2_tmp, adjust_second=not adjust_first)
#
#             if self._compare_shapes(op1_tmp, op2_tmp):
#                 op1, op2 = op1_tmp, op2_tmp
#
#         return op1, op2
#
#     def apply_idx(self, var: Any, idx: Any, update: Optional[Any] = None, update_type: str = None, *args) -> Any:
#         """Applies index to a variable. IF update is passed, variable is updated at positions indicated by index.
#
#         Parameters
#         ----------
#         var
#             Variable to index/update
#         idx
#             Index to variable
#         update
#             Update to variable entries
#         update_type
#             Type of lhs update (e.g. `=` or `+=`)
#
#         Returns
#         -------
#         Any
#             Updated/indexed variable.
#
#         """
#         if update is None:
#             return self.add_op('index', var, idx, *args)
#         if not update_type:
#             update_type = '='
#         return self.add_op(update_type, var, update, idx, *args)
#
#     def stack_vars(self, vars: tuple, **kwargs) -> np.ndarray:
#         """Group variables into a stack, with the number of variables being the first dimension of the stack.
#
#         Parameters
#         ----------
#         vars
#             Variables to stack together. Should have the same shapes.
#         kwargs
#             Additional keyword arguments (Ignored in this version of the backend).
#
#         Returns
#         -------
#         np.ndarray
#             Stacked variables.
#
#         """
#         return self.add_op('asarray', vars)
#
#     def to_file(self, fn: str):
#         """Writes backend to file
#
#         Parameters
#         ----------
#         fn
#
#         Returns
#         -------
#
#         """
#
#         pass
#
#     def from_file(self, fn: str):
#         """
#
#         Parameters
#         ----------
#         fn
#
#         Returns
#         -------
#
#         """
#
#         pass
#
#     @staticmethod
#     def eval(ops: list) -> list:
#         """Evaluates each operation in list.
#
#         Parameters
#         ----------
#         ops
#             List of operations to evaluate.
#
#         """
#         return [op.numpy() for op in ops]
#
#     @staticmethod
#     def eval_layer(layer: tuple) -> Any:
#         """Evaluate layer run function (can only be used after compilation has been performed).
#
#         Parameters
#         ----------
#         layer
#             Layer tuple as returned by `compile`.
#
#         Returns
#         -------
#         Any
#             Result of layer evaluation.
#
#         """
#         return layer[0](*layer[1])
#
#     def _set_op(self, op, idx1, idx2):
#         self.layers[self._base_layer+idx1][idx2] = op
#
#     def _add_op(self, op, layer=None):
#         if layer is None:
#             self.get_current_layer().append(op)
#         else:
#             self.get_layer(layer).append(op)
#
#     def _solve(self, rhs_func, func_args, T, dt, dts, t, solver, output_indices, **kwargs):
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
#         # choose solver
#         ###############
#
#         if solver == 'euler':
#
#             times, results = self._integrate(rhs_func=rhs_func, func_args=func_args, T=T, dt=dt, dts=dts, t=t,
#                                              output_indices=output_indices)
#
#         elif solver == 'scipy':
#
#             from scipy.integrate import solve_ivp
#
#             # solve via scipy's ode integration function
#             fun = lambda t, y: rhs_func(t, y, func_args)
#             if dts:
#                 times = np.arange(0, T, dts)
#                 kwargs['t_eval'] = times
#             outputs = solve_ivp(fun=fun, t_span=(float(t.numpy()), T), y0=self.vars['y'], first_step=dt,
#                                 **kwargs)
#             results = [outputs['y'].T[:, idx] for idx in output_indices]
#             times = outputs['t']
#             self.vars['y'] = outputs['y'].T[:, -1]
#
#         else:
#
#             raise ValueError(f'Invalid solver type: {solver}. See docstring of `CircuitIR.run` for valid solver types.')
#
#         return times, results
#
#     def _integrate(self, rhs_func, func_args, T, dt, dts, t, output_indices):
#
#         sampling_step = int(np.round(dts / dt, decimals=0))
#         sampling_steps = int(np.round(T / dts, decimals=0))
#         steps = int(np.round(T / dt, decimals=0))
#
#         # initialize results storage vectors
#         results = []
#         for i, idx in enumerate(output_indices):
#             if type(idx) is tuple:
#                 var_dim = idx[1]-idx[0]
#             elif type(idx) is list:
#                 if all(np.diff(idx, n=1) == 1):
#                     output_indices[i] = (idx[0], idx[-1]+1)
#                     var_dim = idx[-1]+1-idx[0]
#                 else:
#                     var_dim = len(idx)
#             else:
#                 var_dim = 1
#             results.append(np.zeros((sampling_steps, var_dim)))
#
#         # solve via pyrates internal explicit euler algorithm
#         state_vars = self.vars['y']
#         sampling_idx = 0
#         for i in range(steps):
#             deltas = rhs_func(i, state_vars, func_args)
#             state_vars += dt * deltas
#             if i % sampling_step == 0:
#                 for idx1, idx2 in enumerate(output_indices):
#                     results[idx1][sampling_idx, :] = state_vars[idx2[0]:idx2[1]] if type(idx2) is tuple \
#                         else state_vars[idx2]
#                 sampling_idx += 1
#
#         self.vars['y'] = state_vars
#         times = np.arange(0, T, dts)
#
#         return times, results
#
#     def _match_shapes(self, op1: Any, op2: Any, adjust_second: bool = True) -> tuple:
#         """Re-shapes op1 and op2 such that they can be combined via mathematical operations.
#
#         Parameters
#         ----------
#         op1
#             First operator.
#         op2
#             Second operator.
#         adjust_second
#             If true, the second operator will be re-shaped according to the shape of the first operator. If false,
#             it will be done the other way around.
#
#         Returns
#         -------
#         tuple
#             The re-shaped operators.
#
#         """
#
#         if not hasattr(op1, 'shape'):
#             op1 = self.add_var(vtype='constant', value=op1, name='c')
#         if not hasattr(op2, 'shape'):
#             op2 = self.add_var(vtype='constant', value=op2, name='c')
#
#         if adjust_second:
#             op_adjust = op2
#             op_target = op1
#         else:
#             op_adjust = op1
#             op_target = op2
#
#         if len(op_target.shape) > len(op_adjust.shape):
#
#             # add axis to op_adjust
#             target_shape = op_target.shape
#             idx = list(target_shape).index(1) if 1 in op_target.shape else len(target_shape) - 1
#             op_adjust = self.add_op('expand', op_adjust, idx)
#
#         elif (len(op_adjust.shape) > len(op_target.shape) and 1 in op_adjust.shape) or \
#                 (len(op_target.shape) == 2 and len(op_adjust.shape) == 2 and op_target.shape[1] != op_adjust.shape[0]
#                  and 1 in op_adjust.shape):
#
#             # cut singleton dimension from op_adjust
#             old_shape = list(op_adjust.shape)
#             idx = ",".join(['0' if s == 1 else ':' for s in old_shape])
#             op_adjust = self.add_op('index', op_adjust, idx)
#
#         if adjust_second:
#             return op_target, op_adjust
#         return op_adjust, op_target
#
#     def _match_dtypes(self, op1: Any, op2: Any) -> tuple:
#         """Match data types of two operators/variables.
#         """
#
#         if issubclass(type(op1), PyRatesOp):
#
#             if issubclass(type(op2), PyRatesOp):
#
#                 # cast both variables to lowest precision
#                 for acc in ["16", "32", "64", "128"]:
#                     if acc in op2.dtype:
#                         return self.add_op('cast', op1, op2.dtype), op2
#
#             return op1, self.add_op('cast', op2, op1.dtype)
#
#         elif issubclass(type(op2), PyRatesOp):
#
#             return self.add_op('cast', op1, op2.dtype), op2
#
#         elif hasattr(op1, 'numpy') or type(op1) is np.ndarray:
#
#             # cast op2 to numpy dtype of op1
#             return op1, self.add_op('cast', op2, op1.dtype)
#
#         elif hasattr(op2, 'numpy') or type(op2) is np.ndarray:
#
#             # cast op1 to numpy dtype of op2
#             return self.add_op('cast', op1, op2.dtype), op2
#
#         else:
#
#             # cast op2 to dtype of op1 referred from its type string
#             return op1, self.add_op('cast', op2, str(type(op1)))
#
#     def _create_var(self, vtype, dtype, shape, value, name, squeeze=True):
#         return BaseVar(vtype=vtype, dtype=dtype, shape=shape, value=value, name=name, backend=self, squeeze=squeeze)
#
#     def _create_op(self, op, name, *args):
#         if not self.ops[op]['str']:
#             raise NotImplementedError(f"The operator `{op}` is not implemented for this backend ({self.name}). "
#                                       f"Please consider passing the required operation to the backend initialization "
#                                       f"or choose another backend.")
#         if op in ["=", "+=", "-=", "*=", "/="]:
#             if len(args) > 2 and hasattr(args[2], 'shape') and len(args[2].shape) > 1 and \
#                     'bool' not in str(type(args[2])):
#                 args_tmp = list(args)
#                 if not hasattr(args_tmp[2], 'short_name'):
#                     idx = self.add_var(vtype='state_var', name='idx', value=args_tmp[2])
#                 else:
#                     idx = args_tmp[2]
#                 var, upd, idx = self._process_update_args_old(args[0], args[1], idx)
#                 idx_str = ",".join([f"{idx.short_name}[:,{i}]" for i in range(idx.shape[1])])
#                 args = (var, upd, idx_str, idx)
#             return PyRatesAssignOp(self.ops[op]['str'], self.ops[op]['func'], name, *args,
#                                    idx_l=self.idx_l, idx_r=self.idx_r)
#         elif op == "index":
#             return PyRatesIndexOp(self.ops[op]['str'], self.ops[op]['func'], name, *args, idx_l=self.idx_l,
#                                   idx_r=self.idx_r)
#         else:
#             if op == "cast":
#                 args = list(args)
#                 for dtype in self.dtypes:
#                     if dtype in str(args[1]):
#                         args[1] = f"np.{dtype}"
#                         break
#                 args = tuple(args)
#             return PyRatesOp(self.ops[op]['str'], self.ops[op]['func'], name, *args)
#
#     def _process_update_args_old(self, var, update, idx):
#         """Preprocesses the index and a variable update to match the variable shape.
#
#         Parameters
#         ----------
#         var
#         update
#         idx
#
#         Returns
#         -------
#         tuple
#             Preprocessed index and re-shaped update.
#
#         """
#
#         # pre-process args
#         ###################
#
#         shape = var.shape
#
#         # match shape of index and update to shape
#         ##########################################
#
#         if tuple(update.shape) and idx.shape[0] != update.shape[0]:
#             if update.shape[0] == 1 and len(update.shape) == 1:
#                 idx = self.add_op('reshape', idx, tuple(idx.shape) + (1,))
#             elif idx.shape[0] == 1:
#                 update = self.add_op('reshape', update, (1,) + tuple(update.shape))
#             elif len(update.shape) > len(idx.shape) and 1 in update.shape:
#                 singleton = list(update.shape).index(1)
#                 update = self.add_op('squeeze', update, singleton)
#             else:
#                 raise ValueError(f'Invalid indexing. Operation of shape {shape} cannot be updated with updates of '
#                                  f'shapes {update.shape} at locations indicated by indices of shape {idx.shape}.')
#         elif not tuple(update.shape) and tuple(idx.shape):
#             update = self.add_op('reshape', update, (1,))
#
#         if len(idx.shape) < 2:
#             idx = self.add_op('reshape', idx, tuple(idx.shape) + (1,))
#
#         shape_diff = len(shape) - idx.shape[1]
#         if shape_diff < 0:
#             raise ValueError(f'Invalid index shape. Operation has shape {shape}, but received an index of '
#                              f'length {idx.shape[1]}')
#
#         # manage shape of updates
#         update_dim = 0
#         if len(update.shape) > 1:
#             update_dim = len(update.shape[1:])
#             if update_dim > shape_diff:
#                 singleton = -1 if update.shape[-1] == 1 else list(update.shape).index(1)
#                 update = self.add_op('squeeze', update, singleton)
#                 if len(update.shape) > 1:
#                     update_dim = len(update.shape[1:])
#
#         # manage shape of idx
#         shape_diff = int(shape_diff) - update_dim
#         if shape_diff > 0:
#             indices = []
#             indices.append(idx)
#             for i in range(shape_diff):
#                 indices.append(self.add_op('zeros', tuple(idx.shape), idx.dtype))
#             idx = self.add_op('concat', indices, 1)
#
#         return var, update, idx
#
#     def _process_update_args(self, var, update, idx=None):
#         """Preprocesses the index and a variable update to match the variable shape.
#
#         Parameters
#         ----------
#         var
#         update
#         idx
#
#         Returns
#         -------
#         tuple
#             Preprocessed index and re-shaped update.
#
#         """
#
#         # pre-process args
#         shape = var.shape
#
#         # match shape of index and update to shape
#         ##########################################
#
#         if len(shape) == len(update.shape) and shape != update.shape:
#
#             # make sure that update dims match the variable shape
#             update = self.add_op("squeeze", update)
#
#         if shape[0] == idx.shape[0] and len(shape) > 1 and sum(idx.shape) > 1:
#
#             # make sure that index scatters into first dimension of variable
#             idx = self.add_op("index", idx, ":, None")
#
#         elif update.shape == shape[1:] or update.shape == shape[:-1]:
#
#             # make sure that index and update scatter into the first dimension of update
#             update = self.add_op('reshape', update, (1,) + tuple(update.shape))
#             idx = idx[None, :]
#
#         return var, update, idx
#
#     def _process_vars(self):
#         """
#
#         Returns
#         -------
#
#         """
#         state_vars, constants, var_map = [], [], {}
#         s_idx, c_idx, s_len = 0, 0, 0
#         for key, var in self.vars.items():
#             key, state_var = self._is_state_var(key)
#             if state_var:
#                 state_vars.append((key, var))
#                 var_map[key] = ('state_var', (s_idx, s_len))
#                 s_idx += 1
#                 s_len += 1
#             elif var.vtype == 'constant' or (var.short_name not in self.lhs_vars and key != 'y'):
#                 constants.append((key, var))
#                 var_map[key] = ('constant', c_idx)
#                 c_idx += 1
#         return state_vars, constants, var_map
#
#     def _is_state_var(self, key):
#         return key, key in self.state_vars
#
#     @staticmethod
#     def _process_func_args(args, var_map, dt):
#         for key, var_info in var_map.items():
#             if 'times' in key:
#                 args[var_info[1]] -= dt
#         return list(args)
#
#     @staticmethod
#     def _deepcopy(x):
#         return deepcopy(x)
#
#
# def extract_lhs_var(eq_str: str) -> str:
#     """
#
#     Parameters
#     ----------
#     eq_str
#
#     Returns
#     -------
#
#     """
#
#     lhs = eq_str.split("=")[0]
#     lhs = lhs.replace(" ", "")
#     if "[" in lhs:
#         idx = lhs.index("[")
#         lhs = lhs[:idx]
#     return lhs
