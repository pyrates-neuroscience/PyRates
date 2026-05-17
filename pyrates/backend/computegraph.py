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

"""This module provides the backend class that should be used to set up any backend in pyrates.
"""

# external _imports
from typing import Any, Callable, Union, Iterable, Optional
from networkx import MultiDiGraph
from sympy import Symbol, Expr, Function, lambdify
import numpy as np

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


#########################
# compute graph classes #
#########################


# numpy-based node class
class ComputeNode:
    """Base class for adding variables to the compute graph. Creates a numpy array with additional attributes
    for variable identification/retrieval from graph. Should be used as parent class for custom variable classes.

    Parameters
    ----------
    name
        Full name of the variable in the original NetworkGraph (including the node and operator it belongs to).
    dtype
        Data-type of the variable. For valid data-types, check the documentation of the backend in use.
    shape
        Shape of the variable.
    """

    __slots__ = ["name", "symbol", "dtype", "shape", "_value"]

    def __init__(self, name: str, symbol: Union[Symbol, Expr, Function], dtype: Optional[str] = None,
                 shape: tuple = (), def_shape: tuple = ()):
        """Instantiates a basic node of a ComputeGraph instance.
        """

        self.name = name
        self.symbol = symbol
        self.shape = self._get_shape(shape, def_shape)
        self._value = np.zeros(self.shape)
        self.dtype = dtype
        self.set_dtype()

    def reshape(self, shape: tuple, **kwargs):

        self._value = self.value.reshape(shape, **kwargs)
        self.shape = shape
        return self

    def squeeze(self, axis=None):
        self._value = self.value.squeeze(axis=axis)
        self.shape = self._value.shape
        return self

    def set_value(self, v: Union[float, np.ndarray]):
        self._value = np.asarray(v, dtype=self.dtype)
        self.shape = tuple(v.shape)

    @property
    def value(self):
        """Returns current value of BaseVar.
        """
        return self._value

    @property
    def is_constant(self):
        raise NotImplementedError("This method has to be defined by each child class.")

    @property
    def is_float(self):
        return "float" in self.dtype

    @property
    def is_complex(self):
        return "complex" in self.dtype

    def _is_equal_to(self, v):
        for attr in self.__slots__:
            if not hasattr(v, attr) or getattr(v, attr) != getattr(self, attr):
                return False
        return True

    def _get_value(self, value: Optional[Union[list, np.ndarray]] = None, dtype: Optional[str] = None,
                   shape: tuple = ()):
        """Defines initial value of variable.
        """

        # case I: create new array from shape and dtype
        if value is None:
            return np.zeros(shape=shape, dtype=dtype)

        # case II: transform values into an array
        if not hasattr(value, 'shape'):
            if type(value) is list:
                return self._get_value(value=np.asarray(value, dtype=dtype), dtype=dtype, shape=shape)
            return np.zeros(shape=shape, dtype=dtype) + value

        # case III: match given shape with the shape of the given value array
        if len(shape) > 0:
            value = np.asarray(value, dtype=dtype)
            if value.shape == shape:
                return value
            if sum(shape) < sum(value.shape):
                return value.squeeze()
            idx = ",".join("None" if s == 1 else ":" for s in shape)
            return eval(f'value[{idx}]')

        # case IV: just ensure the correct data type of the value array
        return np.asarray(value, dtype=dtype)

    def set_dtype(self, dtype: str = None):
        if dtype is None:
            if not self.dtype:
                if 'float' in str(self.value.dtype):
                    self.dtype = 'float'
                elif 'complex' in str(self.value.dtype):
                    self.dtype = 'complex'
                else:
                    self.dtype = 'int'
        else:
            self.dtype = dtype

    @staticmethod
    def _get_shape(s: tuple, s_def: tuple):
        if sum(s) <= 1:
            return s_def
        return s

    def __deepcopy__(self, memodict: dict):
        node = ComputeNode(name=self.name, symbol=self.symbol, dtype=self.dtype, shape=self.shape)
        node._value = np.zeros_like(node._value) + node._value
        return node

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(str(self))


class ComputeVar(ComputeNode):
    """Class for variables and vector-valued constants in the ComputeGraph.
    """

    __slots__ = ComputeNode.__slots__ + ["vtype"]

    def __init__(self, name: str, symbol: Union[Symbol, Expr, Function], vtype: str, dtype: Optional[str] = None,
                 shape: tuple = (), value: Optional[Union[list, np.ndarray]] = None, def_shape: tuple = ()):

        # set attributes
        super().__init__(name=name, symbol=symbol, dtype=dtype, shape=shape, def_shape=def_shape)
        self.vtype = vtype

        # adjust variable value
        self.set_value(self._get_value(value=value, shape=self.shape, dtype=self.dtype))

    @property
    def is_constant(self):
        return self.vtype == 'constant'


class ComputeOp(ComputeNode):
    """Class for ComputeGraph nodes that represent mathematical operations.
    """

    __slots__ = ComputeNode.__slots__ + ["func", "expr", "func_args", "backend_funcs"]

    def __init__(self, name: str, symbol: Union[Symbol, Expr, Function], expr: Expr,
                 func: Optional[Callable] = None, func_args: Optional[list] = None,
                 backend_funcs: Optional[dict] = None, dtype: Optional[str] = None, shape: tuple = ()):

        # set attributes
        super().__init__(name=name, symbol=symbol, dtype=dtype, shape=shape)
        self.func = func
        self.expr = expr
        self.func_args = func_args if func_args is not None else []
        self.backend_funcs = backend_funcs if backend_funcs is not None else {}

    def get_func(self) -> Callable:
        if self.func is None:
            self.func = lambdify(self.func_args, expr=self.expr, modules=[self.backend_funcs, "numpy"])
        return self.func

    @property
    def is_constant(self):
        return False


# networkx-based graph class
class ComputeGraph(MultiDiGraph):
    """Creates a compute graph where nodes are all constants and variables of the network and edges are the mathematical
    operations linking those variables/constants together to form equations.
    """

    def __init__(self, backend: str, **kwargs):

        super().__init__()

        # choose a backend
        if backend == 'torch':
            from pyrates.backend.torch import TorchBackend
            backend = TorchBackend
        elif backend == 'fortran':
            from pyrates.backend.fortran import FortranBackend
            backend = FortranBackend
        elif backend == 'julia':
            from pyrates.backend.julia import JuliaBackend
            backend = JuliaBackend
        elif backend == 'matlab':
            from pyrates.backend.matlab import MatlabBackend
            backend = MatlabBackend
        else:
            from pyrates.backend.base import BaseBackend
            backend = BaseBackend

        # backend-related attributes
        self.backend = backend(**kwargs)
        self.var_updates = {'DEs': dict(), 'non-DEs': dict()}
        self._eq_nodes = []
        self._state_var_indices = dict()
        self._state_var_hist = dict()
        self._node_names = {}

    @property
    def state_vars(self):
        return list(self.var_updates['DEs'].keys())

    def add_var(self, label: str, value: Any, vtype: str, **kwargs):

        unique_label = self._generate_unique_label(label)
        var = ComputeVar(name=unique_label, symbol=Symbol(unique_label), value=value, vtype=vtype, **kwargs)
        super().add_node(unique_label, node=var)
        return unique_label, self.nodes[unique_label]['node']

    def add_op(self, inputs: Union[list, tuple], label: str, expr: Expr, func: Optional[Callable] = None,
               func_args: Optional[list] = None, backend_funcs: Optional[dict] = None, **kwargs):

        # add target node that contains result of operation
        unique_label = self._generate_unique_label(label)
        op = ComputeOp(name=unique_label, symbol=Symbol(unique_label), expr=expr, func=func,
                       func_args=func_args, backend_funcs=backend_funcs, **kwargs)
        super().add_node(unique_label, node=op)

        # add edges from source nodes to target node
        for i, v in enumerate(inputs):
            super().add_edge(v, unique_label, key=i)

        return unique_label, self.nodes[unique_label]['node']

    def add_var_update(self, var: str, update: str, differential_equation: bool = False):

        # store mapping between left-hand side variable and right-hand side update
        if differential_equation:
            self.var_updates['DEs'][var] = update
        else:
            self.var_updates['non-DEs'][var] = update

        # remember var and update node to ensure that they are not pruned during compilation
        self._eq_nodes.extend([var, update])

    def get_var(self, var: str, from_backend: bool = False):
        v = self.nodes[var]['node']
        if from_backend:
            return self.backend.get_var(v)
        return v

    def get_op(self, op: str, **kwargs) -> dict:
        return self.backend.get_op(op, **kwargs)

    def eval_graph(self):

        for n in self.var_updates['non-DEs'].values():
            self.eval_subgraph(n)
        return self.eval_nodes(self.var_updates['DEs'].values())

    def eval_nodes(self, nodes: Iterable):

        return [self.eval_node(n) for n in nodes]

    def eval_node(self, n):

        inputs = tuple([self.eval_node(inp) for inp in self.predecessors(n)])
        node = self.get_var(n)
        if isinstance(node, ComputeOp):
            return node.get_func()(*inputs)
        return node.value

    def eval_subgraph(self, n):

        inputs = []
        input_nodes = [node for node in self.predecessors(n)]
        for inp in input_nodes:
            inputs.append(self.eval_subgraph(inp))
            self.remove_node(inp)

        node = self.get_var(n)
        if inputs:
            node.set_value(node.get_func()(*tuple(inputs)))

        return node.value

    def remove_subgraph(self, n):

        for inp in self.predecessors(n):
            self.remove_subgraph(inp)
        self.remove_node(n)

    def compile(self):

        # evaluate constant-based operations
        out_nodes = [node for node, out_degree in self.out_degree if out_degree == 0]
        for node in out_nodes:

            # process inputs of node
            for inp in self.predecessors(node):
                if self.get_var(inp).is_constant:
                    self.eval_subgraph(inp)

            # evaluate node if all its inputs are constants
            if all([self.get_var(inp).is_constant for inp in self.predecessors(node)]) and node not in self._eq_nodes:
                self.eval_subgraph(node)

        # remove unconnected nodes and constants from graph
        self._prune()

        return self

    def to_func(self, func_name: str, to_file: bool = True, dt_adapt: bool = True, dt: float = None, **kwargs):

        # finalize compute graph
        self.compile()

        # create state variable vector and state variable update vector
        ###############################################################

        variables = []
        idx = 0
        for var, update in self.var_updates['DEs'].items():

            # extract left-hand side nodes from graph
            lhs, rhs = self._process_var_update(var, update)
            variables.append(lhs.value)

            # store information of the original, non-vectorized state variable
            vshape = sum(lhs.shape)
            if vshape > 1:
                self._state_var_indices[var] = (idx, idx+vshape)
                idx += vshape
            else:
                self._state_var_indices[var] = idx
                idx += 1

        # add collected state variables to the backend
        try:
            state_vec = np.concatenate(variables, axis=0)
        except ValueError:
            try:
                state_vec = np.asarray(variables)
            except ValueError:
                state_vec = np.asarray([np.squeeze(v) for v in variables])
        dtype = 'complex' if 'complex' in state_vec.dtype.name else 'float'
        state_var_key, y = self.add_var(label='y', vtype='state_var', value=state_vec, dtype=dtype)
        rhs_var_key = self._generate_vecfield_var(state_vec, dtype)
        try:
            t = self.get_var('t')
        except KeyError:
            _, t = self.add_var(label='t', vtype='state_var', value=0.0 if dt_adapt else 0,
                                dtype='float' if dt_adapt else 'int', shape=())
        self.backend.register_vars([t, y])

        # create a string containing all computations and variable updates represented by the compute graph
        func_args, code_gen = self._to_str()
        func_body = code_gen.generate()
        code_gen.code.clear()

        # generate function head
        add_hist_calls = self._state_var_hist and code_gen.add_hist_arg
        func_args = code_gen.generate_func_head(func_name=func_name, state_var=state_var_key, return_var=rhs_var_key,
                                                func_args=[self.get_var(arg) for arg in func_args],
                                                add_hist_func=add_hist_calls)

        # extract state variable histories for delayed interactions
        code_gen.add_linebreak()
        for var, delays in self._state_var_hist.items():

            # extract index of variable in state vector
            idx = self._state_var_indices[var]

            # extract state variable history from backend-specific buffer
            if type(idx) is not int:
                if idx[1]-idx[0] < 2:
                    idx = idx[0]
            for delay, v_hist in delays.items():  # type: ComputeVar, str
                code_gen.add_var_hist(lhs=v_hist, delay=delay, state_idx=idx, var=var,
                                      dt=dt, dt_adapt=dt_adapt)

        # add lines from function body after function head
        code_gen.add_linebreak()
        code_gen.add_code_line(func_body)
        code_gen.add_linebreak()

        # generate function tail
        self._generate_func_tail(code_gen, rhs_var_key)

        # generate the function (and write to file, optionally)
        func_args_tmp = func_args[4:] if add_hist_calls else func_args[3:]
        func = code_gen.generate_func(func_name=func_name, to_file=to_file, func_args=func_args_tmp,
                                      state_vars=self.state_vars, **kwargs)

        # OPTIONAL: write function arguments (state vectors and constants) to file
        c_fn = kwargs.pop('constants_file_name', None)
        if c_fn:
            arg_dict = {arg: self.get_var(arg).value for arg in func_args if arg != 'hist'}
            if code_gen.lags:
                arg_dict['lags'] = list(code_gen.lags.keys())
            fn = f'{self.backend.fdir}/{c_fn}' if self.backend.fdir else c_fn
            code_gen.to_file(fn, **arg_dict)

        # finalize the function arguments
        fargs = []
        for arg in func_args:
            if arg == 'hist':
                if 'hist' in kwargs:
                    arg = kwargs.pop('hist')
                else:
                    y_init = np.asarray(state_vec[:])
                    arg = code_gen.get_hist_func(y_init)
                fargs.append(arg)
            else:
                fargs.append(self.get_var(arg, from_backend=True))

        return func, tuple(fargs), tuple(func_args), self._state_var_indices.copy()

    def run(self, func: Callable, func_args: tuple, T: float, dt: float, dts: Optional[float] = None,
            outputs: Optional[dict] = None, **kwargs) -> dict:

        # pre-process outputs
        if outputs is None:
            outputs = {key: key for key in self.state_vars}
        for key in outputs.copy():
            var = outputs.pop(key)
            outputs[key] = self._state_var_indices[var]

        # handle other arguments
        if dts is None:
            dts = dt
        solver = kwargs.pop('solver', 'euler')

        # call backend method
        results, times = self.backend.run(func=func, func_args=func_args, T=T, dt=dt, dts=dts, solver=solver, **kwargs)

        # set state variables to final simulated value
        for key in self.state_vars:
            var = self.get_var(key)
            idx = self._state_var_indices[key]
            var.set_value(np.reshape(self._index_state_var(results, idx)[-1], var.shape))

        # reduce state recordings to requested state variables
        for key, idx in outputs.items():
            outputs[key] = self._index_state_var(results, idx)
        outputs['time'] = times

        return outputs

    def get_jacobian_func(self, func_name: str, to_file: bool = True, sparse: bool = False,
                          dt_adapt: bool = True, dt: float = None, **kwargs) -> tuple:
        """Generate a function that evaluates the Jacobian of the vector field.

        For ODE systems the generated function has signature ``J(t, y, *params) -> ndarray (n, n)``.
        For DDE systems it returns ``(J0, [J_tau1, J_tau2, ...])`` where ``J0`` is the instantaneous
        Jacobian and ``J_tauK`` is the partial derivative with respect to ``y(t - tau_k)``.
        If ``sparse=True`` each matrix is a ``scipy.sparse.csr_matrix`` instead.

        The Jacobian entries are computed symbolically via ``sympy.diff`` using the reconstructed
        vector-field expressions from the compute graph.  Vector-valued state variables fall back to
        zero blocks (a ``UserWarning`` lists the affected indices).

        Parameters
        ----------
        func_name
            Name of the generated function.
        to_file
            Write source to a file (same semantics as ``to_func``).
        sparse
            Return ``scipy.sparse.csr_matrix`` instead of dense ``ndarray``.
        dt_adapt
            Whether the circuit uses adaptive time-stepping (controls ``t`` dtype).
        dt
            Fixed time-step (passed through to backend).

        Returns
        -------
        tuple
            ``(func, args, arg_names, state_var_indices)`` — same structure as ``to_func``.
        """
        import sympy as sp
        from warnings import warn

        # ── 1. Finalise graph and build state-vector structure (mirrors to_func) ──
        self.compile()
        variables, idx = [], 0
        for var, update in self.var_updates['DEs'].items():
            lhs, _ = self._process_var_update(var, update)
            variables.append(lhs.value)
            vshape = sum(lhs.shape)
            if vshape > 1:
                self._state_var_indices[var] = (idx, idx + vshape)
                idx += vshape
            else:
                self._state_var_indices[var] = idx
                idx += 1
        n = idx
        try:
            state_vec = np.concatenate(variables, axis=0)
        except ValueError:
            try:
                state_vec = np.asarray(variables)
            except ValueError:
                state_vec = np.asarray([np.squeeze(v) for v in variables])
        dtype = 'complex' if 'complex' in state_vec.dtype.name else 'float'
        state_var_key, y_var = self.add_var(label='y', vtype='state_var', value=state_vec, dtype=dtype)
        try:
            t_var = self.get_var('t')
        except KeyError:
            _, t_var = self.add_var(label='t', vtype='state_var',
                                    value=0.0 if dt_adapt else 0,
                                    dtype='float' if dt_adapt else 'int', shape=())
        self.backend.register_vars([t_var, y_var])

        # ── 2. Rebuild symbolic vector field ──
        f_exprs, y_syms, past_map, func_args, var_is_vector = self._get_symbolic_rhs()

        # map state sym → y-index
        sym_to_y_idx = {sym: self._state_var_indices[var]
                        for var, sym in zip(self.var_updates['DEs'].keys(), y_syms)}

        is_dde = bool(past_map)
        start = self.backend._start_idx  # 0 for Python, 1 for Julia/MATLAB

        # group past symbols by delay string for history-Jacobian computation
        # delay_str → [(fresh_sym, var_sym, state_idx)]
        delay_groups: dict = {}
        for (var_sym, delay_sym), fresh_sym in past_map.items():
            d_str = str(delay_sym)
            if d_str not in delay_groups:
                delay_groups[d_str] = []
            for sym, vidx in sym_to_y_idx.items():
                if sym == var_sym:
                    delay_groups[d_str].append((fresh_sym, var_sym, vidx))
                    break

        # fresh past sym → code string used in generated Jacobian expressions
        past_sym_to_str: dict = {}
        for d_str, group in delay_groups.items():
            d_safe = d_str.replace('.', 'p').replace('-', 'm')
            for fresh_sym, _, vidx in group:
                if isinstance(vidx, tuple):
                    past_sym_to_str[fresh_sym] = f'_yhist_{d_safe}[{vidx[0]+start}:{vidx[1]}]'
                else:
                    past_sym_to_str[fresh_sym] = f'_yhist_{d_safe}[{vidx+start}]'

        # ── 3. Compute symbolic Jacobian entries via sympy.diff ──
        J0_entries: dict = {}   # (i_row, j_col) → sympy Expr
        J_hist: dict = {}       # delay_str → {(i_row, j_col) → sympy Expr}
        numerical_blocks = []

        for d_str in delay_groups:
            J_hist[d_str] = {}

        i_row = 0
        for f_i, yi_sym, fi_is_vec in zip(f_exprs, y_syms, var_is_vector):
            fi_idx = sym_to_y_idx[yi_sym]
            fi_nrows = (fi_idx[1] - fi_idx[0]) if isinstance(fi_idx, tuple) else 1

            j_col = 0
            for yj_sym, fj_is_vec in zip(y_syms, var_is_vector):
                fj_idx = sym_to_y_idx[yj_sym]
                fj_ncols = (fj_idx[1] - fj_idx[0]) if isinstance(fj_idx, tuple) else 1
                if fi_is_vec or fj_is_vec:
                    numerical_blocks.append((i_row, i_row + fi_nrows, j_col, j_col + fj_ncols))
                else:
                    d = sp.diff(f_i, yj_sym)
                    if d != 0:
                        J0_entries[(i_row, j_col)] = d
                j_col += fj_ncols

            # history Jacobians
            for d_str, group in delay_groups.items():
                j_col = 0
                for fresh_sym, _, fj_idx in group:
                    fj_ncols = (fj_idx[1] - fj_idx[0]) if isinstance(fj_idx, tuple) else 1
                    if not fi_is_vec and fj_ncols == 1:
                        d = sp.diff(f_i, fresh_sym)
                        if d != 0:
                            J_hist[d_str][(i_row, j_col)] = d
                    j_col += fj_ncols

            i_row += fi_nrows

        if numerical_blocks:
            warn(
                f"get_jacobian_func: {len(numerical_blocks)} vector-valued Jacobian block(s) set to zero "
                f"(analytical differentiation not supported for vector DEs). "
                f"Affected y-index ranges: {numerical_blocks}",
                UserWarning
            )

        # ── 4. Generate code ──
        code_gen = self.backend
        code_gen.code.clear()

        # ensure zeros (and optionally csr_matrix) are imported
        code_gen.add_import("from numpy import zeros")
        if sparse:
            code_gen.add_import("from scipy.sparse import csr_matrix")

        # determine return-variable name(s) for MATLAB function signature
        if J_hist:
            d_safes = [d_str.replace('.', 'p').replace('-', 'm') for d_str in J_hist]
            jk_names = [f'J_hist_{s}' for s in d_safes]
            return_var_name = f'J0'   # Julia/MATLAB named outputs handled below
        else:
            jk_names = []
            return_var_name = 'J0'

        func_args_objects = [self.get_var(a) for a in func_args]
        all_arg_names = code_gen.generate_func_head(
            func_name=func_name,
            state_var=state_var_key,
            return_var=return_var_name,
            func_args=func_args_objects,
            add_hist_func=is_dde and code_gen.add_hist_arg,
        )

        # past-state extraction for DDE
        if is_dde:
            code_gen.add_linebreak()
            for d_str, group in delay_groups.items():
                d_safe = d_str.replace('.', 'p').replace('-', 'm')
                code_gen.add_code_line(f"_yhist_{d_safe} = hist(t - {d_str})")

        # allocate instantaneous Jacobian
        code_gen.add_linebreak()
        dtype_str = code_gen._float_precision
        code_gen.add_code_line(f"J0 = zeros(({n}, {n}), dtype='{dtype_str}')")

        # fill non-zero J0 entries
        for (i_r, j_c), d_expr in sorted(J0_entries.items()):
            d_str_code = self._expr_to_jac_str(d_expr, sym_to_y_idx, {})
            if d_str_code is None:
                code_gen.add_code_line(
                    f"# WARNING: could not differentiate J0[{i_r},{j_c}] analytically — entry left as 0")
            else:
                code_gen.add_code_line(f"J0[{i_r + start}, {j_c + start}] = {d_str_code}")

        # history Jacobians
        code_gen.add_linebreak()
        for d_str, entries in J_hist.items():
            d_safe = d_str.replace('.', 'p').replace('-', 'm')
            jk = f'J_hist_{d_safe}'
            code_gen.add_code_line(f"{jk} = zeros(({n}, {n}), dtype='{dtype_str}')")
            for (i_r, j_c), d_expr in sorted(entries.items()):
                d_str_code = self._expr_to_jac_str(d_expr, sym_to_y_idx, past_sym_to_str)
                if d_str_code is None:
                    code_gen.add_code_line(
                        f"# WARNING: could not differentiate {jk}[{i_r},{j_c}] analytically — entry left as 0")
                else:
                    code_gen.add_code_line(f"{jk}[{i_r + start}, {j_c + start}] = {d_str_code}")
            code_gen.add_linebreak()

        # return statement (replaces generate_func_tail so we control the return value)
        if sparse:
            j0_ret = "csr_matrix(J0)"
            jk_rets = [f"csr_matrix({jk})" for jk in jk_names]
        else:
            j0_ret = "J0"
            jk_rets = jk_names

        if jk_names:
            ret_str = f"{j0_ret}, [{', '.join(jk_rets)}]"
        else:
            ret_str = j0_ret
        code_gen.generate_func_tail(rhs_var=ret_str)

        # compile / write to file
        param_arg_names = [a for a in all_arg_names if a not in ('t', state_var_key, 'hist')]
        func = code_gen.generate_func(
            func_name=func_name, to_file=to_file,
            func_args=param_arg_names,
            state_vars=self.state_vars,
            **kwargs
        )

        # assemble actual argument values
        fargs = [0.0, state_vec.copy()]
        if is_dde and code_gen.add_hist_arg:
            fargs.append(code_gen.get_hist_func(state_vec.copy()))
        for a in all_arg_names:
            if a not in ('t', state_var_key, 'hist'):
                fargs.append(self.get_var(a, from_backend=True))

        return func, tuple(fargs), tuple(all_arg_names), self._state_var_indices.copy()

    def _get_symbolic_rhs(self) -> tuple:
        """Reconstruct the full symbolic vector field from the compute graph.

        Must be called after ``compile()``.

        Returns
        -------
        f_exprs : list[sympy.Expr]
        y_syms : list[sympy.Symbol]
        past_map : dict  ``{(var_sym, delay_sym): fresh_sym}``
        func_args : list[str]  constant node names (function parameters)
        var_is_vector : list[bool]
        """
        import sympy as sp

        # Build symbolic expressions for all non-DE (algebraic) variables so that
        # delayed inputs routed via buffer operators can be expanded into the DE
        # expressions and their past() calls can be detected.
        non_de_exprs: dict = {}
        non_de_args: list = []
        for var, update in self.var_updates['non-DEs'].items():
            nde_var = self.get_var(var)
            try:
                args, expr = self._node_to_expr(update)
            except Exception:
                continue
            non_de_exprs[nde_var.symbol] = expr
            non_de_args.extend(args)

        def _expand_non_de(expr):
            """Substitute non-DE symbols with their full expressions."""
            changed = True
            while changed:
                changed = False
                subs = {}
                for sym in expr.free_symbols:
                    if sym in non_de_exprs:
                        subs[sym] = non_de_exprs[sym]
                if subs:
                    new_expr = expr.subs(subs)
                    if new_expr != expr:
                        expr = new_expr
                        changed = True
            return expr

        all_func_args = list(non_de_args)
        f_exprs, y_syms, past_map, var_is_vector = [], [], {}, []
        for var, update in self.var_updates['DEs'].items():
            lhs_node = self.get_var(var)
            args, expr = self._node_to_expr(update)
            all_func_args.extend(args)
            # expand non-DE symbols to reveal any past() calls
            expr = _expand_non_de(expr)
            expr, new_past = self._extract_past_terms(expr)
            past_map.update(new_past)
            f_exprs.append(expr)
            y_syms.append(lhs_node.symbol)
            var_is_vector.append(sum(lhs_node.shape) > 1)

        seen, ordered = set(), []
        for a in all_func_args:
            if a not in seen:
                seen.add(a)
                ordered.append(a)
        return f_exprs, y_syms, past_map, ordered, var_is_vector

    def _extract_past_terms(self, expr) -> tuple:
        """Replace every ``past(var, delay)`` call in *expr* with a fresh Symbol.

        Returns ``(new_expr, past_map)`` where ``past_map`` maps
        ``(var_sym, delay_sym) → fresh_sym``.
        """
        from sympy import Symbol
        past_map: dict = {}

        def _visit(e):
            if not e.args:
                return e
            if e.func.__name__ == 'past' and len(e.args) == 2:
                key = (e.args[0], e.args[1])
                if key not in past_map:
                    safe = str(e.args[1]).replace('.', 'p').replace('-', 'm')
                    past_map[key] = Symbol(f'_past_{e.args[0]}_{safe}')
                return past_map[key]
            new_args = tuple(_visit(a) for a in e.args)
            if new_args != e.args:
                return e.func(*new_args)
            return e

        return _visit(expr), past_map

    def _resolve_derivatives(self, expr):
        """Replace ``Derivative(f(x), x)`` with known analytical forms.

        Currently handles: ``identity`` (pass-through), ``sigmoid``, and ``absv``.
        """
        import sympy as sp
        from sympy import Derivative, Function

        # identity(x) = x  →  d/dx = 1
        expr = expr.replace(
            lambda e: isinstance(e, Derivative) and e.expr.func.__name__ == 'identity',
            lambda e: sp.Integer(1)
        )
        expr = expr.replace(
            lambda e: isinstance(e, Derivative) and e.expr.func.__name__ == 'sigmoid',
            lambda e: (lambda s: s * (1 - s))(Function('sigmoid')(e.expr.args[0]))
        )
        expr = expr.replace(
            lambda e: isinstance(e, Derivative) and e.expr.func.__name__ == 'absv',
            lambda e: Function('sign')(e.expr.args[0])
        )
        return expr

    def _expr_to_jac_str(self, expr, sym_to_y_idx: dict, past_sym_to_str: dict):
        """Convert a symbolic Jacobian entry to a backend code string.

        Parameters
        ----------
        expr : sympy.Expr
        sym_to_y_idx : dict
            ``{state_sym: int_or_tuple_idx}`` — maps state-variable symbols to their
            position in the flat state vector ``y``.
        past_sym_to_str : dict
            ``{fresh_past_sym: code_string}`` — maps past-state placeholders to the
            code that evaluates them (e.g. ``'_yhist_0p5[0]'``).

        Returns
        -------
        str or None
            ``None`` signals that unevaluated ``Derivative`` nodes remain and a
            numerical fallback should be used for this entry.
        """
        import sympy as sp
        from sympy import Derivative

        # resolve known analytical derivative rules (sigmoid, absv, …)
        expr = self._resolve_derivatives(expr)

        if expr.atoms(Derivative):
            return None

        start = self.backend._start_idx

        # build placeholder substitution: state sym / past sym → unique temp sym
        subs: dict = {}
        ph_to_code: dict = {}
        for i, (sym, idx) in enumerate(sym_to_y_idx.items()):
            ph = sp.Symbol(f'_ypl{i}_')
            subs[sym] = ph
            if isinstance(idx, tuple):
                ph_to_code[str(ph)] = f'y[{idx[0]+start}:{idx[1]}]'
            else:
                ph_to_code[str(ph)] = f'y[{idx+start}]'
        for i, (psym, code_str) in enumerate(past_sym_to_str.items()):
            ph = sp.Symbol(f'_ppl{i}_')
            subs[psym] = ph
            ph_to_code[str(ph)] = code_str

        expr_subst = expr.subs(subs)
        expr_str = str(expr_subst)

        # replace placeholders with actual code strings (longest first to avoid
        # partial substring collisions)
        for ph_str in sorted(ph_to_code.keys(), key=len, reverse=True):
            expr_str = expr_str.replace(ph_str, ph_to_code[ph_str])

        return expr_str

    def clear(self) -> None:
        """Deletes build directory and removes all compute graph nodes
        """

        # delete network nodes and variables from the compute graph
        for n in list(self.nodes.keys()):
            self.remove_subgraph(n)
        self.var_updates.clear()
        self._state_var_indices.clear()
        self._eq_nodes.clear()

        # clear code generator
        self.backend.clear()

    def _to_str(self):

        # preparations
        code_gen = self.backend

        # extract state variable from state vector
        rhs_indices_str = []
        for var in self.state_vars:

            # extract index of variable in state vector
            idx = (self._state_var_indices[var],)

            # extract state variable from state vector
            rhs_idx, _ = code_gen.create_index_str(idx)
            code_gen.add_var_update(lhs=self.get_var(var), rhs=f"y{rhs_idx}")
            rhs_indices_str.append(idx)

        # get equation string and argument list for each non-DE node at the end of the compute graph hierarchy
        func_args2, delete_args1 = self._generate_update_equations(differential_equations=False)
        code_gen.add_linebreak()

        # get equation string and argument list for each DE node at the end of the compute graph hierarchy
        func_args1, delete_args2 = self._generate_update_equations(differential_equations=True, indices=rhs_indices_str)

        # remove unnecessary function arguments
        func_args = func_args1 + func_args2
        for arg in delete_args1 + delete_args2:
            while arg in func_args:
                func_args.pop(func_args.index(arg))

        return func_args, code_gen

    def _generate_update_equations(self, differential_equations: bool, indices: list = None) -> tuple:

        code_gen = self.backend

        # extract relevant compute graph nodes and bring them into the correct order
        nodes = self.var_updates['DEs' if differential_equations else 'non-DEs']
        nodes, updates, def_vars, undef_vars = self._sort_var_updates(nodes=nodes,
                                                                      differential_equations=differential_equations)

        # collect right-hand side expression and all input variables to these expressions
        func_args, expressions, var_names, rhs_shapes, lhs_indices = undef_vars, [], [], [], []
        for node, update in zip(nodes, updates):

            # collect shape of the right-hand side variable
            v = self.get_var(update)
            try:
                v_eval = self.eval_node(update)
                v.set_value(v_eval)
            except IndexError:
                pass
            rhs_shapes.append(v.shape)

            # collect expression and variables of right-hand side of equation
            expr_args, expr = self._node_to_expr(update)
            func_args.extend(expr_args)
            expr_str, expr_args, _, _ = self._expr_to_str(expr, apply=True)
            func_args.extend(expr_args)
            expressions.append(expr_str)

            # process left-hand side of equation
            var = self.get_var(node)
            if isinstance(var, ComputeOp):

                # process indexing of left-hand side variable
                idx_args, lhs = self._node_to_expr(node)
                if lhs.args[0].name not in def_vars:
                    idx_args.append(lhs.args[0].name)
                func_args.extend(idx_args)
                _, idx_args, lhs_var, idx = self._expr_to_str(lhs, apply=False)
                func_args.extend(idx_args)

            else:

                # process normal update of left-hand side variable
                lhs_var = var.name
                idx = None

            var_names.append(lhs_var)
            lhs_indices.append(idx)

        # add the left-hand side assignments of the collected right-hand side expressions to the code generator
        if differential_equations:

            # differential equation (DE) update
            if not indices:
                raise ValueError('State variables need to be stored in a single state vector, for which the indices '
                                 'have to be passed to this method.')

            add_args = self._generate_vecfield(code_gen, indices, expressions, rhs_shapes, var_names)
            func_args = add_args + func_args

        else:

            # non-differential equation update
            if indices:
                raise ValueError('Indices to non-state variables should be defined in the respective equations, not'
                                 'be passed to this method.')
            indices = lhs_indices

            # non-DE update stored in a single variable
            for target_var, expr, idx, shape in zip(var_names, expressions, indices, rhs_shapes):
                try:
                    idx = self.get_var(idx)
                except (KeyError, AttributeError, TypeError):
                    pass
                code_gen.add_var_update(lhs=self.get_var(target_var), rhs=expr, lhs_idx=idx, rhs_shape=shape)

        return func_args, def_vars

    def _generate_vecfield(self, code_gen, indices: list, expressions: list, rhs_shapes: list, lhs_vars: list) -> list:

        # DE updates stored in a state-vector
        dy = self.get_var("dy")
        for idx, expr, shape in zip(indices, expressions, rhs_shapes):
            code_gen.add_var_update(lhs=dy, rhs=expr, lhs_idx=idx, rhs_shape=shape)

        # add rhs var to function arguments
        return ['dy']

    def _generate_vecfield_var(self, state_vec: np.ndarray, dtype: str) -> str:
        key, _ = self.add_var(label='dy', vtype='state_var', value=np.zeros_like(state_vec), dtype=dtype)
        return key

    def _generate_func_tail(self, code_gen, vecfield_key: str):
        code_gen.generate_func_tail(rhs_var=vecfield_key)

    def _node_to_expr(self, n: str, **kwargs) -> tuple:

        expr_args = []
        node = self.get_var(n)

        # case I: node is a mathematical operation and its inputs need to be treated
        try:

            # process node inputs
            expr_info = {self.get_var(inp).symbol: self._node_to_expr(inp, **kwargs) for inp in self.predecessors(n)}

            # replace old inputs with its processed versions
            expr = node.expr
            for expr_old, (args, expr_new) in expr_info.items():
                if expr_old != expr_new:
                    expr = expr.replace(expr_old, expr_new)
                expr_args.extend(args)

            # replace generic function calls with the backend-specific function calls
            try:
                expr_old = expr.func.__name__
                func_info = self.get_op(expr_old, shape=node.shape)
                expr = expr.replace(expr.func, Function(func_info['call']))
            except (AttributeError, KeyError):
                pass

        # case II: node is a simple variable or constant
        except AttributeError:

            # add constants to the expression arguments list
            if node.is_constant:
                expr_args.append(n)
                expr = node.symbol
            elif 'dummy_constant' in node.name:
                val = float(np.squeeze(node.value))
                expr = Symbol(str(val))
            else:
                expr = node.symbol

        return expr_args, expr

    def _expr_to_str(self, expr: Any, expr_str: str = None, apply: bool = True, **kwargs) -> tuple:

        # preparations
        ###############

        # initializations
        index_args = []
        func = ""
        idx = ""

        # ensure expression string exists
        if not expr_str:
            expr_str = str(expr)

        # transform expression arguments into strings
        expr_args = []
        for arg in expr.args:
            expr_part, args, _, _ = self._expr_to_str(arg, **kwargs)
            expr_str = expr_str.replace(str(arg), expr_part)
            index_args.extend(args)
            expr_args.append(expr_part)
        var = str(expr_args[0]) if expr.args else ""

        # process indexing operations
        #############################

        if 'index_1d(' in expr_str:

            # replace `index` calls with brackets-based indexing
            idx = self._get_var_idx(idx=(expr.args[1],), args=index_args, apply=apply, **kwargs)
            func = 'index_1d'

        elif 'index_2d(' in expr_str:

            # replace `2d_index` calls with brackets-based indexing
            idx = self._get_var_idx(idx=(expr.args[1], expr.args[2]), args=index_args, apply=apply, **kwargs)
            func = 'index_2d'

        elif 'index_range(' in expr_str:

            # replace `range_index` calls with brackets-based indexing
            idx = self._get_var_idx(idx=((expr.args[1], expr.args[2]),), args=index_args, apply=apply, **kwargs)
            func = 'index_range'

        elif 'index_axis(' in expr_str:

            # replace `axis_index` calls with brackets-based indexing
            if len(expr.args) < 2:
                idx = self._get_var_idx(idx=(':',), args=index_args, apply=apply, **kwargs)
            else:
                idx = self._get_var_idx(args=index_args, apply=apply,
                                        idx=tuple([':' for _ in range(expr.args[2])] + [f"{expr.args[1]}"]), **kwargs)
            func = "index_axis"

        # either apply the above indexing calls or return them
        if func and apply:
            replacement = self.backend.finalize_idx_str(var=self.get_var(var), idx=idx)
            expr_str = self._process_func_call(expr=expr_str, func=func, replacement=replacement)

        # handle other function calls
        #############################

        if 'identity(' in expr_str:

            # replace `no_op` calls with first argument to the function call
            expr_str = self._process_func_call(expr=expr_str, func='identity', replacement=var)

        if 'past(' in expr_str:

            # replace past calls with the delayed version of the backend variable
            try:
                delay = self.get_var(expr.args[1].name)
            except AttributeError:
                delay = float(expr.args[1])
            replacement = self._get_var_hist(var=var, delay=delay)
            expr_str = self._process_func_call(expr=expr_str, func="past", replacement=replacement)

        # backend-specific function call adjustments
        expr_str = self.backend.expr_to_str(expr_str, expr.args)

        return expr_str, index_args, var, idx

    def _process_var_update(self, var: str, update: str) -> tuple:

        # extract nodes
        lhs = self.get_var(var)
        rhs = self.eval_node(update)

        # extract common shape
        if lhs.shape == rhs.shape:
            return lhs, rhs
        try:
            rhs = rhs.reshape(lhs.shape)
            return lhs, rhs
        except ValueError:
            raise ValueError(
                f"Shapes of state variable {var} and its right-hand side update {self.get_var(update).expr} do not"
                " match.")

    def _sort_var_updates(self, nodes: dict, differential_equations: bool = True) -> tuple:

        # case I: for differential equations, do not perform any sorting
        if differential_equations:
            return list(nodes.keys()), list(nodes.values()), [], []

        # case II: for non-differential equations, sort them according to their graph connections
        #########################################################################################

        # step 1: ensure lhs-indexing operations are considered as well
        node_names, node_keys = [], []
        for node in nodes:
            n = self.get_var(node)
            if type(n) is ComputeVar:
                node_names.append(node)
            else:
                node_names.append(list(self._get_inputs(node))[-1])
            node_keys.append(node)

        keys, values, defined_vars, undefined_vars = [], [], [], []
        n_nodes = len(nodes)
        while nodes:

            for node, update in nodes.copy().items():

                # go through node inputs and check whether it depends on other equations to be evaluated first
                dependent, inp = False, ""
                for inp in self._get_inputs(update):
                    if inp in node_names:
                        idx = node_names.index(inp)
                        if node_keys[idx] != node:
                            dependent = True
                            break

                # decide whether this equation can be evaluated now
                if dependent:
                    continue
                else:
                    idx = node_keys.index(node)
                    n = node_names.pop(idx)
                    node_keys.pop(idx)
                    nodes.pop(node)
                    keys.append(node)
                    values.append(update)
                    if isinstance(self.get_var(node), ComputeVar) and n not in undefined_vars:
                        defined_vars.append(n)
                    elif n not in undefined_vars:
                        undefined_vars.append(n)

            # check whether the algorithm is stuck
            n_nodes_new = len(nodes)
            if n_nodes_new == n_nodes:
                break
            else:
                n_nodes = n_nodes_new

        # add mutually depended nodes
        if nodes:
            node_keys = list(nodes.keys())
            keys.extend(node_keys)
            values.extend(list(nodes.values()))
            for n in node_names:
                if n not in undefined_vars:
                    undefined_vars.append(n)

        return keys, values, defined_vars, undefined_vars

    def _get_inputs(self, n: str):

        inputs = []
        for inp in self.predecessors(n):
            inputs.extend([inp] if isinstance(self.get_var(inp), ComputeVar) else self._get_inputs(inp))
        return inputs

    def _get_var_idx(self, idx: tuple, args: list, apply: bool = True, **kwargs):

        # collect indexing variables where necessary
        new_idx = []
        for idx_tmp in idx:
            try:
                v = self.get_var(idx_tmp.name if type(idx_tmp) is Symbol else idx_tmp)
                new_idx.append(v)
            except KeyError:
                new_idx.append(idx_tmp)

        # turn index into a backend-specific string
        idx_str, new_vars = self.backend.create_index_str(tuple(new_idx), apply=apply, **kwargs)

        # add new variables to graph and index arguments
        for key, v in new_vars.items():
            if key != self.backend.idx_dummy_var:
                vlabel, _ = self.add_var(label='index', value=v, vtype='constant')
                idx_str = idx_str.replace(key, vlabel)
                args.append(vlabel)

        return idx_str

    def _get_var_hist(self, var: str, delay: Union[ComputeVar, float]):

        if var not in self._state_var_hist:
            self._state_var_hist[var] = dict()
        if delay not in self._state_var_hist[var]:
            var_hist = f'{var}_hist{len(self._state_var_hist[var])}'
            self.add_var(var_hist, value=self.get_var(var).value, vtype='variable')
            self._state_var_hist[var][delay] = var_hist
        else:
            var_hist = self._state_var_hist[var][delay]
        return var_hist

    def _prune(self):

        # remove all subgraphs that contain constants only
        for n in [node for node, out_degree in self.out_degree if out_degree == 0]:
            if self.get_var(n).is_constant and n not in self._eq_nodes:
                self.remove_subgraph(n)

        # remove all unconnected nodes
        for n in [node for node, out_degree in self.out_degree if out_degree == 0]:
            if self.in_degree(n) == 0 and n not in self._eq_nodes:
                self.remove_node(n)

    def _generate_unique_label(self, label: str) -> str:

        if label == "t":
            return label
        if label in self._node_names:
            n = self._node_names[label]
            if n == 0:
                label_new = f"{label}_v1"
            else:
                label_new = f"{label}_v{n + 1}"
            self._node_names[label] += 1
        else:
            label_new = label
            self._node_names[label] = 0
        return label_new

    @staticmethod
    def _index_state_var(y: np.ndarray, idx: Union[int, tuple, list]) -> np.ndarray:
        if type(idx) is tuple and idx[1] - idx[0] == 1:
            idx = (idx[0],)
        elif type(idx) is int:
            idx = (idx,)
        return y[:, idx] if len(idx) == 1 else y[:, idx[0]:idx[1]]

    @staticmethod
    def _process_func_call(expr: str, func: str, replacement: str):

        # identify start and end of the function call
        start = expr.find(f"{func}(")
        end = expr[start:].find(')') + 1

        # replace part in expression string
        return expr.replace(expr[start:start + end], replacement)


class ComputeGraphBackProp(ComputeGraph):

    def __init__(self, backend: str, **kwargs):

        super().__init__(backend, **kwargs)
        self._vecfield_vars = []
        self._vecfield_var_str = ""

    def _generate_vecfield_var(self, state_vec: np.ndarray, dtype: str):
        return ""

    def _generate_func_tail(self, code_gen, vecfield_key: str):
        code_gen.generate_func_tail(rhs_var=self._vecfield_var_str)

    def _generate_vecfield(self, code_gen, indices: list, expressions: list, rhs_shapes: list, lhs_vars: list) -> list:
        for lhs, expr in zip(lhs_vars, expressions):
            lhs_var = f"delta_{lhs}"
            code_gen.add_code_line(f"{lhs_var} = {expr}")
            self._vecfield_vars.append(lhs_var)
        if len(self._vecfield_vars) > 1:
            op_dict = self.backend.get_op("concatenate")
            self._vecfield_var_str = f"{op_dict['call']}([{','.join(v for v in self._vecfield_vars)}], 0)"
        else:
            self._vecfield_var_str = self._vecfield_vars.pop()
        return []
