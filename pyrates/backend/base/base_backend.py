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
- Torch: TorchBackend.
- Fortran: FortranBackend (experimental).
- Julia: JuliaBackend.
- Matlab: MatlabBackend.

"""

# pyrates internal _imports
from ..computegraph import ComputeVar
from .base_funcs import base_funcs
from .. import PyRatesException

# external _imports
from typing import Optional, Dict, List, Union, Tuple, Callable, Iterable
import numpy as np
import os, sys, importlib, hashlib, types as _types
from shutil import rmtree


# ---------------------------------------------------------------------------
# Module-level cache for compiled RHS modules, keyed by SHA-256 of the source
# string passed to ``compile()``.  Lets parameter sweeps and the
# ``get_run_func`` / ``get_jacobian_func`` paths skip the compile+exec round
# trip when the same model is regenerated identically (which is the common
# case during optimisation loops).  Cleared explicitly via
# ``clear_compile_cache()`` below — there is no automatic eviction.
# ---------------------------------------------------------------------------
_compiled_module_cache: Dict[str, _types.ModuleType] = {}


def clear_compile_cache() -> None:
    """Drop all cached compiled RHS modules.

    Call this if you have generated many distinct models in a long-running
    process and want to reclaim memory.  Has no effect on already-returned
    callables; only future :meth:`BaseBackend.generate_func` calls are
    affected.
    """
    _compiled_module_cache.clear()


# Helper Functions and Classes
##############################


import bisect


class DDEHistory:
    """Callable history buffer for delay-differential equations.

    Stores (time, state) pairs and returns linearly-interpolated past state.
    Pre-history (t <= t0) always returns the initial condition.

    Implementation
    --------------
    The previous list-of-ndarrays implementation (PyRates <= 1.1.x)
    allocated a fresh ``ndarray`` on every :meth:`update` call via
    ``y.copy()`` — for a 100k-step simulation that was 100k small heap
    allocations.  We replace the ``_y`` storage with a single
    pre-allocated 2-D numpy buffer that holds one row per step; row
    assignment ``self._y[i] = y`` still constitutes a copy, so the
    "update takes ownership of its y argument" contract is unchanged.

    ``_t`` stays as a Python list because :code:`bisect.bisect_right` on
    a list is noticeably faster than :code:`np.searchsorted` on the
    equivalent numpy view when called once per RHS evaluation in the
    integration loop (per-call CPython overhead is the dominant cost
    here, not the underlying binary search).

    Parameters
    ----------
    y0
        Initial state vector.  Defines the ``shape`` and ``dtype`` of all
        subsequent history rows.
    t0
        Initial time.  History returns ``y0`` for any query ``t <= t0``.
    max_steps
        Optional hard cap on the number of stored steps.  If given, the
        buffer is allocated once at this size and :meth:`update` raises
        ``IndexError`` past the cap.  If ``None`` (default) the buffer
        grows geometrically.
    """

    _INITIAL_CAPACITY = 1024
    _GROW_FACTOR = 2

    def __init__(self, y0: np.ndarray, t0: float = 0.0,
                 max_steps: Optional[int] = None):
        y0 = np.asarray(y0)
        if max_steps is None:
            capacity = self._INITIAL_CAPACITY
            self._growable = True
        else:
            capacity = max(int(max_steps), 1)
            self._growable = False

        self._t = [float(t0)]                                    # bisect-friendly
        self._y = np.empty((capacity,) + y0.shape, dtype=y0.dtype)  # pre-allocated rows
        self._y[0] = y0
        self._n = 1

    def update(self, t: float, y: np.ndarray) -> None:
        """Record state ``y`` at time ``t``.

        ``y`` is copied into the pre-allocated row buffer; the caller may
        free or overwrite its own ``y`` after this call returns.
        """
        if self._n >= len(self._y):
            if self._growable:
                self._grow()
            else:
                raise IndexError(
                    f"DDEHistory: exceeded max_steps={len(self._y)}; "
                    "increase the bound or omit max_steps to allow growth."
                )
        self._t.append(float(t))
        self._y[self._n] = y       # row assignment copies y into the buffer
        self._n += 1

    def _grow(self) -> None:
        old_cap = len(self._y)
        new_cap = old_cap * self._GROW_FACTOR
        new_y = np.empty((new_cap,) + self._y.shape[1:], dtype=self._y.dtype)
        new_y[:self._n] = self._y[:self._n]
        self._y = new_y

    def __call__(self, t: float) -> np.ndarray:
        t = float(t)
        if t <= self._t[0]:
            return self._y[0]
        if t >= self._t[-1]:
            return self._y[self._n - 1]
        idx = bisect.bisect_right(self._t, t) - 1
        t0_ = self._t[idx]
        t1_ = self._t[idx + 1]
        alpha = (t - t0_) / (t1_ - t0_)
        return self._y[idx] + alpha * (self._y[idx + 1] - self._y[idx])


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

    # ----------------------------------------------------------------------
    # Class attribute: which `solver=` values does this backend accept?
    # Subclasses override (or extend) this tuple.  Validated early via
    # ``_validate_solver`` so the user sees a clear error instead of the
    # current "the call silently took a wrong branch and produced numpy
    # output" failure mode.
    # ----------------------------------------------------------------------
    SUPPORTED_SOLVERS: Tuple[str, ...] = ('euler', 'heun', 'scipy')

    # ----------------------------------------------------------------------
    # Class attribute: ops whose `def` strings must NOT be appended to the
    # generated module's helper-funcs list (they are language primitives or
    # pure-Python no-ops handled directly by the parser).  Subclasses can
    # extend this with e.g. ``_no_funcs = BaseBackend._no_funcs + ("foo",)``.
    # ----------------------------------------------------------------------
    _no_funcs: Tuple[str, ...] = ("identity", "index_1d", "index_2d",
                                  "index_range", "index_axis")

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
        # `_no_funcs` lives on the class (see top of class body).  Do not
        # shadow it here — subclasses extend the class attribute.

        # Tracks which ComputeVars have already had `_start_idx` baked into
        # their `.value` by `_process_idx`.  Prevents double-application when
        # the same ComputeVar is processed more than once (review §4.3).
        # Keyed by ``id(ComputeVar)`` — safe because ComputeVars live as long
        # as the backend itself (owned by the ComputeGraph that wraps it).
        self._offsetted_var_ids: set = set()

        # file-creation-related attributes
        fdir, *fname = self.get_fname(kwargs.pop('file_name', 'pyrates_run'))
        cwdir = os.getcwd()
        sys.path.append(cwdir)
        if fdir:
            fdir = f"{cwdir}/{fdir}"
            sys.path.append(fdir)
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
        result = np.asarray(v.value, dtype=dtype)
        # Squeeze single-element constants to 0-d scalars.
        # PyRates stores scalar parameters internally as shape (1,) but
        # numpy 2.3+ raises an error when a (1,) array is assigned to a scalar
        # state-vector slot (e.g. dy[i] = (1,)_param * expr).
        if result.shape == (1,) and v.vtype == 'constant':
            result = result.squeeze()
        return result

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

        lhs_str = lhs.name
        if lhs_idx:
            idx, _ = self.create_index_str(lhs_idx, apply=True)
            lhs_str = f"{lhs_str}{idx}"
        indexed = bool(lhs_idx) or bool(rhs_shape)
        self.add_code_line(self._format_assignment(lhs_str, rhs, indexed))

    def _format_assignment(self, lhs: str, rhs: str, indexed: bool) -> str:
        """Render a single ``lhs = rhs`` assignment for the target language.

        Subclasses override this hook to splice in language-specific syntax
        (Julia's broadcast prefix ``@.``, Matlab's ``vectorize`` + trailing
        ``;``, …) instead of the old pop-and-rewrite pattern that
        ``super().add_var_update`` then ``self.code.pop() / line.split(' = ')``
        used to implement.  ``indexed`` is True iff the assignment writes to
        an index expression (``lhs[idx] = rhs``) or the rhs has non-scalar
        shape — i.e. the cases where broadcasting matters.
        """
        return f"{lhs} = {rhs}"

    def add_var_hist(self, lhs: str, delay: Union[ComputeVar, float], state_idx: str,
                     dt: Optional[float] = None, dt_adapt: bool = True, **kwargs):
        idx = self._process_idx(state_idx)
        d = self._process_delay(delay)
        if dt is not None and not dt_adapt:
            self.add_code_line(f"{lhs} = hist(t*{dt:.10e}-{d})[{idx}]")
        else:
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
                           add_hist_func: Optional[bool] = None):
        """Generate the function header for the RHS file.

        ``add_hist_func``
            Whether to include the ``hist`` callable in the generated function
            signature.  If ``None`` (default), falls back to the backend's
            ``add_hist_arg`` constructor flag — making that flag the single
            source of truth.  Callers that already know whether the model is a
            DDE (e.g. ``ComputeGraph.to_func``) pass the resolved bool
            explicitly and skip the fallback.
        """
        if add_hist_func is None:
            add_hist_func = self.add_hist_arg


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

        # Write the source to disk first (so users can still inspect / debug it
        # even on a cache hit) — file IO is cheap compared to compile().
        if to_file:
            file = f'{self.fdir}/{self._fname}' if self.fdir else self._fname
            src_path = f'{file}{self._fend}'
            with open(src_path, 'w') as f:
                f.writelines(func_str)
        else:
            src_path = f'<pyrates:{self._fname}>'

        # Consult the SHA-256-keyed module cache.  A hit means we already have
        # a fully-compiled-and-executed module for this exact source string;
        # we can pull the function object out of its namespace directly and
        # skip both compile() and exec().  See module-level docstring for
        # ``_compiled_module_cache``.
        cache_key = hashlib.sha256(func_str.encode('utf-8')).hexdigest()
        _mod = _compiled_module_cache.get(cache_key)
        if _mod is None:
            _mod = _types.ModuleType(self._fname)
            _mod.__file__ = src_path
            # Compile from the in-memory source string and exec into the fresh
            # module, bypassing the .pyc bytecode cache entirely.  A stale
            # .pyc can persist when clear() removes the .py but not
            # __pycache__, and the next write lands in the same second, making
            # the source mtime appear unchanged to Python's cache validator.
            exec(compile(func_str, src_path, 'exec'), _mod.__dict__)
            _compiled_module_cache[cache_key] = _mod

        # Refresh sys.modules so subsequent introspection (e.g. tracebacks)
        # finds the right module under self._fname.  We always replace the
        # entry — the cached module may have been registered under a previous
        # _fname or have been removed by .clear().
        if to_file:
            sys.modules[self._fname] = _mod

        rhs_eval = _mod.__dict__[func_name]
        return self._apply_decorator(rhs_eval, **kwargs)

    @staticmethod
    def _apply_decorator(rhs_eval: Callable, **kwargs) -> Callable:
        """Optionally wrap the generated function with a user-supplied decorator.

        Looks for ``decorator`` (callable) and ``decorator_kwargs`` (dict) in
        ``kwargs``; pops them out and applies ``decorator(rhs_eval,
        **decorator_kwargs)`` when present.  Returns ``rhs_eval`` unchanged
        otherwise.  Centralised here so backend subclasses with their own
        ``generate_func`` don't each have to repeat the same four lines.
        """
        decorator = kwargs.pop('decorator', None)
        if decorator:
            decorator_kwargs = kwargs.pop('decorator_kwargs', dict())
            rhs_eval = decorator(rhs_eval, **decorator_kwargs)
        return rhs_eval

    def run(self, func: Callable, func_args: tuple, T: float, dt: float, dts: float, solver: str, **kwargs) -> tuple:

        # initial values
        t0 = func_args[0]
        y0 = func_args[1]

        # use a safer way to generate time points (endpoint=False ensures times match Euler step indices)
        step = dts if dts else dt
        n_time_points = round(T/step)
        times = np.linspace(0.0, T, num=n_time_points, endpoint=False)

        # perform simulation
        results = self._solve(solver=solver, func=func, args=func_args[2:], T=T, dt=dt, dts=dts, y0=y0, t0=t0,
                              times=times, **kwargs)

        return results, times

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
    def get_hist_func(y: np.ndarray, t0: float = 0.0) -> DDEHistory:
        return DDEHistory(y, t0=t0)

    def _get_func_info(self, name: str, **kwargs):
        return self._funcs[name]

    def _process_idx(self, idx: Union[Tuple[int, int], int, str, ComputeVar], **kwargs) -> str:
        if type(idx) is ComputeVar:
            # Idempotent offset: bake `self._start_idx` into `idx.value` only
            # the first time we see this particular ComputeVar.  The previous
            # implementation re-added the offset on every call, which forced
            # Julia (and indirectly Matlab) to temporarily toggle
            # `self._start_idx = 0` around calls that could touch an already
            # processed var — fragile and easy to break (review §4.3).
            if self._start_idx and id(idx) not in self._offsetted_var_ids:
                idx.set_value(idx.value + self._start_idx)
                self._offsetted_var_ids.add(id(idx))
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

    def _validate_solver(self, solver: str) -> None:
        """Raise a helpful error if the requested solver is not supported.

        Called by every ``_solve`` override (in this class and subclasses) so
        users get a clear "solver X is not supported by backend Y; supported
        are Z" message instead of the previous fall-through into a generic
        ``PyRatesException`` (or, worse, a silent dispatch into a method
        that happened to share a prefix).
        """
        if solver not in self.SUPPORTED_SOLVERS:
            raise PyRatesException(
                f"Backend `{type(self).__name__}` does not support solver "
                f"`{solver}`. Supported solvers: {list(self.SUPPORTED_SOLVERS)}."
            )

    def _solve(self, solver: str, func: Callable, args: tuple, T: float, dt: float, dts: float, y0: np.ndarray,
               t0: np.ndarray, times: np.ndarray, **kwargs) -> np.ndarray:

        self._validate_solver(solver)

        if solver == 'euler':
            return self._solve_euler(func, args, T, dt, dts, y0, t0)

        if solver == 'heun':
            return self._solve_heun(func, args, T, dt, dts, y0, t0)

        # solver == 'scipy'
        if len(args) > 0 and isinstance(args[0], DDEHistory):
            return self._solve_scipy_dde(func, args, T, dt, y0, t0, times, **kwargs)
        return self._solve_scipy(func, args, T, dt, y0, t0, times, **kwargs)

    def _add_func_call(self, name: str, args: Iterable, return_var: str = 'dy'):
        self.add_code_line(f"def {name}({','.join(args)}):")

    @staticmethod
    def _solve_euler(func: Callable, args: tuple, T: float, dt: float, dts: float, y: np.ndarray, t0):

        # preparations for fixed step-size integration
        idx = 0
        steps = int(np.round(T / dt))
        store_steps = int(np.round(T / dts))
        store_step = int(np.round(dts / dt))
        # state_rec is fully overwritten row-by-row in the loop below before any
        # row is read, so np.empty is safe and avoids the zero-fill (which
        # dominates startup for large state vectors).
        state_rec = np.empty((store_steps, y.shape[0]) if y.shape else (store_steps, 1), dtype=y.dtype)
        has_dde = len(args) > 0 and isinstance(args[0], DDEHistory)

        # solve ivp for forward Euler method.  Storage cadence is driven by the
        # iteration counter `i`, not by the wall-clock step number — using
        # ``step % store_step == t0`` (the previous formulation) silently
        # produces zero stored samples whenever ``t0 >= store_step``.
        for i in range(steps):
            if i % store_step == 0:
                state_rec[idx, :] = y
                idx += 1
            step = i + t0
            rhs = func(step, y, *args)
            y += dt * rhs
            if has_dde:
                args[0].update((i + 1) * dt, y)

        return state_rec

    @staticmethod
    def _solve_heun(func: Callable, args: tuple, T: float, dt: float, dts: float, y: np.ndarray, t0):

        # preparations for fixed step-size integration
        idx = 0
        steps = int(np.round(T / dt))
        store_steps = int(np.round(T / dts))
        store_step = int(np.round(dts / dt))
        # state_rec is fully overwritten row-by-row in the loop below before any
        # row is read, so np.empty is safe and avoids the zero-fill (which
        # dominates startup for large state vectors).
        state_rec = np.empty((store_steps, y.shape[0]) if y.shape else (store_steps, 1), dtype=y.dtype)
        has_dde = len(args) > 0 and isinstance(args[0], DDEHistory)

        # solve ivp via Heun's method.  See `_solve_euler` for the rationale
        # behind the iteration-counter-based storage condition.
        for i in range(steps):
            if i % store_step == 0:
                state_rec[idx, :] = y
                idx += 1
            step = i + t0
            rhs = func(step, y, *args)
            y_0 = y + dt * rhs
            y += dt/2 * (rhs + func(step, y_0, *args))
            if has_dde:
                args[0].update((i + 1) * dt, y)

        return state_rec

    @staticmethod
    def _solve_scipy_dde(func: Callable, args: tuple, T: float, dt: float, y: np.ndarray, t0: np.ndarray,
                         times: np.ndarray, **kwargs):

        from scipy.integrate import ode

        hist = args[0]
        kwargs.pop('method', None)

        def rhs(t, y_):
            return func(t, y_, *args)

        solver = ode(rhs).set_integrator('dopri5', first_step=dt, nsteps=50000)
        solver.set_initial_value(y, float(t0))

        def solout(t, y_):
            # DDEHistory.update copies y_ into its pre-allocated buffer.
            hist.update(t, y_)
            return 0

        solver.set_solout(solout)

        state_rec = np.zeros((len(times), y.shape[0]), dtype=y.dtype)
        for i, t_out in enumerate(times):
            if not solver.successful():
                break
            solver.integrate(t_out)
            state_rec[i, :] = solver.y

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
