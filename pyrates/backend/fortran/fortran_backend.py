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
import subprocess
import sys
import os
import numpy as np
# Note: `from numpy import f2py` is deferred to FortranBackend.__init__.
# In numpy >= 2.0 the f2py module pulls in distutils / meson eagerly, which
# adds noticeable startup latency and an extra hard dependency that should
# only matter for users who actually instantiate this backend.
from typing import Optional, Dict, List, Union, Tuple, Iterable, Callable

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

        # Lazy availability check: importing f2py at module-top forces the
        # numpy.f2py / distutils / meson chain to load for every user of
        # pyrates, including those who never touch the fortran backend.
        # We do the import here so that:
        #   (a) `import pyrates` (and even `from pyrates.backend.fortran ...`)
        #       stays cheap when numpy.f2py is unavailable;
        #   (b) instantiating FortranBackend without f2py raises a clear
        #       ImportError instead of a cryptic subprocess failure later.
        try:
            from numpy import f2py  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "FortranBackend requires `numpy.f2py`. On numpy >= 2.0 you may "
                "also need `pip install meson meson-python ninja`."
            ) from e

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
        self._imports.append("complex :: I = (0.0, 1.0)")

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
                           add_hist_func: Optional[bool] = None):

        # resolve default from the backend-level flag (see BaseBackend docstring)
        if add_hist_func is None:
            add_hist_func = self.add_hist_arg

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

        file = f'{self.fdir}/{self._fname}{self._fend}' if self.fdir else f'{self._fname}{self._fend}'

        # generate the final string representing the function file
        auto_compatible = kwargs.pop('auto', False)
        if auto_compatible:

            # case I: generate the auto fortran source + one ``c.<scenario>``
            # file per requested scenario (defaults to a single ``c.ivp``).
            # ``_generate_auto_files`` consumes ``auto_constants`` (scenario
            # names), ``auto_parnames`` / ``auto_unames``, and any direct
            # auto-07p constant overrides from kwargs.
            func_file, constants_files = self._generate_auto_files(
                func_name=func_name, func_args=func_args, state_vars=state_vars, **kwargs)

            # write each scenario's constants file
            build_dir = f"{self.fdir}/" if self.fdir else ""
            # ``auto_constants_file`` is preserved as a legacy override of the
            # name only when a single scenario was requested.  New code should
            # use ``auto_constants=('eq', 'lc', ...)`` instead.
            legacy_name = kwargs.pop('auto_constants_file', None)
            if legacy_name and len(constants_files) == 1:
                only_key = next(iter(constants_files))
                constants_files = {legacy_name: constants_files[only_key]}
            for scen_name, const_text in constants_files.items():
                with open(f'{build_dir}c.{scen_name}', 'wt') as cfile:
                    cfile.write(const_text)

        else:

            # case II: generate a standard fortran function string
            func_file = self.generate()

        # write function to file
        with open(file, 'w') as f:
            f.writelines(func_file)
            f.close()

        # compile fortran function via f2py.  Use sys.executable so we hit
        # the same interpreter (and therefore the same numpy) the caller is
        # running, drop shell=True (small command-injection surface around
        # `self._fname`), and surface any compiler errors as a Python
        # exception instead of letting the next `import` line fail with an
        # opaque ImportError.
        completed = subprocess.run(
            [sys.executable, '-m', 'numpy.f2py', '-c', '-m', self._fname, file],
            capture_output=True, text=True,
        )
        if completed.returncode != 0:
            stderr_tail = (completed.stderr or '').strip().splitlines()
            tail = '\n'.join(stderr_tail[-30:]) if stderr_tail else '<no stderr>'
            raise RuntimeError(
                f"f2py compilation of {file} failed (exit {completed.returncode}). "
                f"Last lines of stderr:\n{tail}"
            )

        # import function from temporary file
        exec(f"from {self._fname} import {self._fname}", globals())
        exec(f"rhs_eval = {self._fname}.{func_name}", globals())
        rhs_eval = globals().pop('rhs_eval')

        rhs_eval = self._apply_decorator(rhs_eval, **kwargs)

        if not to_file:
            os.remove(file)

        return rhs_eval

    def register_vars(self, variables: list):
        for v in variables:
            if v.name not in self._var_declaration_info:
                self._var_declaration_info[v.name] = v

    def clear(self) -> None:
        """Removes all layers, variables and operations from graph. Deletes build directory.

        Also removes the auto-07p artefacts generated by ``_generate_auto_files``:
        every ``c.<scenario>`` file in the build dir, plus any pycobi-style
        ``s.<name>`` / ``b.<name>`` / ``d.<name>`` / ``.lab``/``.dat`` output
        files matching ``self._fname``.  Previously only ``c.ivp`` was removed,
        which orphaned all the other ``c.*`` files once we started generating
        multiple scenarios.
        """

        # delete fortran-specific temporary files
        wdir = self.fdir if self.fdir else os.getcwd()
        for f in [f for f in os.listdir(wdir)]:
            full = f"{wdir}/{f}"
            if "cpython" in f and self._fname in f and f[-3:] == ".so":
                os.remove(full)
            elif f.startswith('c.') or (f[:5] == 'fort.' and len(f) == 6):
                # all c.<scenario> auto-07p constants files
                os.remove(full)
            elif f == f"{self._fname}.exe" or f == f"{self._fname}.mod" or f == f"{self._fname}.o":
                os.remove(full)
            elif f.startswith(('s.', 'b.', 'd.')) and f.endswith(self._fname):
                # pycobi-style auto-07p output files for THIS model
                os.remove(full)

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

    # ------------------------------------------------------------------
    # Auto-07p emits PAR(11..14) as reserved slots (PERIOD, TIME, ...).
    # PyRates routes its first 10 model parameters into PAR(1..10), then
    # skips to PAR(15+) for additional ones.  The (10, 15) tuple defines
    # that "blocked" range; class-attributed here so users / readers can
    # find the convention without rummaging through ``_generate_auto_files``.
    # ------------------------------------------------------------------
    _AUTO_BLOCKED_PAR_RANGE = (10, 15)

    # ------------------------------------------------------------------
    # Constants-file templates for typical auto-07p continuation tasks.
    # Each template specifies only the entries that DIFFER from the
    # generic defaults below; the rest are inherited.  Each can be
    # overridden / extended at call time via the ``auto_constants``
    # kwarg to ``get_run_func``.
    # ------------------------------------------------------------------
    _AUTO_CONSTANTS_DEFAULTS = {
        'NDIM': 1, 'NPAR': 1, 'IPS': -2, 'ILP': 0, 'ICP': [14],
        'NTST': 1, 'NCOL': 3, 'IAD': 0, 'ISP': 0, 'ISW': 1, 'IPLT': 0,
        'NBC': 0, 'NINT': 0, 'NMX': 9000, 'NPR': 20, 'MXBF': 10,
        'IID': 2, 'ITMX': 2, 'ITNW': 5, 'NWTN': 2, 'JAC': 0,
        'EPSL': 1e-6, 'EPSU': 1e-6, 'EPSS': 1e-4,
        'IRS': 0, 'DS': 1e-4, 'DSMIN': 1e-8, 'DSMAX': 1e-2, 'IADS': 1,
        'THL': {}, 'THU': {}, 'UZR': {}, 'UZSTOP': {},
    }

    # Per-scenario overrides.  The user picks a scenario by name (e.g.
    # ``auto_constants=('ivp', 'eq')``); each generates a separate
    # ``c.<scenario>`` file.
    _AUTO_CONSTANTS_SCENARIOS = {
        # Initial-value problem / time integration — the default.
        'ivp':  {'IPS': -2, 'ICP': [14], 'ILP': 0, 'ISP': 0,
                 'NMX': 9000, 'DS': 1e-4, 'DSMAX': 1e-2},
        # Equilibrium continuation in 1 parameter.  Defaults to PAR(1);
        # users override ICP via PyCoBi's ``.run(ICP=...)``.
        'eq':   {'IPS': 1, 'ICP': [1], 'ILP': 1, 'ISP': 2,
                 'NMX': 2000, 'NTST': 1, 'NCOL': 4,
                 'DS': 1e-3, 'DSMIN': 1e-6, 'DSMAX': 1e-1},
        # Limit cycle continuation in 1 parameter; PAR(11)=period.
        'lc':   {'IPS': 2, 'ICP': [1, 11], 'ILP': 1, 'ISP': 2,
                 'NMX': 2000, 'NTST': 50, 'NCOL': 4,
                 'DS': 1e-3, 'DSMIN': 1e-6, 'DSMAX': 1e-1},
        # Boundary-value problem.
        'bvp':  {'IPS': 4, 'ICP': [1, 2], 'ILP': 1, 'ISP': 2,
                 'NMX': 500, 'NTST': 20, 'NCOL': 4,
                 'DS': 1e-2, 'DSMIN': 1e-6, 'DSMAX': 2e-1},
    }

    def _generate_auto_files(self, func_name: str, func_args: tuple = (), state_vars: tuple = (),
                             blocked_indices: tuple = None, **kwargs):
        """Emit a fortran source file + one or more c.* constants files for auto-07p.

        Returns ``(func_file: str, constants: dict[scenario_name, str])``.  The
        caller is responsible for writing each ``constants[name]`` to ``c.<name>``.

        ``kwargs`` may include:
            - ``auto_constants``: scenario name (``'ivp'``) or iterable of
              scenario names — each gets its own ``c.<name>`` file.
              Recognised scenarios: see ``_AUTO_CONSTANTS_SCENARIOS``.
              Defaults to ``('ivp',)`` for backward compatibility.
            - Direct overrides for any auto-07p constant (``NMX=5000`` etc.).
              The override applies to every generated scenario.
            - ``auto_parnames`` / ``auto_unames``: explicit ``{idx: name}``
              dicts to emit in the c.* files.  When omitted, PyRates derives
              them from ``func_args`` / ``state_vars`` (recommended).
        """
        if blocked_indices is None:
            blocked_indices = self._AUTO_BLOCKED_PAR_RANGE

        # ------------------------------------------------------------------
        # 1. Generate the fortran source file (func + stpnt + dummy stubs).
        # ------------------------------------------------------------------
        dtype = self._get_dtype(self._var_declaration_info['y'].dtype)
        param_indices = self._auto_param_indices(func_args, blocked_indices)

        # Optional symbolic Jacobian data — passed by ComputeGraph.to_func when
        # ``auto=True`` and ``auto_jac=True``.  Used to emit DFDU/DFDP inside the
        # ``func`` wrapper, gated by IJAC > 0.  Absent → JAC=0 path (auto-07p
        # uses finite differences).
        auto_jac = kwargs.pop('auto_jacobian', None)

        # `func` wrapper around the pyrates RHS subroutine
        self.add_linebreak()
        self.add_linebreak()
        self.add_code_line("subroutine func(ndim,y,icp,args,ijac,dy,dfdu,dfdp)")
        self.add_linebreak()
        self.add_code_line(f"use {self._fname}")
        self.add_code_line("implicit none")
        self.add_code_line("integer, intent(in) :: ndim, icp(*), ijac")
        self.add_code_line(f"{dtype}, intent(in) :: y(ndim), args(*)")
        self.add_code_line(f"{dtype}, intent(out) :: dy(ndim)")
        self.add_code_line(f"{dtype}, intent(inout) :: dfdu(ndim,ndim), dfdp(ndim,*)")

        params = [f'args({i})' for i in param_indices]
        additional_args = f", {', '.join(params)}" if params else ""
        self.add_linebreak()
        self.add_code_line(f"call {func_name}(args(14), y, dy{additional_args})")

        # Emit the analytical Jacobian if available.
        provides_jac = self._emit_auto_jacobian_block(
            auto_jac, func_args, param_indices,
        ) if auto_jac else False

        self.add_linebreak()
        self.add_code_line("end subroutine func")
        self.add_linebreak()

        # `stpnt` — initial parameter values + starting state vector
        self.add_linebreak()
        self.add_code_line("subroutine stpnt(ndim, y, args, t)")
        self.add_linebreak()
        self.add_code_line("implicit None")
        self.add_code_line("integer, intent(in) :: ndim")
        self.add_code_line(f"{dtype}, intent(inout) :: y(ndim), args(*)")
        self.add_code_line(f"{dtype}, intent(in) :: t")
        self.add_linebreak()
        for idx, arg in zip(param_indices, func_args):
            p = self._var_declaration_info[arg]
            if sum(p.shape) > 1:
                raise ValueError(
                    f"Vector-valued parameter detected ({p.name} with shape {p.shape}), "
                    "which cannot be handled by Auto-07p. Please change the definition "
                    "of your network (e.g. remove extrinsic inputs) such that no "
                    "vectorized model parameters exist."
                )
            self.add_code_line(f"args({idx}) = {self._var_to_str(p)}  ! {p.name}")
        for i, var in enumerate(state_vars):
            v = self._var_declaration_info[var]
            self.add_code_line(f"y({i+1}) = {self._var_to_str(v)}  ! {v.name}")
        self.add_linebreak()
        self.add_code_line("end subroutine stpnt")
        self.add_linebreak()

        # Dummy stubs for the four unused user-defined routines.  Auto-07p's
        # own demos (`ab.f90`, `lor.f90`) use exactly this bare form, so we
        # do the same — full signatures only matter when the routine is
        # actually exercised by the chosen IPS.
        self.add_linebreak()
        for routine in ['bcnd', 'icnd', 'fopt', 'pvls']:
            self.add_linebreak()
            self.add_code_line(f"subroutine {routine}")
            self.add_code_line(f"end subroutine {routine}")
            self.add_linebreak()

        func_file = self.generate()
        self.code.clear()

        # ------------------------------------------------------------------
        # 2. Generate one or more c.* constants files.
        # ------------------------------------------------------------------
        scenarios = kwargs.pop('auto_constants', ('ivp',))
        if isinstance(scenarios, str):
            scenarios = (scenarios,)
        for scen in scenarios:
            if scen not in self._AUTO_CONSTANTS_SCENARIOS:
                raise ValueError(
                    f"Unknown auto-07p constants scenario {scen!r}.  "
                    f"Known: {sorted(self._AUTO_CONSTANTS_SCENARIOS)}"
                )

        # Build parnames / unames (modern auto-07p syntax).  Lets users write
        # ``ICP=['eta']`` instead of ``ICP=[4]`` in pycobi.
        parnames = kwargs.pop(
            'auto_parnames',
            {idx: self._var_declaration_info[arg].name
             for idx, arg in zip(param_indices, func_args)},
        )
        unames = kwargs.pop(
            'auto_unames',
            {i + 1: self._var_declaration_info[var].name
             for i, var in enumerate(state_vars)},
        )

        # User-specified constant overrides (kwargs that match top-level
        # auto constant names) — applied to every scenario.
        overrides = {k: kwargs.pop(k) for k in list(kwargs.keys())
                     if k in self._AUTO_CONSTANTS_DEFAULTS}
        # Tell auto-07p to use the user-supplied analytical Jacobian if we
        # emitted one.  ``JAC=1`` makes it call FUNC with IJAC=2 during
        # equilibrium / limit-cycle continuation; the inline block we
        # generated below fills both DFDU and DFDP.
        if provides_jac:
            overrides.setdefault('JAC', 1)

        constants_files: Dict[str, str] = {}
        for scen in scenarios:
            constants_files[scen] = self._build_auto_constants_file(
                scenario=scen,
                ndim=len(state_vars),
                npar=max(param_indices) if param_indices else 1,
                parnames=parnames,
                unames=unames,
                overrides=overrides,
            )

        return func_file, constants_files

    def _emit_auto_jacobian_block(self, jac: dict, func_args: tuple,
                                  param_indices: list) -> bool:
        """Emit ``IF(IJAC > 0)`` / ``IF(IJAC > 1)`` blocks filling DFDU / DFDP.

        Parameters
        ----------
        jac
            Output of :meth:`ComputeGraph._compute_symbolic_jacobian`
            (or compatible dict).  Must carry ``dfdu`` and ``dfdp``
            sympy-expression dicts plus the metadata needed to translate
            symbols to ``U(i)`` / ``args(idx)`` references.
        func_args
            Ordered tuple of parameter names PyRates passes to the RHS —
            same as in the surrounding ``_generate_auto_files`` call.
        param_indices
            1-based ``PAR(...)`` slot for each entry of ``func_args``,
            already accounting for auto-07p's reserved PAR(11..14) range.

        Returns
        -------
        bool
            True if any analytical Jacobian content was emitted (and the
            caller should set ``JAC=1`` in the constants files), else False.
        """
        import sympy as sp

        dfdu_entries = jac.get('dfdu') or {}
        dfdp_entries = jac.get('dfdp') or {}
        if not dfdu_entries and not dfdp_entries:
            return False

        # Build substitution maps so the sympy expressions print with the
        # Fortran identifiers PyRates' auto-07p ``func`` wrapper exposes.
        # ``sym_to_y_idx`` is 0-based; the generated Fortran signature
        # declares ``y(ndim)`` (lowercase, matching PyRates' convention),
        # so emit ``y(i)`` for 1-based ``i``.  Auto-07p docs use ``U`` but
        # Fortran is case-insensitive only when names match exactly — we
        # have to use the actual signature identifier.
        sym_to_y_idx: dict = jac.get('sym_to_y_idx', {})
        u_subs = {}
        for sym, idx in sym_to_y_idx.items():
            if isinstance(idx, tuple):
                # vector state — skipped by the Jacobian builder anyway
                continue
            u_subs[sym] = sp.Symbol(f'__PYR_Y_{idx + 1}__')

        # ``param_indices`` is parallel to ``func_args``; we want a quick
        # lookup from the *param name* (matches the dfdp keys) to its
        # ``args(k)`` slot, plus a mapping from the param's *sympy symbol*
        # to that same slot (used for substituting DFDU entries that
        # reference parameters explicitly).
        param_syms: dict = jac.get('param_syms', {})
        name_to_arg_idx = {n: i for n, i in zip(func_args, param_indices)}
        arg_subs = {}
        for name, sym in param_syms.items():
            if name in name_to_arg_idx:
                arg_subs[sym] = sp.Symbol(f'__PYR_ARG_{name_to_arg_idx[name]}__')

        def _to_fortran(expr) -> str:
            return self._sympy_to_fortran(expr, {**u_subs, **arg_subs})

        # ------- DFDU: ∂F/∂U(j), gated by IJAC > 0 -------
        if dfdu_entries:
            self.add_linebreak()
            self.add_code_line("if (ijac .eq. 0) return")
            self.add_linebreak()
            for (i_row, j_col), expr in sorted(dfdu_entries.items()):
                self.add_code_line(f"dfdu({i_row + 1},{j_col + 1}) = {_to_fortran(expr)}")

        # ------- DFDP: ∂F/∂PAR, gated by IJAC > 1 -------
        if dfdp_entries:
            self.add_linebreak()
            self.add_code_line("if (ijac .eq. 1) return")
            self.add_linebreak()
            for (i_row, pname), expr in sorted(
                dfdp_entries.items(), key=lambda kv: (kv[0][0], name_to_arg_idx.get(kv[0][1], 0))
            ):
                arg_idx = name_to_arg_idx.get(pname)
                if arg_idx is None:
                    continue
                self.add_code_line(f"dfdp({i_row + 1},{arg_idx}) = {_to_fortran(expr)}")

        return True

    def _sympy_to_fortran(self, expr, substitutions: dict) -> str:
        """Render a sympy expression as a Fortran (free-form, F90) literal.

        ``substitutions`` maps original sympy symbols to placeholder symbols
        whose names spell out the desired Fortran reference (e.g.
        ``__PYR_U_1__``).  After ``fcode`` prints the expression the
        placeholders are textually replaced with proper ``U(...)`` /
        ``args(...)`` calls — going via placeholders avoids ``fcode``
        mangling parentheses in symbol names.

        Two ``fcode`` quirks we work around:

        * ``sympy.pi`` makes ``fcode`` prepend a ``parameter (pi = ...)``
          declaration to the returned string — illegal mid-statement.  We
          replace ``sp.pi`` with a plain ``Symbol('pi')`` first; the
          surrounding module already declares ``PI`` as a constant.
        * ``fcode`` line-wraps long expressions with ``&`` continuation
          markers and indents the continuation lines.  Our own
          ``add_code_line`` does line-wrapping at the Fortran statement
          level, so we collapse fcode's wrapping back to one line and
          let ``add_code_line`` rebreak it.
        """
        import sympy as sp
        from sympy.printing.fortran import fcode

        substituted = expr.xreplace(substitutions) if substitutions else expr
        substituted = substituted.xreplace({sp.pi: sp.Symbol('pi')})
        # ``human=False`` returns ``(constants, not_supported, code)`` and
        # therefore skips the leading ``parameter (...)`` declarations.
        _consts, _not_supported, text = fcode(
            substituted, source_format='free', standard=95, human=False,
        )
        # Collapse fcode's own line wrapping — re-emit as a single line and
        # let our add_code_line rewrap.  The trailing ``&`` marks lines that
        # continue to the next; the leading whitespace on the continuation
        # line is harmless once we join.
        text = ' '.join(line.rstrip('&').strip() for line in text.splitlines())
        text = ' '.join(text.split())  # collapse runs of whitespace
        for sym in substitutions.values():
            name = sym.name
            if name.startswith('__PYR_Y_'):
                idx = name[len('__PYR_Y_'):-2]
                text = text.replace(name, f'y({idx})')
            elif name.startswith('__PYR_ARG_'):
                idx = name[len('__PYR_ARG_'):-2]
                text = text.replace(name, f'args({idx})')
        return text.strip()

    def _auto_param_indices(self, func_args: tuple, blocked: tuple) -> list:
        """Map each func arg to its 1-based PAR(...) slot, skipping reserved range."""
        increment = 1
        out = []
        for i, _arg in enumerate(func_args):
            idx = i + increment
            if blocked[0] <= idx <= blocked[1]:
                idx -= increment
                increment += blocked[1] - blocked[0]
                idx += increment
            out.append(idx)
        return out

    def _build_auto_constants_file(self, scenario: str, ndim: int, npar: int,
                                   parnames: dict, unames: dict, overrides: dict) -> str:
        """Produce the text for one ``c.<scenario>`` file."""
        consts = dict(self._AUTO_CONSTANTS_DEFAULTS)
        consts.update(self._AUTO_CONSTANTS_SCENARIOS[scenario])
        consts['NDIM'] = ndim
        consts['NPAR'] = npar
        consts.update(overrides)

        lines = []
        if parnames:
            lines.append(f"parnames = {parnames}")
        if unames:
            lines.append(f"unames = {unames}")
        for key, val in consts.items():
            lines.append(f"{key} = {val}")
        return '\n'.join(lines) + '\n'

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

        self._validate_solver(solver)

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
        elif 'complex' in dtype:
            dtype = 'complex'
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

    @staticmethod
    def _var_to_str(y: ComputeVar) -> str:
        if y.is_complex:
            return f"({np.real(y.value)}, {np.imag(y.value)})"
        return f"{y.value}"
