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
        func_args = [func_args[idx] for idx in np.sort(indices)]
        # Reorder PARAMETERS to match `_var_declaration_info`'s declaration
        # order so the generated subroutine signature matches the PAR slot
        # ordering downstream (the auto-07p wrapper calls `vfx(..., args(1),
        # args(2), ...)` where `args(i)` corresponds to PAR(i) per the c.*
        # `parnames` dict, which is itself indexed by `_var_declaration_info`
        # order in `_generate_auto_files`). Without this, the subroutine
        # would declare parameters in equation-walk order while the call
        # passes them in declaration order — parameter values get silently
        # assigned to wrong named slots inside the routine.
        #
        # `return_var` (`dy`) is pinned to position 0 of the reordered list
        # because `to_func` in computegraph.py slices `func_args[3:]` to
        # peel off `[t, y, dy]` and treat the remainder as parameters —
        # any move of `dy` would break that contract for downstream
        # consumers.
        if self._var_declaration_info and func_args:
            params = [n for n in func_args if n != return_var]
            declared_params = [n for n in self._var_declaration_info if n in params]
            other_params = [n for n in params if n not in declared_params]
            reordered = declared_params + other_params
            if return_var in func_args:
                func_args = [return_var] + reordered
            else:
                func_args = reordered
        func_args = state_vars + func_args

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
        # HomCont (IPS=9) extension — see auto-07p Ch 20.  Defaults below
        # are HomCont's own neutral values; the entries appear only in the
        # generated ``c.hom`` file (the ``_HOMCONT_KEYS`` filter below drops
        # them from every other scenario).  Users tune them via the same
        # kwargs path as any other auto-07p constant on ``from_template`` /
        # ``run``.
        'NUNSTAB': -1, 'NSTAB': -1,         # -1: derive from NDIM
        'IEQUIB': 1, 'ITWIST': 0, 'ISTART': 5,
        'IREV': [], 'IFIXED': [], 'IPSI': [],
    }

    # HomCont-only keys (auto-07p's ``main.f90:286`` re-routes these through
    # ``INSTRHO`` only — they're harmless in non-HomCont c.* files but clutter
    # them, and ``IPSI`` / ``IREV`` etc. would print as ``[]`` everywhere
    # without the filter below).  Bundle them so we can drop them in one go.
    _HOMCONT_KEYS = frozenset({
        'NUNSTAB', 'NSTAB', 'IEQUIB', 'ITWIST', 'ISTART',
        'IREV', 'IFIXED', 'IPSI',
    })

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
        # Limit-cycle continuation tolerances differ from the equilibrium
        # defaults: the BVP system auto-07p solves at IPS=2 has dimension
        # NTST*NCOL*NDIM, so Newton residuals and bifurcation test functions
        # need ~1-2 extra digits of accuracy to avoid spurious LP / BP / PD
        # detections from numerical noise.  EPSL/EPSU 1e-7 (vs the global
        # 1e-6) and EPSS 1e-5 (vs 1e-4) match the auto-07p LC demo
        # conventions and reliably suppress the spurious-LP failure mode
        # where a "fold" is detected but no stability flip follows.
        'lc':   {'IPS': 2, 'ICP': [1, 11], 'ILP': 1, 'ISP': 2,
                 'NMX': 2000, 'NTST': 50, 'NCOL': 4,
                 'DS': 1e-3, 'DSMIN': 1e-6, 'DSMAX': 1e-1,
                 'EPSL': 1e-7, 'EPSU': 1e-7, 'EPSS': 1e-5},
        # Boundary-value problem.
        'bvp':  {'IPS': 4, 'ICP': [1, 2], 'ILP': 1, 'ISP': 2,
                 'NMX': 500, 'NTST': 20, 'NCOL': 4,
                 'DS': 1e-2, 'DSMIN': 1e-6, 'DSMAX': 2e-1},
        # Homoclinic continuation via auto-07p's HomCont extension (IPS=9).
        # ICP defaults to ``[1, 11]`` — one model parameter + the orbit's
        # truncation interval (period); two-parameter HomCont continuations
        # extend this to ``ICP=['eta', 'J', 22, 24, 25, ...]`` etc. (the
        # test-function PARs at 20 + IPSI(j) are appended at run time).
        # JAC=1 because PyRates emits the analytical Jacobian and HomCont
        # consumes DFDU through the BVP wrapper.  NUNSTAB/NSTAB default to
        # -1 ("auto-derive from NDIM") since the right values depend on
        # the saddle, which is model-specific.
        'hom':  {'IPS': 9, 'ICP': [1, 11], 'ILP': 0, 'ISP': 0,
                 'NMX': 200, 'NPR': 100, 'NTST': 35, 'NCOL': 4,
                 'NBC': 0, 'NINT': 0, 'JAC': 1,
                 'DS': 0.05, 'DSMIN': 1e-4, 'DSMAX': 0.5,
                 'EPSL': 1e-7, 'EPSU': 1e-7, 'EPSS': 1e-5},
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

        # Reorder `func_args` to match `_var_declaration_info`'s order before
        # computing PAR slots. After the YAML-order-preservation fix in
        # `parse_equations`, `_var_declaration_info` carries variables in the
        # user's original declaration order; without this reordering step the
        # equation-walk order coming in via `func_args` would still drive the
        # resulting `parnames` dict (e.g. YAML order p1, p2, p3, p4 would
        # become {1: 'p4', 2: 'p2', 3: 'p1', 4: 'p3'} just from the first
        # equation's RHS arrangement).
        if func_args:
            declaration_order = [a for a in self._var_declaration_info if a in func_args]
            # Defensive: include any args present in func_args but somehow
            # missing from _var_declaration_info (no-op for normal flows).
            declaration_order += [a for a in func_args if a not in declaration_order]
            func_args = tuple(declaration_order)

        # Boundary-value problems often reference parameters that appear ONLY
        # in BCND / ICND residuals (e.g. ``intval`` for ``∫u dt = intval``),
        # not in the FUNC body. Track them separately: ``rhs_args`` is what the
        # inner ``vector_field`` subroutine actually accepts; ``func_args`` is
        # the full PAR-slot vector (rhs_args + bvp extras) used for STPNT,
        # parnames, and param_indices.
        rhs_args = tuple(func_args)
        bvp_extras = self._collect_bvp_extra_params(
            kwargs.get('boundary_conditions') or (),
            kwargs.get('integral_constraints') or (),
        )
        bvp_extras = tuple(p for p in bvp_extras
                           if p in self._var_declaration_info and p not in rhs_args)
        if bvp_extras:
            func_args = rhs_args + bvp_extras

        param_indices = self._auto_param_indices(func_args, blocked_indices)
        # Indices that line up with rhs_args specifically — the inner call below
        # passes only those slots, leaving BVP-extras untouched by FUNC.
        rhs_param_indices = param_indices[:len(rhs_args)]

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

        # Only pass the inner RHS's actual arguments; BVP-extra parameters live
        # in PAR slots but are not consumed by FUNC.
        rhs_params = [f'args({i})' for i in rhs_param_indices]
        additional_args = f", {', '.join(rhs_params)}" if rhs_params else ""
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

        # BCND / ICND — boundary-value problem residuals (IPS=4 path).
        #
        # Two ways for the user to populate these routines:
        #
        # (a) DSL: ``boundary_conditions=['u1_r - u0_r', ...]`` lists residuals
        #     in PyRates-name space.  Each entry is sympified and tokens like
        #     ``u0_<var>`` / ``u1_<var>`` / ``par_<param>`` resolve to the
        #     proper ``u0(idx)`` / ``args(idx)`` references.  NBC / NINT are
        #     derived from the list length.
        # (b) Escape hatch: ``bcnd_fortran="FB(1) = U1(1) - U0(1)\nFB(2)=..."``
        #     plus an explicit ``nbc`` lets the user write raw Fortran when the
        #     DSL is too restrictive (PDE BCs, custom Jacobians, etc.).
        #
        # When neither is given the routines fall back to the auto-07p
        # ``ab.f90`` / ``lor.f90`` bare stub form so IPS=1/2/-2 paths stay
        # unchanged.
        bc_dsl   = kwargs.pop('boundary_conditions', None)
        ic_dsl   = kwargs.pop('integral_constraints', None)
        bc_raw   = kwargs.pop('bcnd_fortran', None)
        ic_raw   = kwargs.pop('icnd_fortran', None)
        nbc_user = kwargs.pop('nbc', None)
        nint_user = kwargs.pop('nint', None)

        if bc_dsl and bc_raw is not None:
            raise ValueError("Pass either `boundary_conditions` (DSL) or "
                             "`bcnd_fortran` (raw Fortran), not both.")
        if ic_dsl and ic_raw is not None:
            raise ValueError("Pass either `integral_constraints` (DSL) or "
                             "`icnd_fortran` (raw Fortran), not both.")

        # State / param name → 1-based index dicts that the DSL needs.
        state_indices = {self._var_declaration_info[v].name: i + 1
                         for i, v in enumerate(state_vars)}
        param_idx_by_name = {self._var_declaration_info[a].name: i
                             for i, a in zip(param_indices, func_args)}

        bcnd_body, nbc_emit = self._compose_bvp_body(
            dsl=bc_dsl, raw=bc_raw, n_user=nbc_user, kind='bcnd',
            state_indices=state_indices, param_idx=param_idx_by_name,
        )
        icnd_body, nint_emit = self._compose_bvp_body(
            dsl=ic_dsl, raw=ic_raw, n_user=nint_user, kind='icnd',
            state_indices=state_indices, param_idx=param_idx_by_name,
        )

        self.add_linebreak()
        self._emit_bcnd_subroutine(dtype, bcnd_body)
        self._emit_icnd_subroutine(dtype, icnd_body)
        # FOPT / PVLS are not yet exposed to the user — keep bare stubs.
        for routine in ('fopt', 'pvls'):
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
        user_set_jac = 'JAC' in overrides
        # NBC / NINT auto-derived from the populated BCND / ICND bodies so
        # the c.* file matches the emitted subroutines.  User-supplied
        # ``nbc`` / ``nint`` kwargs (or explicit overrides on the
        # ``ODESystem.run`` call) still win via ``overrides``.
        if nbc_emit:
            overrides.setdefault('NBC', nbc_emit)
        if nint_emit:
            overrides.setdefault('NINT', nint_emit)
        # JAC selection.  With BVP residuals present, auto-07p's ``JAC=1`` path
        # expects ALL four routines (FUNC + BCND + ICND + their Jacobians DBC /
        # DINT) to be user-supplied, but we only emit FUNC's DFDU / DFDP. Force
        # ``JAC=0`` (finite differences everywhere) in that case unless the
        # user explicitly opted in. Without BVP residuals, the analytical FUNC
        # Jacobian is safe to use — ``provides_jac`` reflects whether it was
        # emitted.
        if not user_set_jac:
            if nbc_emit or nint_emit:
                overrides['JAC'] = 0
            elif provides_jac:
                overrides['JAC'] = 1

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

    # ------------------------------------------------------------------
    # Boundary-value problem helpers: emit populated BCND / ICND with
    # either DSL-resolved residuals or raw user-supplied Fortran.
    # ------------------------------------------------------------------

    # ICP and NBC / NINT are positional in the auto-07p contract; the
    # local parameter-name choices below (``args`` for PAR, lowercase
    # array names) are free.  We pick names that match the rest of the
    # generated code so identifier conventions stay consistent.
    _BVP_PREFIXES = {
        'bcnd': {'u0': 'u0', 'u1': 'u1'},
        'icnd': {'u': 'u', 'uold': 'uold', 'udot': 'udot', 'upold': 'upold'},
    }

    @staticmethod
    def _collect_bvp_extra_params(bc_dsl, ic_dsl) -> list:
        """Scan DSL residuals for ``par_<name>`` tokens and return the names.

        Used so ICND-only / BCND-only parameters (the ``intval`` style of
        integral target) still get a PAR slot in the c.* file. Order is
        preserved with deduplication.
        """
        import re
        seen, out = set(), []
        # Same identifier regex sympify accepts; we just want to enumerate.
        pat = re.compile(r'\bpar_([A-Za-z_][A-Za-z0-9_]*)\b')
        for expr_str in tuple(bc_dsl) + tuple(ic_dsl):
            for name in pat.findall(expr_str):
                if name not in seen:
                    seen.add(name)
                    out.append(name)
        return out

    def _compose_bvp_body(self, dsl, raw, n_user, kind: str,
                          state_indices: dict, param_idx: dict):
        """Return ``(body_lines, n_residuals)`` for a BCND / ICND subroutine.

        Resolution order: DSL list → raw Fortran block → empty (bare stub).
        ``n_residuals == 0`` means we should fall back to the bare stub.
        """
        residual_arr = 'fb' if kind == 'bcnd' else 'fi'
        n_kw = 'nbc' if kind == 'bcnd' else 'nint'

        if dsl:
            prefixes = self._BVP_PREFIXES[kind]
            lines = []
            for i, expr_str in enumerate(dsl, start=1):
                resolved = self._resolve_bvp_residual(
                    expr_str, prefixes=prefixes,
                    state_indices=state_indices, param_indices=param_idx,
                )
                lines.append(f"{residual_arr}({i}) = {resolved}")
            n = len(dsl)
            if n_user is not None and n_user != n:
                raise ValueError(
                    f"{kind}: provided {n} DSL residual(s) but {n_kw}={n_user}. "
                    f"Drop {n_kw} (it is auto-derived from the list length)."
                )
            return lines, n

        if raw is not None:
            if n_user is None:
                raise ValueError(
                    f"`{kind}_fortran` requires an explicit `{n_kw}` "
                    f"(the number of residuals the raw block fills)."
                )
            lines = [line for line in raw.splitlines() if line.strip()]
            return lines, int(n_user)

        # nothing populated — caller will emit a bare stub.
        return [], 0

    def _resolve_bvp_residual(self, expr_str: str, prefixes: dict,
                              state_indices: dict, param_indices: dict) -> str:
        """Resolve one DSL residual to a Fortran expression string.

        Tokens recognised:
          * ``{prefix}_{statevar}`` for each prefix in *prefixes* → ``{prefix}(idx)``
            (e.g. ``u0_r`` → ``u0(1)`` in BCND, ``udot_v`` → ``udot(2)`` in ICND)
          * ``par_{paramname}`` → ``args(idx)``
          * standard arithmetic, ``pi``, and sympy-known math functions
            (``sin``, ``cos``, ``exp``, ``sqrt``, ...)

        Any free symbol that does not match one of these patterns is an error
        (typo in a state/param name, or use of an unsupported function).
        """
        import sympy as sp
        import re

        # build the substitution dict from PyRates names → unique placeholders
        # that survive fcode pretty-printing untouched.
        subs: dict = {}
        for prefix, arr_name in prefixes.items():
            for vname, idx in state_indices.items():
                subs[sp.Symbol(f'{prefix}_{vname}')] = sp.Symbol(
                    f'__PYR_ARR_{arr_name}_{idx}__'
                )
        for pname, idx in param_indices.items():
            subs[sp.Symbol(f'par_{pname}')] = sp.Symbol(f'__PYR_ARG_{idx}__')

        try:
            expr = sp.sympify(expr_str)
        except (sp.SympifyError, SyntaxError) as e:
            raise ValueError(f"Could not parse BVP residual {expr_str!r}: {e}") from e

        # Validate: every free symbol should either be in `subs` (which then
        # gets replaced) or be a math-only Symbol like `pi`.  Catch typos up
        # front rather than letting them slip through to gfortran.
        known_symbols = set(subs.keys()) | {sp.Symbol('pi')}
        unknown = {s for s in expr.free_symbols if s not in known_symbols}
        if unknown:
            raise ValueError(
                f"BVP residual {expr_str!r} contains unrecognised symbol(s) "
                f"{sorted(s.name for s in unknown)}. Use `u0_<var>` / `u1_<var>` "
                f"(BCND) or `u_<var>` / `uold_<var>` / `udot_<var>` / "
                f"`upold_<var>` (ICND) for state references, and "
                f"`par_<param>` for parameters."
            )

        expr = expr.xreplace(subs)
        text = self._sympy_to_fortran(expr, {})  # no extra subs — already done

        # Textual replacement: __PYR_ARR_<arr>_<idx>__ → <arr>(<idx>),
        #                     __PYR_ARG_<idx>__       → args(<idx>).
        text = re.sub(r'__PYR_ARR_([A-Za-z][A-Za-z0-9]*)_(\d+)__',
                      r'\1(\2)', text)
        text = re.sub(r'__PYR_ARG_(\d+)__', r'args(\1)', text)
        return text.strip()

    def _emit_bcnd_subroutine(self, dtype: str, body_lines: list) -> None:
        self.add_linebreak()
        if not body_lines:
            self.add_code_line("subroutine bcnd")
            self.add_code_line("end subroutine bcnd")
            self.add_linebreak()
            return
        self.add_code_line(
            "subroutine bcnd(ndim, args, icp, nbc, u0, u1, fb, ijac, dbc)"
        )
        self.add_code_line("implicit none")
        self.add_code_line("integer, intent(in) :: ndim, icp(*), nbc, ijac")
        self.add_code_line(f"{dtype}, intent(in) :: args(*), u0(ndim), u1(ndim)")
        self.add_code_line(f"{dtype}, intent(out) :: fb(nbc)")
        self.add_code_line(f"{dtype}, intent(inout) :: dbc(nbc, *)")
        self.add_linebreak()
        for line in body_lines:
            self.add_code_line(line)
        self.add_linebreak()
        self.add_code_line("end subroutine bcnd")
        self.add_linebreak()

    def _emit_icnd_subroutine(self, dtype: str, body_lines: list) -> None:
        self.add_linebreak()
        if not body_lines:
            self.add_code_line("subroutine icnd")
            self.add_code_line("end subroutine icnd")
            self.add_linebreak()
            return
        self.add_code_line(
            "subroutine icnd(ndim, args, icp, nint, u, uold, "
            "udot, upold, fi, ijac, dint)"
        )
        self.add_code_line("implicit none")
        self.add_code_line("integer, intent(in) :: ndim, icp(*), nint, ijac")
        self.add_code_line(
            f"{dtype}, intent(in) :: args(*), u(ndim), uold(ndim), "
            f"udot(ndim), upold(ndim)"
        )
        self.add_code_line(f"{dtype}, intent(out) :: fi(nint)")
        self.add_code_line(f"{dtype}, intent(inout) :: dint(nint, *)")
        self.add_linebreak()
        for line in body_lines:
            self.add_code_line(line)
        self.add_linebreak()
        self.add_code_line("end subroutine icnd")
        self.add_linebreak()

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

        # HomCont keys (NUNSTAB / IEQUIB / IPSI / ...) are only consumed when
        # the run uses IPS=9; the auto-07p c.* parser silently routes them to
        # the HomCont module via INSTRHO regardless, but emitting them in
        # every scenario produces noisy and misleading c.eq / c.lc files.
        # Drop them when the scenario isn't the homoclinic one.
        if scenario != 'hom':
            for key in self._HOMCONT_KEYS:
                consts.pop(key, None)
        else:
            # Drop empty list-valued HomCont keys (IREV / IFIXED / IPSI).
            # Auto-07p's INSTRHO sets ``NREV=1`` unconditionally whenever
            # ``IREV`` is parsed — even for ``IREV=[]`` (LISTLEN=0).  An
            # accidental reversibility flag breaks NFREE bookkeeping in
            # ``INHO``, so the cleanest fix is to not write the key at all
            # when the user hasn't picked any indices.
            for key in ('IREV', 'IFIXED', 'IPSI'):
                if not consts.get(key):
                    consts.pop(key, None)

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
