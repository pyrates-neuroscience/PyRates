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
"""JAX backend.  Generates :code:`jax.numpy` source code for the model RHS,
JIT-compiles the resulting function, and integrates ODEs via :code:`diffrax`
when available — with a scipy fall-back that handles JAX↔NumPy conversion
at the integrator boundary.

Differences from the numpy default that warrant a dedicated backend:

* JAX arrays are immutable, so the standard :code:`dy[i] = ...` lines that
  the base code generator emits would error at JIT-trace time.  We override
  :code:`add_var_update` to emit JAX functional updates (:code:`dy = dy.at[i]
  .set(...)`) instead.
* The generated function is decorated with :code:`@jax.jit` so subsequent
  calls (parameter fitting, sweeps) reuse the compiled trace.
* :code:`get_var` returns :code:`jax.numpy` arrays so the JIT trace sees
  consistent types across calls.
"""

# pyrates internal imports
from ..base import BaseBackend
from ..computegraph import ComputeVar
from .jax_funcs import jax_funcs

# external imports
import numpy as np
from typing import Callable, Iterable, Optional, Dict, List, Tuple, Union

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


# JAX needs to be importable but we delay heavy initialisation until the
# backend is instantiated so users without JAX installed can still use the
# rest of PyRates.
try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:                                                          # pragma: no cover
    jax = None
    jnp = None
    _JAX_AVAILABLE = False


class JaxBackend(BaseBackend):
    """Code-generating JAX backend.

    Notes
    -----
    Requires :code:`jax` (and, for the recommended :code:`solver='diffrax'`
    path, the :code:`diffrax` package).  Set :code:`float_precision='float64'`
    if you need x64 precision — the backend will enable :code:`jax_enable_x64`
    for the whole process automatically.
    """

    def __init__(self,
                 ops: Optional[Dict[str, str]] = None,
                 imports: Optional[List[str]] = None,
                 **kwargs) -> None:

        if not _JAX_AVAILABLE:
            raise ImportError(
                "JaxBackend requires the `jax` package. Install with "
                "`pip install 'jax[cpu]' diffrax` (or `jax[cuda]` for GPU)."
            )

        # x64 mode is global to the JAX process — opt-in via float_precision
        if kwargs.get('float_precision', 'float32') == 'float64':
            jax.config.update("jax_enable_x64", True)

        # merge default jax_funcs with any user-provided ops
        all_ops = jax_funcs.copy()
        if ops:
            all_ops.update(ops)

        super().__init__(ops=all_ops, imports=imports, **kwargs)
        # replace the default "from numpy import pi, sqrt" with a JAX equivalent
        # (the base class always sets this as the first import line)
        self._imports[0] = "from jax.numpy import pi, sqrt"
        # ensure `jit` is available inside the generated module
        self.add_import("from jax import jit")
        self.add_import("import jax.numpy as jnp")

    # ---------------------------------------------------------------------
    #  Variable conversion: numpy → jax.numpy
    # ---------------------------------------------------------------------
    def get_var(self, v: ComputeVar):
        """Return a :code:`jax.numpy` array for the given ComputeVar.

        Constants kept at NumPy scalar shape :code:`(1,)` are squeezed to 0-d
        (same as the base class) so that arithmetic with state vectors does
        not broadcast unintentionally.
        """
        arr = super().get_var(v)
        # Time / state variables are produced by the solver — keep them as
        # plain NumPy here; conversion happens at the diffrax/scipy boundary.
        if v.name in ('t', 'time'):
            return arr
        return jnp.asarray(arr)

    # ---------------------------------------------------------------------
    #  Code generation: functional index-updates instead of in-place writes
    # ---------------------------------------------------------------------
    def add_var_update(self,
                       lhs: ComputeVar,
                       rhs: str,
                       lhs_idx: Optional[str] = None,
                       rhs_shape: Optional[tuple] = ()):
        """Emit a JAX-safe variable update.

        Without an index this collapses to a plain Python rebind, which works
        for both NumPy and JAX.  With an index we cannot do
        :code:`a[i] = expr` on a JAX array, so we emit
        :code:`a = a.at[i].set(expr)` instead.
        """
        lhs_name = lhs.name
        if lhs_idx:
            idx, _ = self.create_index_str(lhs_idx, apply=False)
            self.add_code_line(f"{lhs_name} = {lhs_name}.at[{idx}].set({rhs})")
        else:
            self.add_code_line(f"{lhs_name} = {rhs}")

    # ---------------------------------------------------------------------
    #  Decorate the generated function with @jit
    # ---------------------------------------------------------------------
    def _add_func_call(self, name: str, args: Iterable, return_var: str = 'dy'):
        self.add_code_line('@jit')
        self.add_code_line(f"def {name}({','.join(args)}):")

    # ---------------------------------------------------------------------
    #  ODE solvers
    # ---------------------------------------------------------------------
    def _solve(self, solver: str, func: Callable, args: tuple, T: float, dt: float, dts: float,
               y0: np.ndarray, t0: np.ndarray, times: np.ndarray, **kwargs) -> np.ndarray:

        # cast initial value to JAX so the JIT-compiled function gets the right type
        y0 = jnp.asarray(y0)

        if solver == 'diffrax':
            return self._solve_diffrax(func, args, T, dt, y0, t0, times, **kwargs)

        if solver == 'scipy':
            from ..base.base_backend import DDEHistory
            if len(args) > 0 and isinstance(args[0], DDEHistory):
                # DDEs need history-buffer mutation, which is not currently
                # JIT-safe — fall back to the base scipy DDE solver.
                return super()._solve_scipy_dde(func, args, T, dt, y0, t0, times, **kwargs)
            return self._solve_scipy(func, args, T, dt, y0, t0, times, **kwargs)

        if solver == 'euler':
            return self._solve_euler(func, args, T, dt, dts, y0, t0)

        if solver == 'heun':
            return self._solve_heun(func, args, T, dt, dts, y0, t0)

        raise ValueError(f"Unknown solver `{solver}` for JaxBackend. "
                         f"Supported: 'diffrax', 'scipy', 'euler', 'heun'.")

    def _solve_diffrax(self, func: Callable, args: tuple, T: float, dt: float,
                       y0: 'jnp.ndarray', t0: 'np.ndarray', times: np.ndarray, **kwargs):
        """Adaptive integration via :code:`diffrax.diffeqsolve`."""
        try:
            import diffrax
        except ImportError as e:                                             # pragma: no cover
            raise ImportError(
                "solver='diffrax' requires the `diffrax` package. "
                "Install with `pip install diffrax`."
            ) from e

        # diffrax expects f(t, y, args) -> dy.  The PyRates-generated function
        # has signature f(t, y, dy_buf, *params); args[0] is the original dy
        # buffer and the remaining entries are the model parameters.  We pass
        # the full args tuple through diffrax and unpack inside term_fun.
        def term_fun(t, y, params):
            return func(t, y, *params)

        method_name = kwargs.pop('method', 'Tsit5')
        try:
            solver_cls = getattr(diffrax, method_name)
        except AttributeError:
            solver_cls = diffrax.Tsit5
        rtol = kwargs.pop('rtol', 1e-6)
        atol = kwargs.pop('atol', 1e-8)
        kwargs.pop('first_step', None)

        ts_eval = jnp.asarray(times)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(term_fun),
            solver_cls(),
            t0=float(t0), t1=float(T), dt0=float(dt),
            y0=y0, args=tuple(args),
            saveat=diffrax.SaveAt(ts=ts_eval),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=kwargs.pop('max_steps', 1_000_000),
        )
        return np.asarray(sol.ys)

    def _solve_scipy(self, func: Callable, args: tuple, T: float, dt: float,
                     y0: 'jnp.ndarray', t0: 'np.ndarray', times: np.ndarray, **kwargs):
        """scipy.solve_ivp wrapper that converts at the integrator boundary."""
        from scipy.integrate import solve_ivp
        kwargs['t_eval'] = times

        dtype = jnp.result_type(y0)

        def f(t, y):
            return np.asarray(
                func(jnp.asarray(t, dtype=dtype), jnp.asarray(y, dtype=dtype), *args)
            )

        results = solve_ivp(fun=f, t_span=(float(t0), float(T)),
                            y0=np.asarray(y0), first_step=dt, **kwargs)
        return results['y'].T

    def _solve_euler(self, func: Callable, args: tuple, T: float, dt: float, dts: float,
                     y: 'jnp.ndarray', t0) -> np.ndarray:
        """Plain Python forward Euler — uses JAX arrays but no fori_loop.

        :code:`jax.lax.fori_loop` would be JIT-friendly but is incompatible
        with DDE-style mutable history; the simple form is good enough for
        the small problems the Euler solver targets.

        :code:`args` is the full PyRates parameter tuple including the dy
        buffer at index 0 — we pass it through unchanged.
        """
        idx = 0
        steps = int(np.round(T / dt))
        store_steps = int(np.round(T / dts))
        store_step = int(np.round(dts / dt))
        n_state = y.shape[0] if y.ndim > 0 else 1
        state_rec = np.zeros((store_steps, n_state), dtype=np.asarray(y).dtype)

        t = float(t0)
        for step in range(int(t0), steps + int(t0)):
            if step % store_step == int(t0):
                state_rec[idx, :] = np.asarray(y)
                idx += 1
            rhs = func(jnp.asarray(t), y, *args)
            y = y + dt * rhs
            t = t + dt
        return state_rec

    def _solve_heun(self, func: Callable, args: tuple, T: float, dt: float, dts: float,
                    y: 'jnp.ndarray', t0) -> np.ndarray:
        """Heun's (improved Euler) method — same buffer convention as `_solve_euler`."""
        idx = 0
        steps = int(np.round(T / dt))
        store_steps = int(np.round(T / dts))
        store_step = int(np.round(dts / dt))
        n_state = y.shape[0] if y.ndim > 0 else 1
        state_rec = np.zeros((store_steps, n_state), dtype=np.asarray(y).dtype)

        t = float(t0)
        for step in range(int(t0), steps + int(t0)):
            if step % store_step == int(t0):
                state_rec[idx, :] = np.asarray(y)
                idx += 1
            k1 = func(jnp.asarray(t), y, *args)
            y_pred = y + dt * k1
            k2 = func(jnp.asarray(t + dt), y_pred, *args)
            y = y + 0.5 * dt * (k1 + k2)
            t = t + dt
        return state_rec
