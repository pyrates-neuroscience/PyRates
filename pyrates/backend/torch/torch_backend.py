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

"""Wraps torch such that it's low-level functions can be used by PyRates to create and simulate a compute graph.
"""

# pyrates internal _imports
from ..base import BaseBackend
from ..computegraph import ComputeVar
from .torch_funcs import torch_funcs

# external _imports
import torch
import numpy as np
from typing import Callable, Optional, Dict, List

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


#######################################
# classes for backend functionalities #
#######################################


class TorchBackend(BaseBackend):

    # `heun` falls back to BaseBackend._solve_heun, which writes to a numpy
    # state-record buffer and would silently sever the autograd graph.
    # Declare it unsupported until a tensor-native heun is added.
    SUPPORTED_SOLVERS = ('euler', 'scipy')

    def __init__(self,
                 ops: Optional[Dict[str, str]] = None,
                 imports: Optional[List[str]] = None,
                 **kwargs
                 ) -> None:
        """Instantiates PyTorch backend.
        """

        # add user-provided operations to function dict
        torch_ops = torch_funcs.copy()
        if ops:
            torch_ops.update(ops)

        # ensure that long is the standard integer type
        if 'int_precision' in kwargs:
            print(f"Warning: User-provided integer precision `{kwargs.pop('int_precision')}` will be ignored, since the"
                  f"torch backend requires integer precision `int64` for some indexing operations.")
        kwargs['int_precision'] = 'int64'

        # call parent method
        super().__init__(ops=torch_ops, imports=imports, **kwargs)
        self._imports[0] = "from torch import pi, sqrt"

    def get_var(self, v: ComputeVar):
        return torch.from_numpy(super().get_var(v))

    def _solve_scipy(self, func: Callable, args: tuple, T: float, dt: float, y: torch.Tensor, t0: torch.Tensor,
                     times: torch.Tensor, **kwargs):

        # solve ivp via scipy methods (solvers of various orders with adaptive step-size)
        from scipy.integrate import solve_ivp
        kwargs['t_eval'] = times
        dtype = self._torch_float_dtype(y)

        # wrapper to rhs function: use torch.as_tensor for a zero-copy view of
        # the numpy arrays scipy hands us (a copy only occurs when the dtypes
        # differ, matching the previous torch.tensor() behavior).
        def f(t, y):
            rhs = func(torch.as_tensor(t, dtype=dtype), torch.as_tensor(y, dtype=dtype), *args)
            return rhs.numpy()

        # call scipy solver
        results = solve_ivp(fun=f, t_span=(t0, T), y0=y, first_step=dt, **kwargs)
        return results['y'].T

    def _solve_scipy_dde(self, func: Callable, args: tuple, T: float, dt: float, y: torch.Tensor,
                         t0: torch.Tensor, times: np.ndarray, **kwargs):
        """DDE integration via scipy.integrate.ode.

        Closes the silent-fallback flagged in review §1.2: when a model with
        ``delay`` edges is run on TorchBackend with ``solver='scipy'``, the
        BaseBackend implementation called the compiled torch function with
        raw numpy inputs (it happened to work for purely-arithmetic RHS but
        was undefined for tensor ops).  This override wraps the RHS with the
        same :code:`torch.as_tensor` boundary conversion used by
        :meth:`_solve_scipy`.

        Note: autograd does not flow through the scipy step; the returned
        trajectory is plain numpy.  A tensor-native DDE solver would be
        needed for differentiable simulation.
        """
        from scipy.integrate import ode
        from ..base.base_backend import DDEHistory

        hist = args[0]
        if not isinstance(hist, DDEHistory):
            raise TypeError("_solve_scipy_dde expects args[0] to be a DDEHistory instance.")
        kwargs.pop('method', None)
        dtype = self._torch_float_dtype(y)

        # rhs wrapper that crosses the numpy ↔ torch boundary cleanly
        def rhs(t, y_):
            out = func(torch.as_tensor(t, dtype=dtype), torch.as_tensor(y_, dtype=dtype), *args)
            return out.numpy() if hasattr(out, 'numpy') else np.asarray(out)

        y_np = y.numpy() if hasattr(y, 'numpy') else np.asarray(y)

        solver = ode(rhs).set_integrator('dopri5', first_step=dt, nsteps=50000)
        solver.set_initial_value(y_np, float(t0))

        def solout(t, y_):
            hist.update(t, y_.copy())
            return 0
        solver.set_solout(solout)

        # np.zeros (not np.empty): the loop may break early on integrator
        # failure, leaving unwritten rows that must remain defined.
        state_rec = np.zeros((len(times), y_np.shape[0]), dtype=y_np.dtype)
        for i, t_out in enumerate(times):
            if not solver.successful():
                break
            solver.integrate(t_out)
            state_rec[i, :] = solver.y

        return state_rec

    def _torch_float_dtype(self, y: torch.Tensor) -> torch.dtype:
        """Resolve the torch dtype used inside scipy-bridged RHS wrappers."""
        if y.dtype.is_complex:
            return y.dtype
        try:
            return getattr(torch, self._float_precision)
        except AttributeError:
            return torch.get_default_dtype()

    @staticmethod
    def _solve_euler(func: Callable, args: tuple, T: float, dt: float, dts: float, y: torch.Tensor, t0: int
                     ) -> np.ndarray:
        """Forward-Euler integration with tensor-valued state.

        Mirrors :code:`BaseBackend._solve_euler`'s control flow: the write
        cursor (``idx``) and the time index (``step``) are separate variables.
        The previous implementation conflated them, which broke any nonzero
        ``t0`` (review §4.1) — the first stored sample landed at
        ``state_rec[t0, :]`` instead of ``state_rec[0, :]``.

        DDE history updates from :code:`BaseBackend._solve_euler` are not
        replicated here because :class:`DDEHistory.update` calls
        :code:`y.copy()`, which is not a method on torch tensors.  A
        tensor-native DDE+Euler path would need its own ring buffer; until
        then DDE simulation on the torch backend should use ``solver='scipy'``
        (see :meth:`_solve_scipy_dde` below).
        """
        # preparations for fixed step-size integration
        idx = 0
        steps = int(np.round(T / dt))
        store_steps = int(np.round(T / dts))
        store_step = int(np.round(dts / dt))
        # state_rec is fully overwritten row-by-row; torch.empty skips the
        # zero-fill (safe now that the idx-shadow bug is fixed).
        state_rec = torch.empty((store_steps, y.shape[0]) if y.shape else (store_steps, 1), dtype=y.dtype)

        # solve ivp via forward Euler
        for step in range(int(t0), steps + int(t0)):
            if step % store_step == int(t0):
                state_rec[idx, :] = y
                idx += 1
            rhs = func(step, y, *args)
            y += dt * rhs

        return state_rec.numpy()
