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

    def get_var(self, v: ComputeVar):
        return torch.from_numpy(super().get_var(v))

    def _solve_scipy(self, func: Callable, args: tuple, T: float, dt: float, y: torch.Tensor, t0: torch.Tensor,
                     times: torch.Tensor, **kwargs):

        # solve ivp via scipy methods (solvers of various orders with adaptive step-size)
        from scipy.integrate import solve_ivp
        kwargs['t_eval'] = times
        if y.dtype.is_complex:
            dtype = y.dtype
        else:
            try:
                dtype = getattr(torch, self._float_precision)
            except AttributeError:
                dtype = torch.get_default_dtype()

        # wrapper to rhs function
        def f(t, y):
            rhs = func(torch.tensor(t, dtype=dtype), torch.tensor(y, dtype=dtype), *args)
            return rhs.numpy()

        # call scipy solver
        results = solve_ivp(fun=f, t_span=(t0, T), y0=y, first_step=dt, **kwargs)
        return results['y'].T

    @staticmethod
    def _solve_euler(func: Callable, args: tuple, T: float, dt: float, dts: float, y: torch.Tensor, idx: int
                     ) -> torch.Tensor:

        # preparations for fixed step-size integration
        steps = int(np.round(T / dt))
        store_steps = int(np.round(T / dts))
        store_step = int(np.round(dts / dt))
        state_rec = torch.zeros((store_steps, y.shape[0]) if y.shape else (store_steps, 1), dtype=y.dtype)

        # solve ivp for forward Euler method
        for step in torch.arange(int(idx), steps):
            if step % store_step == 0:
                state_rec[idx, :] = y
                idx += 1
            rhs = func(step, y, *args)
            y += dt * rhs

        return state_rec.numpy()
