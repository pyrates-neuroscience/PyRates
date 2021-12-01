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

# external _imports
import torch
import numpy as np
from typing import Callable

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


#######################################
# classes for backend functionalities #
#######################################


class TorchBackend(BaseBackend):

    def get_var(self, v: ComputeVar):
        return torch.from_numpy(super().get_var(v))

    def _solve(self, solver: str, func: Callable, args: tuple, T: float, dt: float, dts: float, y0: torch.Tensor,
               times: np.ndarray, **kwargs) -> np.ndarray:

        # perform integration via scipy solver (mostly Runge-Kutta methods)
        if solver == 'euler':

            # solve ivp via forward euler method (fixed integration step-size)
            results = self._solve_euler(func, args, T, dt, dts, y0)
            results = results.numpy()

        else:

            # solve ivp via scipy methods (solvers of various orders with adaptive step-size)
            from scipy.integrate import solve_ivp
            t = 0.0
            kwargs['t_eval'] = times

            # wrapper to rhs function
            y_tmp = y0.numpy()
            y = torch.from_numpy(y_tmp)
            def f(t, y_new):
                y_tmp[:] = y_new
                rhs = func(t, y, *args)
                return rhs.numpy()

            # call scipy solver
            results = solve_ivp(fun=f, t_span=(t, T), y0=y_tmp, first_step=dt, **kwargs)
            results = results['y'].T

        return results

    @staticmethod
    def _solve_euler(func: Callable, args: tuple, T: float, dt: float, dts: float, y: torch.Tensor) -> torch.Tensor:

        # preparations for fixed step-size integration
        idx = 0
        steps = int(np.round(T / dt))
        store_steps = int(np.round(T / dts))
        store_step = int(np.round(dts / dt))
        state_rec = torch.zeros((store_steps, y.shape[0]) if y.shape else (store_steps, 1))

        # solve ivp for forward Euler method
        for step in range(steps):
            if step % store_step == 0:
                state_rec[idx, :] = y
                idx += 1
            rhs = func(step, y, *args)
            y += dt * rhs

        return state_rec
