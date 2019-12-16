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

"""Contains functions that may be used as backend operations

"""

# external imports
import numpy as np
from scipy.interpolate import interp1d

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


# function definitions
######################

def pr_sigmoid(x, scaling=1.0, steepness=1.0, offset=0.0):
    return scaling/(1. + np.exp(steepness*(offset-x)))


def pr_softmax(x, axis=0):
    x_exp = np.exp(x)
    return x_exp/np.sum(x_exp, axis=axis)


def pr_identity(x):
    return x


def pr_interp_1d_linear(x, y, x_new):
    return np.interp(x_new, x, y)


def pr_interp_nd_linear(x, y, x_new, y_idx, t):
    return np.asarray([np.interp(t - x_new_tmp, x, y[i, :]) for i, x_new_tmp in zip(y_idx, x_new)])


def pr_interp_1d(x, y, x_new):
    return interp1d(x, y, kind=3, axis=-1)(x_new)


def pr_interp_nd(x, y, x_new, y_idx, t):
    try:
        f = interp1d(x, y, kind=3, axis=-1, fill_value='extrapolate', copy=False)
    except ValueError:
        try:
            x, idx = np.unique(x, return_index=True)
            f = interp1d(x, y[:, idx], kind=3, axis=-1, copy=False, fill_value='extrapolate')
        except ValueError:
            f = interp1d(x, y[:, idx], kind='linear', axis=-1, copy=False, fill_value='extrapolate')
    return np.asarray([f(x_new_tmp)[i] for i, x_new_tmp in zip(y_idx, t-x_new)])


def pr_interp(f, x_new):
    return f(x_new)
