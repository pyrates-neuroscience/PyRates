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
from numba import jit

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


# function definitions
######################

@jit(nopython=True, fastmath=True)
def pr_sigmoid(x):
    return 1./(1. + np.exp(-x))


@jit(nopython=True, fastmath=True)
def pr_softmax(x, axis=0):
    x_exp = np.exp(x)
    return x_exp/np.sum(x_exp, axis=axis)
