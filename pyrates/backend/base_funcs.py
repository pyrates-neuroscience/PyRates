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

"""Contains string-based function definitions that may be used as backend operations
"""

# external imports
import numpy as np

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


# function definitions
######################

neg_one = """
def neg_one(x):
    return -1*x
"""

sigmoid = """
def sigmoid(x):
    return 1./(1. + exp(-x))
"""

identity = """
def identity(x):
    return x
"""

index_1d = """
def index_1d(x, idx):
    return x[idx]
"""

index_2d = """
def index_2d(x, idx1, idx2):
    return x[idx1, idx2]
"""

index_range = """
def index_range(x, idx1, idx2):
    return x[idx1:idx2]
"""

index_axis = """
def index_axis(x, idx=None, axis=0):
    if idx is not None:
        return x[idx] if axis == 0 else x[:, idx]
    else:
        return x[:]
"""

# dictionary for backend import
###############################

base_funcs = {
    'max': {'call': 'maximum', 'func': np.maximum, 'imports': ['numpy.maximum']},
    'min': {'call': 'minimum', 'func': np.minimum, 'imports': ['numpy.minimum']},
    'argmax': {'call': 'argmax', 'func': np.argmax, 'imports': ['numpy.argmax']},
    'argmin': {'call': 'argmin','func': np.argmin, 'imports': ['numpy.argmin']},
    'round': {'call': 'round', 'func': np.round, 'imports': ['numpy.round']},
    'sum': {'call': 'sum', 'func': np.sum, 'imports': ['numpy.sum']},
    'mean': {'call': 'mean', 'func': np.mean, 'imports': ['numpy.mean']},
    'matmul': {'call': 'dot', 'func': np.dot, 'imports': ['numpy.dot']},
    'matvec': {'call': 'dot', 'func': np.dot, 'imports': ['numpy.dot']},
    'roll': {'call': 'roll', 'func': np.roll, 'imports': ['numpy.roll']},
    'cast': {'call': 'asarray', 'func': np.asarray, 'imports': ['numpy.asarray']},
    'randn': {'call': 'randn', 'func': np.random.randn, 'imports': ['numpy.random.randn']},
    'tanh': {'call': 'tanh', 'func': np.tanh, 'imports': ['numpy.tanh']},
    'sinh': {'call': 'sinh', 'func': np.sinh, 'imports': ['numpy.sinh']},
    'cosh': {'call': 'cosh', 'func': np.cosh, 'imports': ['numpy.cosh']},
    'arctan': {'call': 'arctan', 'func': np.arctan, 'imports': ['numpy.arctan']},
    'arcsin': {'call': 'arcsin', 'func': np.arcsin, 'imports': ['numpy.arcsin']},
    'arccos': {'call': 'arccos', 'func': np.arccos, 'imports': ['numpy.arccos']},
    'sin': {'call': 'sin', 'func': np.sin, 'imports': ['numpy.sin']},
    'cos': {'call': 'cos', 'func': np.cos, 'imports': ['numpy.cos']},
    'tan': {'call': 'tan', 'func': np.tan, 'imports': ['numpy.tan']},
    'exp': {'call': 'exp', 'func': np.exp, 'imports': ['numpy.exp']},
    'interp': {'call': 'interp', 'func': np.interp, 'imports': ['numpy.interp']},
    'neg_one': {'call': 'neg_one', 'def': neg_one},
    'sigmoid': {'call': 'sigmoid', 'def': sigmoid},
    'no_op': {'call': 'identity', 'def': identity},
    'index': {'call': 'index_1d', 'def': index_1d},
    'index_range': {'call': 'index_range', 'def': index_range},
    'index_2d': {'call': 'index_2d', 'def': index_2d},
    'index_axis': {'call': 'index_axis', 'def': index_axis}
}
