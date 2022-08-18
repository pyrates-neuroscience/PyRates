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

"""Contains tensorflow function definitions that may be used for PyRates model equations.
"""

# external _imports
import numpy as np

# meta infos
__author__ = "Richard Gast"
__status__ = "development"

# function definitions
######################

sigmoid = lambda x: 1./(1. + np.exp(-x))

interp = """
def interp(x_new, x, y):
    idx = argmin(abs(x-x_new))
    if x[idx] > x_new:
        i1, i2 = idx-1, idx
    else:
        i1, i2 = idx, idx+1
    return (y[i1] + y[i1])*0.5
"""

# dictionary for backend import
###############################

tf_funcs = {
    'max': {'call': 'reduce_max', 'func': np.maximum, 'imports': ['tensorflow.reduce_max']},
    'min': {'call': 'reduce_min', 'func': np.minimum, 'imports': ['tensorflow.reduce_min']},
    'round': {'call': 'round', 'func': np.round, 'imports': ['tensorflow.round']},
    'sum': {'call': 'reduce_sum', 'func': np.sum, 'imports': ['tensorflow.reduce_sum']},
    'mean': {'call': 'reduce_mean', 'func': np.mean, 'imports': ['tensorflow.mean']},
    'matmul': {'call': 'matmul', 'func': np.dot, 'imports': ['tensorflow.matmul']},
    'matvec': {'call': 'matvec', 'func': np.dot, 'imports': ['tensorflow.linalg.matvec']},
    'roll': {'call': 'roll', 'func': np.roll, 'imports': ['tensorflow.roll']},
    'randn': {'call': 'normal', 'func': np.random.randn, 'imports': ['tensorflow.random.normal']},
    'tanh': {'call': 'tanh', 'func': np.tanh, 'imports': ['tensorflow.tanh']},
    'sinh': {'call': 'sinh', 'func': np.sinh, 'imports': ['tensorflow.sinh']},
    'cosh': {'call': 'cosh', 'func': np.cosh, 'imports': ['tensorflow.cosh']},
    'arctan': {'call': 'atan', 'func': np.arctan, 'imports': ['tensorflow.atan']},
    'arcsin': {'call': 'asin', 'func': np.arcsin, 'imports': ['tensorflow.asin']},
    'arccos': {'call': 'acos', 'func': np.arccos, 'imports': ['tensorflow.acos']},
    'sin': {'call': 'sin', 'func': np.sin, 'imports': ['tensorflow.sin']},
    'cos': {'call': 'cos', 'func': np.cos, 'imports': ['tensorflow.cos']},
    'tan': {'call': 'tan', 'func': np.tan, 'imports': ['tensorflow.tan']},
    'exp': {'call': 'exp', 'func': np.exp, 'imports': ['tensorflow.exp']},
    'sigmoid': {'call': 'sigmoid', 'func': sigmoid, 'imports': ['tensorflow.math.sigmoid']},
    'interp': {'call': 'interp', 'func': np.interp, 'def': interp, 'imports': ['tensorflow.abs', 'tensorflow.argmin']},
    'real': {'call': 'real', 'func': np.real, 'imports': ['tensorflow.math.real']},
    'imag': {'call': 'imag', 'func': np.imag, 'imports': ['tensorflow.math.imag']},
    'conj': {'call': 'conj', 'func': np.conjugate, 'imports': ['tensorflow.math.conj']},
    'absv': {'call': 'abs', 'func': np.abs, 'imports': ['tensorflow.abs']},
    'log': {'call': 'log', 'func': np.log, 'imports': ['tensorflow.math.log']},
    'concatenate': {'call': 'concat', 'func': np.concatenate, 'imports': ['tensorflow.concat']}
}
