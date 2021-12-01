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
import tensorflow as tf
import numpy as np

# meta infos
__author__ = "Richard Gast"
__status__ = "development"

# function definitions
######################

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
    'max': {'call': 'reduce_max', 'func': tf.reduce_max, 'imports': ['tensorflow.reduce_max']},
    'min': {'call': 'reduce_min', 'func': tf.reduce_min, 'imports': ['tensorflow.reduce_min']},
    'argmax': {'call': 'argmax', 'func': tf.argmax, 'imports': ['tensorflow.argmax']},
    'argmin': {'call': 'argmin', 'func': tf.argmin, 'imports': ['tensorflow.argmin']},
    'round': {'call': 'round', 'func': tf.round, 'imports': ['tensorflow.round']},
    'sum': {'call': 'reduce_sum', 'func': tf.reduce_sum, 'imports': ['tensorflow.reduce_sum']},
    'mean': {'call': 'reduce_mean', 'func': tf.reduce_mean, 'imports': ['tensorflow.mean']},
    'matmul': {'call': 'matmul', 'func': tf.matmul, 'imports': ['tensorflow.matmul']},
    'matvec': {'call': 'matvec', 'func': tf.linalg.matvec, 'imports': ['tensorflow.linalg.matvec']},
    'roll': {'call': 'roll', 'func': tf.roll, 'imports': ['tensorflow.roll']},
    'randn': {'call': 'normal', 'func': tf.random.normal, 'imports': ['tensorflow.random.normal']},
    'tanh': {'call': 'tanh', 'func': tf.tanh, 'imports': ['tensorflow.tanh']},
    'sinh': {'call': 'sinh', 'func': tf.sinh, 'imports': ['tensorflow.sinh']},
    'cosh': {'call': 'cosh', 'func': tf.cosh, 'imports': ['tensorflow.cosh']},
    'arctan': {'call': 'atan', 'func': tf.atan, 'imports': ['tensorflow.atan']},
    'arcsin': {'call': 'asin', 'func': tf.asin, 'imports': ['tensorflow.asin']},
    'arccos': {'call': 'acos', 'func': tf.acos, 'imports': ['tensorflow.acos']},
    'sin': {'call': 'sin', 'func': tf.sin, 'imports': ['tensorflow.sin']},
    'cos': {'call': 'cos', 'func': tf.cos, 'imports': ['tensorflow.cos']},
    'tan': {'call': 'tan', 'func': tf.tan, 'imports': ['tensorflow.tan']},
    'exp': {'call': 'exp', 'func': tf.exp, 'imports': ['tensorflow.exp']},
    'sigmoid': {'call': 'sigmoid', 'func': tf.math.sigmoid, 'imports': ['tensorflow.math.sigmoid']},
    'interp': {'call': 'interp', 'def': interp, 'imports': ['tensorflow.abs', 'tensorflow.argmin']}
}
