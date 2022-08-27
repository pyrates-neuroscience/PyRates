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

"""Contains Julia function definitions that may be used for PyRates model equations.
"""

# external _imports
import numpy as np

# meta infos
__author__ = "Richard Gast"
__status__ = "development"

# function definitions
######################

sigmoid_func = lambda x: 1./(1. + np.exp(-x))
sigmoid_def = """
function sigmoid(x)
    return 1.0/(1.0 + exp(-x))
end
"""

interp = """
function interp(x_new, x, y)
    idx = argmin(abs.(x.-x_new))
    if x[idx] > x_new
        i1, i2 = max(idx-1, 1), idx
    else
        i1, i2 = idx, min(idx+1, length(y))
    end
    return (y[i1] + y[i2])*0.5
end
"""

# dictionary for backend import
###############################

julia_funcs = {
    'max': {'call': 'maximum', 'func': np.maximum, 'imports': []},
    'min': {'call': 'mininum', 'func': np.minimum, 'imports': []},
    'round': {'call': 'round', 'func': np.round, 'imports': []},
    'sum': {'call': 'sum', 'func': np.sum, 'imports': []},
    'mean': {'call': 'mean', 'func': np.mean, 'imports': []},
    'matmul': {'call': '*', 'func': np.dot, 'imports': []},
    'matvec': {'call': '*', 'func': np.dot, 'imports': []},
    'roll': {'call': 'roll', 'func': np.roll, 'imports': []},
    'randn': {'call': 'normal', 'func': np.random.randn, 'imports': []},
    'tanh': {'call': 'tanh', 'func': np.tanh, 'imports': []},
    'sinh': {'call': 'sinh', 'func': np.sinh, 'imports': []},
    'cosh': {'call': 'cosh', 'func': np.cosh, 'imports': []},
    'arctan': {'call': 'atan', 'func': np.arctan, 'imports': []},
    'arcsin': {'call': 'asin', 'func': np.arcsin, 'imports': []},
    'arccos': {'call': 'acos', 'func': np.arccos, 'imports': []},
    'sin': {'call': 'sin', 'func': np.sin, 'imports': []},
    'cos': {'call': 'cos', 'func': np.cos, 'imports': []},
    'tan': {'call': 'tan', 'func': np.tan, 'imports': []},
    'exp': {'call': 'exp', 'func': np.exp, 'imports': []},
    'sigmoid': {'call': 'sigmoid', 'func': sigmoid_func, 'def': sigmoid_def, 'imports': []},
    'interp': {'call': 'interp', 'func': np.interp, 'def': interp, 'imports': []},
    'real': {'call': 'real', 'func': np.real, 'imports': []},
    'imag': {'call': 'imag', 'func': np.imag, 'imports': []},
    'conj': {'call': 'conj', 'func': np.conjugate, 'imports': []},
    'absv': {'call': 'abs', 'func': np.abs, 'imports': []},
    'log': {'call': 'log', 'func': np.log, 'imports': []},
    'concatenate': {'call': 'cat', 'func': np.concatenate, 'imports': []}
}
