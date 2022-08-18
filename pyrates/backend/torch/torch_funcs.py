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

"""Contains torch function definitions that may be used for PyRates model equations.
"""

# external _imports
import torch
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
    if abs(x[idx]) > abs(x_new):
        i1, i2 = idx-1, idx
    else:
        i1, i2 = idx, idx+1
    return (y[i1] + y[i1])*0.5
"""

# dictionary for backend import
###############################

torch_funcs = {
    'max': {'call': 'max', 'func': np.max, 'imports': ['torch.max']},
    'min': {'call': 'min', 'func': np.min, 'imports': ['torch.min']},
    'round': {'call': 'round', 'func': np.round, 'imports': ['torch.round']},
    'sum': {'call': 'sum', 'func': np.sum, 'imports': ['torch.sum']},
    'mean': {'call': 'mean', 'func': np.mean, 'imports': ['torch.mean']},
    'matmul': {'call': 'matmul', 'func': np.dot, 'imports': ['torch.matmul']},
    'matvec': {'call': 'matmul', 'func': np.dot, 'imports': ['torch.matmul']},
    'roll': {'call': 'roll', 'func': np.roll, 'imports': ['torch.roll']},
    'randn': {'call': 'randn', 'func': np.random.randn, 'imports': ['torch.randn']},
    'tanh': {'call': 'tanh', 'func': np.tanh, 'imports': ['torch.tanh']},
    'sinh': {'call': 'sinh', 'func': np.sinh, 'imports': ['torch.sinh']},
    'cosh': {'call': 'cosh', 'func': np.cosh, 'imports': ['torch.cosh']},
    'arctan': {'call': 'arctan', 'func': np.arctan, 'imports': ['torch.arctan']},
    'arcsin': {'call': 'arcsin', 'func': np.arcsin, 'imports': ['torch.arcsin']},
    'arccos': {'call': 'arccos', 'func': np.arccos, 'imports': ['torch.arccos']},
    'sin': {'call': 'sin', 'func': np.sin, 'imports': ['torch.sin']},
    'cos': {'call': 'cos', 'func': np.cos, 'imports': ['torch.cos']},
    'tan': {'call': 'tan', 'func': np.tan, 'imports': ['torch.tan']},
    'exp': {'call': 'exp', 'func': np.exp, 'imports': ['torch.exp']},
    'sigmoid': {'call': 'sigmoid', 'func': sigmoid, 'imports': ['torch.sigmoid']},
    'interp': {'call': 'interp', 'func': np.interp, 'def': interp, 'imports': ['torch.abs', 'torch.argmin']},
    'real': {'call': 'real', 'func': np.real, 'imports': ['torch.real']},
    'imag': {'call': 'imag', 'func': np.imag, 'imports': ['torch.imag']},
    'conj': {'call': 'conj', 'func': np.conjugate, 'imports': ['torch.conj']},
    'absv': {'call': 'abs', 'func': np.abs, 'imports': ['torch.abs']},
    'log': {'call': 'log', 'func': np.log, 'imports': ['torch.log']},
    'concatenate': {'call': 'concat', 'func': np.concatenate, 'imports': ['torch.concat']}
}
