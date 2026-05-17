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

"""JAX function definitions used by PyRates' code generator.

Each entry maps a PyRates equation-language primitive to the corresponding
:code:`jax.numpy` call.  The :code:`func` field still references the NumPy
implementation — it is used by the symbolic parser for shape/value inference,
not for runtime evaluation in the generated function.
"""

# external imports
import numpy as np

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


# Helper function definitions
# ---------------------------
sigmoid = lambda x: 1. / (1. + np.exp(-x))

interp = """
def interp(x_new, x, y):
    idx = argmin(abs(x - x_new))
    if abs(x[idx]) > abs(x_new):
        i1, i2 = idx - 1, idx
    else:
        i1, i2 = idx, idx + 1
    return (y[i1] + y[i2]) * 0.5
"""


# Function registry — same schema as torch_funcs / base_funcs
# -----------------------------------------------------------
jax_funcs = {
    'maxi':        {'call': 'maximum',  'func': np.maximum,      'imports': ['jax.numpy.maximum']},
    'mini':        {'call': 'minimum',  'func': np.minimum,      'imports': ['jax.numpy.minimum']},
    'round':       {'call': 'round',    'func': np.round,        'imports': ['jax.numpy.round']},
    'sum':         {'call': 'sum',      'func': np.sum,          'imports': ['jax.numpy.sum']},
    'mean':        {'call': 'mean',     'func': np.mean,         'imports': ['jax.numpy.mean']},
    'matmul':      {'call': 'matmul',   'func': np.dot,          'imports': ['jax.numpy.matmul']},
    'matvec':      {'call': 'matmul',   'func': np.dot,          'imports': ['jax.numpy.matmul']},
    'roll':        {'call': 'roll',     'func': np.roll,         'imports': ['jax.numpy.roll']},
    'randn':       {'call': 'randn',    'func': np.random.randn, 'imports': ['jax.numpy.random.randn']},
    'tanh':        {'call': 'tanh',     'func': np.tanh,         'imports': ['jax.numpy.tanh']},
    'sinh':        {'call': 'sinh',     'func': np.sinh,         'imports': ['jax.numpy.sinh']},
    'cosh':        {'call': 'cosh',     'func': np.cosh,         'imports': ['jax.numpy.cosh']},
    'arctan':      {'call': 'arctan',   'func': np.arctan,       'imports': ['jax.numpy.arctan']},
    'arcsin':      {'call': 'arcsin',   'func': np.arcsin,       'imports': ['jax.numpy.arcsin']},
    'arccos':      {'call': 'arccos',   'func': np.arccos,       'imports': ['jax.numpy.arccos']},
    'sin':         {'call': 'sin',      'func': np.sin,          'imports': ['jax.numpy.sin']},
    'cos':         {'call': 'cos',      'func': np.cos,          'imports': ['jax.numpy.cos']},
    'tan':         {'call': 'tan',      'func': np.tan,          'imports': ['jax.numpy.tan']},
    'exp':         {'call': 'exp',      'func': np.exp,          'imports': ['jax.numpy.exp']},
    'sigmoid':     {'call': 'sigmoid',  'func': sigmoid,         'imports': ['jax.nn.sigmoid']},
    'interp':      {'call': 'interp',   'func': np.interp, 'def': interp, 'imports': ['jax.numpy.abs', 'jax.numpy.argmin']},
    'real':        {'call': 'real',     'func': np.real,         'imports': ['jax.numpy.real']},
    'imag':        {'call': 'imag',     'func': np.imag,         'imports': ['jax.numpy.imag']},
    'conj':        {'call': 'conj',     'func': np.conjugate,    'imports': ['jax.numpy.conj']},
    'absv':        {'call': 'abs',      'func': np.abs,          'imports': ['jax.numpy.abs']},
    'sign':        {'call': 'sign',     'func': np.sign,         'imports': ['jax.numpy.sign']},
    'log':         {'call': 'log',      'func': np.log,          'imports': ['jax.numpy.log']},
    'concatenate': {'call': 'concatenate', 'func': np.concatenate, 'imports': ['jax.numpy.concatenate']},
}
