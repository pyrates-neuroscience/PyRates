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

"""Contains Matlab function definitions that may be used for PyRates model equations.
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
function y = sigmoid(x)
    y = 1.0 ./ (1.0 + exp(-x));
end
"""

interp = """
function y_new = interp(x_new, x, y)
    y_new = interp1(x, y, x_new);
end
"""

matmul = """
function z = matmul(x,y)
    z = mtimes(y,x');
end
"""

concat = """
function z = concatenate(x,dim)
    z = cat(dim,x{:});
end
"""

# dictionary for backend import
###############################

matlab_funcs = {
    'max': {'call': 'max', 'func': np.maximum, 'imports': []},
    'min': {'call': 'min', 'func': np.minimum, 'imports': []},
    'matmul': {'call': 'matmul', 'func': np.dot, 'def': matmul, 'imports': []},
    'matvec': {'call': 'matmul', 'func': np.dot, 'def': matmul, 'imports': []},
    'roll': {'call': 'circshift', 'func': np.roll, 'imports': []},
    'randn': {'call': 'randn', 'func': np.random.randn, 'imports': []},
    'sigmoid': {'call': 'sigmoid', 'func': sigmoid_func, 'def': sigmoid_def, 'imports': []},
    'interp': {'call': 'interp', 'func': np.interp, 'def': interp, 'imports': []},
    'conj': {'call': 'conj', 'func': np.conjugate, 'imports': []},
    'concatenate': {'call': 'concatenate', 'func': np.concatenate, 'def': concat}
}
