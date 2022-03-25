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

"""Contains fortran function definitions that may be used for PyRates model equations.
"""

from typing import Union
import numpy as np

# function definitions
######################


# sigmoid function
sigmoid = lambda x: 1./(1. + np.exp(-x))


# function for calculating the logistic function of an nd-array
def get_sigmoid_def(idx: int, out_shape: Union[tuple, str] = '', dtype: str = 'real') -> tuple:
    isarray = len(out_shape) > 0
    fname = f"fsigmoid_{idx}"
    sigmoid_n = f"s = size(x)\ndo n=1,s\n  {fname}(n) = 1 / (1 + exp(-x(n)))\nend do"
    sigmoid_0 = f"{fname} = 1 / (1 + exp(-x))"

    func = f"""
function {fname}(x)

implicit none

{dtype} :: {fname}{out_shape if isarray else ''}
{dtype} :: x{out_shape if isarray else ''}
{"integer :: n, s" if isarray else ""}

{sigmoid_n if isarray else sigmoid_0}

end function {fname}
    """
    return fname, func


# wrapper function for interpolating a 1d-array
def get_interp_def(idx: int, out_shape: Union[tuple, str] = '', dtype: str = 'real') -> tuple:
    fname = f"finterp_{idx}"
    func = f"""
function {fname}(x_new,x,y)

implicit none 

{dtype} :: {fname}
{dtype} :: x(:), y(:), x_new
{dtype} :: x_inc
integer :: n, s

s = size(x) 

if (x_new < x(1)) then
  {fname} = y(1)
else if (x_new > x(s)) then
  {fname} = y(s)
else
  do n = 1, s 
    if (x(n) > x_new) exit 
  end do
  if (n == 1) then 
    {fname} = y(1)
  else if (n == s+1) then
    {fname} = y(s)
  else
    x_inc = (x_new - x(n-1)) / (x(n) - x(n-1))
    {fname} = y(n) + x_inc*(y(n) - y(n-1))
  end if 
end if 

end function {fname}
"""
    return fname, func


# dictionary for backend import
###############################

fortran_funcs = {
    'max': {'call': 'max', 'func': np.maximum},
    'min': {'call': 'min', 'func': np.minimum},
    'round': {'call': 'nint', 'func': np.round},
    'sum': {'call': 'sum', 'func': np.sum},
    'matmul': {'call': 'matmul', 'func': np.dot},
    'matvec': {'call': 'matmul', 'func': np.dot},
    'roll': {'call': 'cshift', 'func': np.roll},
    'tanh': {'call': 'tanh', 'func': np.tanh},
    'sinh': {'call': 'sinh', 'func': np.sinh},
    'cosh': {'call': 'cosh', 'func': np.cosh},
    'arctan': {'call': 'atan', 'func': np.arctan},
    'arcsin': {'call': 'asin', 'func': np.arcsin},
    'arccos': {'call': 'acos', 'func': np.arccos},
    'sin': {'call': 'sin', 'func': np.sin},
    'cos': {'call': 'cos', 'func': np.cos},
    'tan': {'call': 'tan', 'func': np.tan},
    'exp': {'call': 'exp', 'func': np.exp},
    'sigmoid': {'call': get_sigmoid_def, 'func': sigmoid},
    'interp': {'call': get_interp_def, 'func': np.interp},
    'real': {'call': 'realpart', 'func': np.real},
    'imag': {'call': 'imagpart', 'func': np.imag},
    'conj': {'call': 'conjg', 'func': np.conjugate},
    'absv': {'call': 'abs', 'func': np.abs},
    'log': {'call': 'log', 'func': np.log},
}
