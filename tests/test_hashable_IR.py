
# -*- coding: utf-8 -*-
#
#
# PyRates software framework for flexible implementation of neural
# network models and simulations. See also:
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
""" Tests for new structure of IR classes that separate the defining structural features and the values assigned to them
for computations as well as efficient gathering of information for vectorization.
"""

__author__ = "Daniel Rose"
__status__ = "Development"

import pytest


def setup_module():
    print("\n")
    print("===========================")
    print("| Test Suite: Hashable IR |")
    print("===========================")


def test_operator_caching():
    """Test whether an operator IR that is created is cached properly and if an identical operator definition
    actually produces the same instance."""

    # 1: Load operator and test, if it is cached properly

    # 2: Create same operator again using the same unique structural features and compare by identity and content


def test_operator_graph_caching():
    """Test whether an operator graph IR that is created is cached properly and if an identical operator definition
    actually produces the same instance."""

    # 1: Load operator graph and test, if it is cached properly

    # 2: Recreate same operator graph using the same unique structural features and compare by identity and content

    # 3: Test if operator references are collected properly
