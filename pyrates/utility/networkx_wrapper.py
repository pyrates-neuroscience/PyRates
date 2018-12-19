
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
"""Defines a few custom functions on the backend graph for convenience.
"""

# external packages
from networkx import MultiDiGraph

# pyrates internal imports
from pyrates.population import Population

# meta infos
__author__ = "Daniel Rose, Richard Gast"
__status__ = "Development"


####################
# networkx wrapper #
####################


class WrappedMultiDiGraph(MultiDiGraph):
    """Wrapper for MultiDiGraph that has a few convenience customizations."""

    def add_edge(self, source, target, weight=1, delay=0, synapse=None):

        if synapse is None:

            # connect source to target population (directly)
            source_pop = self.nodes[source]["data"]  # type: Population
            source_pop.connect(self.nodes[target]["data"], weight, delay)

        else:

            # connect source population to synapse on target population
            source_pop = self.nodes[source]["data"]  # type: Population
            source_pop.connect(synapse, weight, delay)

        super().add_edge(source, target, weight=weight, delay=delay, synapse=synapse)



