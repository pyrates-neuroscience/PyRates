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

"""This module provides the backend class that should be used to set up any backend in pyrates.
"""

# external imports
from typing import Optional, Any

# meta infos
__author__ = "Richard Gast"
__status__ = "development"


class ComputeGraph(object):
    """Creates a compute graph that contains all nodes in the network plus their recurrent connections.

    Parameters
    ----------
    net_config
        Intermediate representation of the network configuration. For a more detailed description, see the documentation
        of `pyrates.IR.CircuitIR`.
    step_size
        Step-size with which the network should be simulated later on.
        Important for discretizing delays, differential equations, ...
    vectorization
        Defines the mode of automatic parallelization optimization that should be used. Can be `nodes` for lumping all
        nodes together in a vector, `full` for full vectorization of the network, or `None` for no vectorization.
    name
        Name of the network.
    backend
        Backend in which to build the compute graph.
    solver
        Numerical solver to use for differential equations.

    """

    def __new__(cls,
                net_config: Any,
                vectorization: bool = True,
                name: Optional[str] = 'net0',
                backend: str = 'numpy',
                float_precision: str = 'float32',
                **kwargs
                ) -> Any:
        """Instantiates operator.
        """

        if type(net_config) is str:
            net_config = net_config.apply()

        return net_config.compile(vectorization=vectorization, backend=backend, float_precision=float_precision,
                                  **kwargs)
