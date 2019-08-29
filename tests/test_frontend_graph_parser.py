
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
"""
"""

import pytest

__author__ = "Daniel Rose, Richard Gast"
__status__ = "Development"


@pytest.mark.xfail
def test_circuit_from_graph():
    """Runs JRC with tensorflow on population basis
    """

    import numpy as np
    from networkx import MultiDiGraph

    # Comment this out for making GPU available for Tensorflow.
    ###########################################################
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    ##########################################################

    # parameter definition
    ######################

    #np.random.seed(4)

    # general
    step_size = 1e-3
    simulation_time = 1.0
    n_steps = int(simulation_time / step_size)

    # Connection Percentage (If Low that means Connections are few!!)
    sparseness_e = 0.
    sparseness_i = sparseness_e * 0.

    # No_of_JansenRitCircuit
    n_jrcs = 10

    # connectivity parameters
    c_intra = 135.
    c_inter = 0.

    # No of nodes triple the circuit size.
    n_nodes = int(n_jrcs * 3)

    # synapse parameters
    H = []
    tau = []

    for _ in range(n_jrcs):
        H += [[3.25e-3, -22e-3],
              [3.25e-3, 0.],
              [3.25e-3, 0.]]
        tau += [[10e-3, 20e-3],
                [10e-3, 1.],
                [10e-3, 1.]]

    C_e = np.zeros((n_nodes, n_nodes))
    C_i = np.zeros((n_nodes, n_nodes))

    # inter-circuit connectivity
    for i in range(n_jrcs):
        for j in range(n_jrcs - 1):
            if i != j:
                weight_e = np.random.uniform()
                if weight_e > (1 - sparseness_e):
                    C_e[i * 3, j * 3] = np.random.uniform()
                weight_i = np.random.uniform()
                if weight_i > (1 - sparseness_i):
                    C_e[i * 3 + 2, j * 3] = np.random.uniform()

    for i in range(n_nodes):
        ce_sum = np.sum(C_e[i, :])
        if ce_sum > 0:
            C_e[i, :] /= ce_sum
    C_e *= c_inter

    # intra-circuit connectivity
    for i in range(n_jrcs):
        C_e[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = np.array([[0., 0.8 * c_intra, 0.],
                                                              [1.0 * c_intra, 0., 0.],
                                                              [0.25 * c_intra, 0., 0.]])
        C_i[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = np.array([[0., 0., 0.25 * c_intra],
                                                              [0., 0., 0.],
                                                              [0., 0., 0.]])

    # define backend dictionary
    ###########################

    graph = MultiDiGraph()
    for i in range(0, n_jrcs):
        data = {'operators': {'RPO_e_pc.0': {
                                   'equations': ["d/dt * x = H/tau * (m_in + u) - 2. * 1./tau * x - (1./tau)^2 * psp",
                                                 "d/dt * psp = x"],
                                   'inputs': {},
                                   'output': 'psp'},
                              'RPO_i_pc.0': {
                                  'equations': ["d/dt * x = H/tau * (m_in + u) - 2. * 1./tau * x - (1./tau)^2 * psp",
                                                "d/dt * psp = x"],
                                  'inputs': {},
                                  'output': 'psp'},
                              'PRO.0': {
                                  'equations': ["m_out = m_max / (1. + exp(r * (v_th - psp)))"],
                                  'inputs': {'psp': {'sources': ['RPO_e_pc.0', 'RPO_i_pc.0'],
                                                     'reduce_dim': True}},
                                  'output': 'm_out'},
                              },
                'operator_order': ['RPO_e_pc.0', 'RPO_i_pc.0', 'PRO.0'],
                'operator_args': {'RPO_e_pc.0/m_in': {'vtype': 'state_var',
                                                              'dtype': 'float32',
                                                              'shape': (),
                                                              'value': 0.},
                                  'RPO_i_pc.0/m_in': {'vtype': 'state_var',
                                                              'dtype': 'float32',
                                                              'shape': (),
                                                              'value': 0.},
                                  'PRO.0/m_out': {'vtype': 'state_var',
                                                         'dtype': 'float32',
                                                         'shape': (),
                                                         'value': 0.16},
                                  'PRO.0/psp': {'vtype': 'state_var',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': 0.},
                                  'RPO_e_pc.0/psp': {'vtype': 'state_var',
                                                             'dtype': 'float32',
                                                             'shape': (),
                                                             'value': 0.},
                                  'RPO_i_pc.0/psp': {'vtype': 'state_var',
                                                             'dtype': 'float32',
                                                             'shape': (),
                                                             'value': 0.},
                                  'RPO_e_pc.0/x': {'vtype': 'state_var',
                                                           'dtype': 'float32',
                                                           'shape': (),
                                                           'value': 0.},
                                  'RPO_i_pc.0/x': {'vtype': 'state_var',
                                                           'dtype': 'float32',
                                                           'shape': (),
                                                           'value': 0.},
                                  'RPO_e_pc.0/H': {'vtype': 'constant',
                                                           'dtype': 'float32',
                                                           'shape': (),
                                                           'value': 3.25e-3},
                                  'RPO_i_pc.0/H': {'vtype': 'constant',
                                                           'dtype': 'float32',
                                                           'shape': (),
                                                           'value': -22e-3},
                                  'RPO_e_pc.0/tau': {'vtype': 'constant',
                                                             'dtype': 'float32',
                                                             'shape': (),
                                                             'value': 10e-3},
                                  'RPO_i_pc.0/tau': {'vtype': 'constant',
                                                             'dtype': 'float32',
                                                             'shape': (),
                                                             'value': 20e-3},
                                  'PRO.0/m_max': {'vtype': 'constant',
                                                         'dtype': 'float32',
                                                         'shape': (),
                                                         'value': 5.},
                                  'PRO.0/r': {'vtype': 'constant',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 560.},
                                  'PRO.0/v_th': {'vtype': 'constant',
                                                        'dtype': 'float32',
                                                        'shape': (),
                                                        'value': 6e-3},
                                  'RPO_i_pc.0/u': {'vtype': 'constant',
                                                           'dtype': 'float32',
                                                           'shape': (),
                                                           'value': 0.},
                                  'RPO_e_pc.0/u': {'vtype': 'placeholder',
                                                           'dtype': 'float32',
                                                           'shape': (),
                                                           'value': 220.},
                                  },
                'inputs': {}
                }
        graph.add_node(f'PC.{i}', **data)

        data = {'operators': {'RPO_e.0': {
                                   'equations': ["d/dt * x = H/tau * (m_in + u) - 2. * 1./tau * x - (1./tau)^2 * psp",
                                                 "d/dt * psp = x"],
                                   'inputs': {},
                                   'output': 'psp'},
                              'PRO.0': {
                                  'equations': ["m_out = m_max / (1. + exp(r * (v_th - psp)))"],
                                  'inputs': {'psp': {'sources': ['RPO_e.0'],
                                                     'reduce_dim': False}},
                                  'output': 'm_out'}},
                'operator_order': ['RPO_e.0', 'PRO.0'],
                'operator_args': {'RPO_e.0/m_in': {'vtype': 'state_var',
                                                            'dtype': 'float32',
                                                            'shape': (),
                                                            'value': 0.},
                                  'PRO.0/m_out': {'vtype': 'state_var',
                                                         'dtype': 'float32',
                                                         'shape': (),
                                                         'value': 0.16},
                                  'PRO.0/psp': {'vtype': 'state_var',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': 0.},
                                  'RPO_e.0/psp': {'vtype': 'state_var',
                                                           'dtype': 'float32',
                                                           'shape': (),
                                                           'value': 0.},
                                  'RPO_e.0/x': {'vtype': 'state_var',
                                                         'dtype': 'float32',
                                                         'shape': (),
                                                         'value': 0.},
                                  'RPO_e.0/H': {'vtype': 'constant',
                                                         'dtype': 'float32',
                                                         'shape': (),
                                                         'value': 3.25e-3},
                                  'RPO_e.0/tau': {'vtype': 'constant',
                                                           'dtype': 'float32',
                                                           'shape': (),
                                                           'value': 10e-3},
                                  'PRO.0/m_max': {'vtype': 'constant',
                                                         'dtype': 'float32',
                                                         'shape': (),
                                                         'value': 5.},
                                  'PRO.0/r': {'vtype': 'constant',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 560.},
                                  'PRO.0/v_th': {'vtype': 'constant',
                                                        'dtype': 'float32',
                                                        'shape': (),
                                                        'value': 6e-3},
                                  'RPO_e.0/u': {'vtype': 'constant',
                                                         'dtype': 'float32',
                                                         'shape': (),
                                                         'value': 0.},
                                  },
                'inputs': {}
                }
        graph.add_node(f'EIN.{i}', **data)

        data = {'operators': {'RPO_e.0': {
                                   'equations': ["d/dt * x = H/tau * (m_in + u) - 2. * 1./tau * x - (1./tau)^2 * psp",
                                                 "d/dt * psp = x"],
                                   'inputs': {},
                                   'output': 'psp'},
                              'PRO.0': {
                                  'equations': ["m_out = m_max / (1. + exp(r * (v_th - psp)))"],
                                  'inputs': {'psp': {'sources': ['RPO_e.0'],
                                                     'reduce_dim': False}},
                                  'output': 'm_out'}},
                'operator_order': ['RPO_e.0', 'PRO.0'],
                'operator_args': {'RPO_e.0/m_in': {'vtype': 'state_var',
                                                            'dtype': 'float32',
                                                            'shape': (),
                                                            'value': 0.},
                                  'PRO.0/m_out': {'vtype': 'state_var',
                                                         'dtype': 'float32',
                                                         'shape': (),
                                                         'value': 0.16},
                                  'PRO.0/psp': {'vtype': 'state_var',
                                                       'dtype': 'float32',
                                                       'shape': (),
                                                       'value': 0.},
                                  'RPO_e.0/psp': {'vtype': 'state_var',
                                                           'dtype': 'float32',
                                                           'shape': (),
                                                           'value': 0.},
                                  'RPO_e.0/x': {'vtype': 'state_var',
                                                         'dtype': 'float32',
                                                         'shape': (),
                                                         'value': 0.},
                                  'RPO_e.0/H': {'vtype': 'constant',
                                                         'dtype': 'float32',
                                                         'shape': (),
                                                         'value': 3.25e-3},
                                  'RPO_e.0/tau': {'vtype': 'constant',
                                                           'dtype': 'float32',
                                                           'shape': (),
                                                           'value': 10e-3},
                                  'PRO.0/m_max': {'vtype': 'constant',
                                                         'dtype': 'float32',
                                                         'shape': (),
                                                         'value': 5.},
                                  'PRO.0/r': {'vtype': 'constant',
                                                     'dtype': 'float32',
                                                     'shape': (),
                                                     'value': 560.},
                                  'PRO.0/v_th': {'vtype': 'constant',
                                                        'dtype': 'float32',
                                                        'shape': (),
                                                        'value': 6e-3},
                                  'RPO_e.0/u': {'vtype': 'constant',
                                                         'dtype': 'float32',
                                                         'shape': (),
                                                         'value': 0.},
                                  },
                'inputs': {}
                }
        graph.add_node(f'IIN.{i}', **data)

    # For the Un_vectorized Connection Dict.
    ########################################

    for a in range(0, n_nodes):
        for b in range(0, n_nodes):

            # source stuff
            if b % 3 == 2:
                source = f'IIN.{int(b/3)}'
                c = C_i[a, b]
            elif b % 3 == 1:
                source = f'EIN.{int(b/3)}'
                c = C_e[a, b]
            else:
                source = f'PC.{int(b/3)}'
                c = C_e[a, b]

            if c != 0:
                edge = {}
                if a % 3 == 0:
                    target = f'PC.{int(a/3)}'
                    if source.split('.')[0] == 'IIN':
                        edge['target_var'] = 'RPO_i_pc.0/m_in'
                    else:
                        edge['target_var'] = 'RPO_e_pc.0/m_in'
                elif a % 3 == 1:
                    target = f'EIN.{int(a/3)}'
                    edge['target_var'] = 'RPO_e.0/m_in'
                else:
                    target = f'IIN.{int(a/3)}'
                    edge['target_var'] = 'RPO_e.0/m_in'

                edge['source_var'] = 'PRO.0/m_out'
                edge['weight'] = c
                if int(a/3) == int(b/3):
                    edge['delay'] = 0.
                else:
                    edge['delay'] = np.random.uniform(0., 6e-3)

                graph.add_edge(source, target, **edge)

    from pyrates.frontend.nxgraph import to_circuit

    circuit = to_circuit(graph)

    assert circuit
