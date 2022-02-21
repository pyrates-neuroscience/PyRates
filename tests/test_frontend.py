
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
""" Tests for the parser that translates circuits and components defined in YAML into the intermediate
python representation.
"""

__author__ = "Daniel Rose"
__status__ = "Development"

import pytest


def setup_module():
    print("\n")
    print("===========================")
    print("| Test Suite: YAML Parser |")
    print("===========================")


@pytest.mark.parametrize("operator", ["model_templates.base_templates.li_op",
                                      "model_templates.base_templates.alpha_op",
                                      "model_templates.base_templates.sigmoid_op",
                                      "model_templates.neural_mass_models.jansenrit.rpo_e_in",
                                      "model_templates.neural_mass_models.qif.qif_sfa_op",
                                      "model_templates.oscillators.kuramoto.sin_op"
                                      ])
def test_import_operator_templates(operator):
    """test basic (vanilla) YAML parsing using ruamel.yaml (for YAML 1.2 support)"""
    from pyrates import OperatorTemplate
    from pyrates.frontend.template import template_cache, clear_cache
    clear_cache()

    template = OperatorTemplate.from_yaml(operator)  # type: OperatorTemplate

    assert template.path in template_cache

    cached_template = template_cache[operator]  # type: OperatorTemplate
    assert template is cached_template
    assert template.path == cached_template.path
    assert template.equations == cached_template.equations
    assert repr(template) == repr(cached_template) == f"<OperatorTemplate '{operator}'>"


def test_full_jansen_rit_circuit_template_load():
    """Test a simple circuit template, including all nodes, edges and operators to be loaded."""

    path = "model_templates.neural_mass_models.jansenrit.JRC"
    from pyrates.frontend.template.circuit import CircuitTemplate
    from pyrates.frontend.template.node import NodeTemplate
    from pyrates.frontend.template.operator import OperatorTemplate
    from pyrates.frontend.template import template_cache, clear_cache
    clear_cache()

    template = CircuitTemplate.from_yaml(path)

    # test, whether circuit is in loader cache
    assert template is template_cache[path]

    # test, whether node templates have been loaded successfully
    nodes = {"pc": "model_templates.neural_mass_models.jansenrit.PC",
             "ein": "model_templates.neural_mass_models.jansenrit.IN",
             "iin": "model_templates.neural_mass_models.jansenrit.IN"}

    for key, value in nodes.items():
        assert isinstance(template.nodes[key], NodeTemplate)
        assert template.nodes[key] is template_cache[value]
        # test operators in node templates
        for op in template.nodes[key].operators:
            assert op.path in template_cache
            assert isinstance(op, OperatorTemplate)


def test_edge_definition_via_matrix():
    """Test, if CircuitTemplate.add_edges_from_matrix works as expected."""

    from pyrates import CircuitTemplate, NodeTemplate, EdgeTemplate
    import numpy as np
    from copy import deepcopy

    # define the network without edges
    node = NodeTemplate.from_yaml("model_templates.oscillators.kuramoto.phase_pop")
    node_names = ['p1', 'p2', 'p3']
    n = len(node_names)
    circuit = CircuitTemplate(name='delay_coupled_kmos', nodes={key: node for key in node_names})

    # add the edges
    edge = EdgeTemplate.from_yaml("model_templates.oscillators.kuramoto.sin_edge")
    weights = np.random.randn(n, n)
    delays = np.random.uniform(low=1, high=2, size=(n, n))
    edge_attr = {'sin_edge/coupling_op/theta_s': 'source', 'sin_edge/coupling_op/theta_t': 'p2/phase_op/theta',
                 'delay': delays}
    circuit.add_edges_from_matrix(source_var='phase_op/theta', target_var='phase_op/net_in', nodes=node_names,
                                  weight=weights, template=edge, edge_attr=edge_attr)

    # test whether edges have been added as expected
    edge_attr_tmp = deepcopy(edge_attr)
    edge_attr_tmp['weight'] = weights[2, 1]
    edge_attr_tmp['delay'] = delays[2, 1]
    assert len(circuit.edges) == int(n**2)
    assert ('p3/phase_op/theta', 'p2/phase_op/net_in', edge, edge_attr_tmp) in circuit.edges

    # perform short simulation to ensure that the network has been constructed correctly
    circuit.run(simulation_time=1.0, step_size=1e-4, outputs=['all/phase_op/theta'])


def test_circuit_instantiation():
    """Test, if apply() functions all work properly"""
    path = "model_templates.neural_mass_models.jansenrit.JRC"
    from pyrates import clear_frontend_caches
    from pyrates.frontend import template
    clear_frontend_caches()

    circuit = template.from_yaml(path)

    circuit.apply()
    ir = circuit.intermediate_representation

    # TODO: rework the below tests.
    # test whether edge operator is properly connected with network
    # assert circuit.edges[('LCEdge', 'JR_PC', 0)]
    # assert circuit.edges[('LCEdge', 'JR_PC', 1)]
    # assert len(circuit.edges[('LCEdge', 'JR_IIN', 0)]['target_idx']) == 2
    # assert circuit.edges[('JR_IIN', 'LCEdge', 0)]
    # assert circuit.edges[('JR_IIN', 'LCEdge', 1)]
    # assert len(circuit.edges[('JR_PC', 'LCEdge', 0)]['target_idx']) == 2
    #
    # # now test, if JR_EIN and JR_IIN have been vectorized into a single operator graph
    # assert len(circuit["JR_IIN"]['JansenRitPRO']['variables']['m_out']['value']) == 2
    #
    # # now test, if the references are collected properly
    # for node in circuit_template.nodes:
    #     if node in circuit_template._ir_map:
    #         node = circuit_template._ir_map[node]
    #     assert node in circuit
    # circuit_template.clear()
    #
    # # verify that .apply also understands value updates to nodes
    # value_dict = {"JR_PC/JansenRitExcitatorySynapseRCO/h": 0.1234}
    # clear_frontend_caches()
    # circuit_template = template.from_yaml(path)
    # circuit2 = circuit_template.apply(node_values=value_dict)[0]
    # var = circuit2["JR_PC"]["JansenRitExcitatorySynapseRCO"]['variables']['h']
    # circuit_template.clear()
    # assert float(var['value']) - 0.1234 == pytest.approx(0, rel=1e-4, abs=1e-4)


def test_multi_circuit_instantiation():
    """Test, if a circuit with subcircuits is also working."""
    path = "model_templates.neural_mass_models.jansenrit.JRC_2delaycoupled"
    from pyrates import clear_frontend_caches, CircuitTemplate
    clear_frontend_caches()

    circuit = CircuitTemplate.from_yaml(path)
    assert circuit


def test_equation_alteration():
    """Test, if properties of a template that mean to alter a certain parent equation are treated correctly"""

    path = "model_templates.neural_mass_models.jansenrit.rpo_e_in"
    # template replaces the component "m_in" with "(m_in + u)" in the equation "X' = h*m_in/tau - 2*X/tau - V/tau**2"
    from pyrates.frontend.template.operator import OperatorTemplate

    template = OperatorTemplate.from_yaml(path)
    operator, _ = template.apply()

    assert operator.equations[1] == "X' = h*(m_in + u)/tau - 2*X/tau - V/tau**2"
