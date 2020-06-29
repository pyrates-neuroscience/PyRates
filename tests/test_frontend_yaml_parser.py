
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


@pytest.mark.parametrize("operator", ["model_templates.jansen_rit.axon.axon.PotentialToRateOperator",
                                      "model_templates.jansen_rit.axon.templates.SigmoidPRO",
                                      "model_templates.jansen_rit.axon.templates.JansenRitPRO",
                                      "model_templates.jansen_rit.population.population.CurrentToPotentialOperator",
                                      "model_templates.jansen_rit.synapse.synapse.RateToCurrentOperator"
                                      ])
def test_import_operator_templates(operator):
    """test basic (vanilla) YAML parsing using ruamel.yaml (for YAML 1.2 support)"""
    from pyrates.frontend.template.operator import OperatorTemplate
    from pyrates.frontend.template import template_cache, clear_cache
    clear_cache()

    template = OperatorTemplate.from_yaml(operator)  # type: OperatorTemplate

    assert template.path in template_cache

    cached_template = template_cache[operator]  # type: OperatorTemplate
    assert template is cached_template
    assert template.path == cached_template.path
    assert template.equations == cached_template.equations
    assert repr(template) == repr(cached_template) == f"<OperatorTemplate '{operator}'>"


# @pytest.mark.parametrize("node", ["pyrates.population.templates.JansenRitIN",
#                                   "pyrates.population.templates.JansenRitPC",
#                                   "pyrates.population.population.NeuralMass"
#                                   ])
# def test_import_node_templates(node):
#     """test import of node templates"""
#
#     from pyrates.utility.yaml_parser import TemplateLoader
#     from pyrates.node.node import NodeTemplateLoader
#
#     template = NodeTemplate.from_yaml(node)  # type: NodeTemplate
#
#     assert template.path in TemplateLoader.cache  # just to check if cache is really shared among subclasses
#
#     cached_template = NodeTemplateLoader.cache[node]
#     assert template is cached_template
#     assert template.path == cached_template.path
#     for op in template.operators:
#         assert isinstance(op, OperatorTemplate)
#     assert template.operators == cached_template.operators
#     assert repr(template) == repr(cached_template) == f"<NodeTemplate '{node}'>"


def test_full_jansen_rit_circuit_template_load():
    """Test a simple circuit template, including all nodes and operators to be loaded."""

    path = "model_templates.jansen_rit.circuit.JansenRitCircuit"
    from pyrates.frontend.template.circuit import CircuitTemplate
    from pyrates.frontend.template.edge import EdgeTemplate
    from pyrates.frontend.template.node import NodeTemplate
    from pyrates.frontend.template.operator import OperatorTemplate
    from pyrates.frontend.template import template_cache, clear_cache
    clear_cache()

    template = CircuitTemplate.from_yaml(path)

    # test, whether circuit is in loader cache
    assert template is template_cache[path]

    # test, whether node templates have been loaded successfully
    nodes = {"JR_PC": "model_templates.jansen_rit.population.templates.JansenRitPC",
             "JR_IIN": "model_templates.jansen_rit.population.templates.JansenRitIN",
             "JR_EIN": "model_templates.jansen_rit.population.templates.JansenRitIN"}

    for key, value in nodes.items():
        assert isinstance(template.nodes[key], NodeTemplate)
        assert template.nodes[key] is template_cache[value]
        # test operators in node templates
        for op in template.nodes[key].operators:
            assert op.path in template_cache
            assert isinstance(op, OperatorTemplate)

    # test, whether coupling operator has been loaded correctly
    coupling_path = "model_templates.jansen_rit.edges.LinearCouplingOperator"
    edge_temp = template.edges[0][2]
    assert isinstance(edge_temp, EdgeTemplate)
    assert list(edge_temp.operators)[0] is template_cache[coupling_path]

    assert repr(template) == f"<CircuitTemplate '{path}'>"


def test_circuit_instantiation():
    """Test, if apply() functions all work properly"""
    path = "model_templates.jansen_rit.circuit.JansenRitCircuit"
    from pyrates.frontend import template
    template.clear_cache()

    circuit_template = template.from_yaml(path)

    circuit = circuit_template.apply()
    # used to be: test if two edges refer to the same coupling operator by comparing ids
    # this is why we referenced by "operator"
    # now: compare operators directly
    edge_to_compare = circuit.edges[('JR_PC', 'JR_EIN', 0)]["edge_ir"]
    for op_key, op in circuit.edges[("JR_PC", "JR_IIN", 0)]["edge_ir"].op_graph.nodes(data=True):
        if op_key in edge_to_compare:
            assert op["operator"].equations == edge_to_compare[op_key].equations

    # now test, if JR_EIN and JR_IIN refer to the same operator graph
    assert circuit["JR_EIN"].op_graph is circuit["JR_IIN"].op_graph

    # now test, if the references are collected properly
    for key, data in circuit.nodes(data=True):
        node = data["node"]
        assert node in circuit._reference_map[node.op_graph]

    assert len(circuit._reference_map[circuit["JR_EIN"].op_graph]) == 2

    # verify that .apply also understands value updates to nodes
    value_dict = {"JR_PC/JansenRitExcitatorySynapseRCO/h": 0.1234}
    circuit2 = circuit_template.apply(node_values=value_dict)
    node = circuit2["JR_PC"]
    assert node.values["JansenRitExcitatorySynapseRCO"]["h"] == 0.1234


def test_multi_circuit_instantiation():
    """Test, if a circuit with subcircuits is also working."""
    path = "model_templates.jansen_rit.circuit.MultiJansenRitCircuit"
    from pyrates.frontend import template as tpl
    tpl.clear_cache()

    template = tpl.from_yaml(path)

    circuit = template.apply()
    assert circuit


def test_equation_alteration():
    """Test, if properties of a template that mean to alter a certain parent equation are treated correctly"""

    path = "model_templates.jansen_rit.population.templates.InstantaneousCPO"
    # this template removes the component "L_m * " from the base equation "L_m * V = k * I"
    from pyrates.frontend.template.operator import OperatorTemplate

    template = OperatorTemplate.from_yaml(path)

    operator, values = template.apply()

    assert operator.equations == ("V = k * I",)


def test_yaml_dump():
    """Test the functionality to dump an object to YAML"""
    from pyrates.frontend import fileio

    with pytest.raises(AttributeError):
        fileio.dump("no_to_dict()", "random_art", "yaml")

    from pyrates.frontend.template.circuit import CircuitTemplate
    circuit = CircuitTemplate.from_yaml("model_templates.jansen_rit.circuit.JansenRitCircuit").apply()

    with pytest.raises(ValueError):
        fileio.dump(circuit, "output/yaml_dump.yaml", "yml")

    with pytest.raises(TypeError):
        fileio.dump(circuit, "output/yaml_dump.yaml", "yaml")


    fileio.dump(circuit, "output/yaml_dump.yaml", "yaml", "DumpedCircuit")

    # reload saved circuit
    saved_circuit = CircuitTemplate.from_yaml("output/yaml_dump/DumpedCircuit").apply()
    assert saved_circuit

    # ToDo: figure out a simple way to compare both instances
