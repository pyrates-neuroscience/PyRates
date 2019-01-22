
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


# @pytest.mark.parametrize("section", ["axon", "population", "synapse"])
# def test_parse_operator_templates(section):
#     """test basic (vanilla) YAML parsing using ruamel.yaml (for YAML 1.2 support)"""
#     from ruamel.yaml import YAML
#     import os
#
#     yaml = YAML(typ="safe", pure=True)
#
#     path = os.path.abspath(f"pyrates/{section}/")
#     base_filename = f"{section}.yaml"
#     template_filename = f"templates.yaml"
#
#     # load base template
#     filepath = os.path.join(path, base_filename)
#     with open(filepath, "r") as file:
#         base = yaml.load(file)
#
#     # load further templates
#     filepath = os.path.join(path, template_filename)
#     with open(filepath, "r") as file:
#         templates = yaml.load(file)


@pytest.mark.parametrize("operator", ["pyrates.examples.jansen_rit.axon.axon.PotentialToRateOperator",
                                      "pyrates.examples.jansen_rit.axon.templates.SigmoidPRO",
                                      "pyrates.examples.jansen_rit.axon.templates.JansenRitPRO",
                                      "pyrates.examples.jansen_rit.population.population.CurrentToPotentialOperator",
                                      "pyrates.examples.jansen_rit.synapse.synapse.RateToCurrentOperator"
                                      ])
def test_import_operator_templates(operator):
    """test basic (vanilla) YAML parsing using ruamel.yaml (for YAML 1.2 support)"""
    from pyrates.frontend.template.operator import OperatorTemplateLoader
    from pyrates.frontend.template.operator import OperatorTemplate
    from pyrates.frontend import template_from_yaml_file

    template = template_from_yaml_file(operator, OperatorTemplate)  # type: OperatorTemplate

    assert template.path in OperatorTemplateLoader.cache

    cached_template = OperatorTemplateLoader.cache[operator]  # type: OperatorTemplate
    assert template is cached_template
    assert template.path == cached_template.path
    assert template.equations == cached_template.equations
    assert repr(template) == repr(cached_template) == f"<OperatorTemplate '{operator}'>"
    # assert template == TemplateLoader.cache[template.path]


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


# @pytest.mark.parametrize("circuit", ["pyrates.circuit.templates.JansenRitCircuit",
#                                      "pyrates.circuit.circuit.BaseCircuit"
#                                      ])
# def test_import_circuit_templates(circuit):
#     """test import of circuit templates"""
#
#     from pyrates.circuit.circuit import CircuitTemplateLoader
#
#     template = CircuitTemplate.from_yaml(circuit)  # type: CircuitTemplate
#
#     assert template.path in CircuitTemplateLoader.cache
#
#     cached_template = CircuitTemplateLoader.cache[circuit]  # type: CircuitTemplate
#
#     assert template is cached_template
#     assert template.path == cached_template.path
#     assert template.nodes == cached_template.nodes
#     assert template.edges == cached_template.edges
#
#     for value in template.coupling.values():
#         assert isinstance(value, OperatorTemplate)
#     for value in template.nodes.values():
#         assert isinstance(value, NodeTemplate)
#     assert repr(template) == repr(cached_template) == f"<CircuitTemplate '{circuit}'>"


def test_full_jansen_rit_circuit_template_load():
    """Test a simple circuit template, including all nodes and operators to be loaded."""

    path = "pyrates.examples.jansen_rit.circuit.JansenRitCircuit"
    from pyrates.frontend.template.circuit import CircuitTemplate
    from pyrates.frontend.template.edge.edge import EdgeTemplate
    from pyrates.frontend.template.node import NodeTemplate
    from pyrates.frontend.template.operator import OperatorTemplate

    template = CircuitTemplate.from_yaml(path)

    # test, whether circuit is in loader cache
    from pyrates.frontend.yaml import TemplateLoader
    assert template is TemplateLoader.cache[path]

    # test, whether node templates have been loaded successfully
    nodes = {"JR_PC": "pyrates.examples.jansen_rit.population.templates.JansenRitPC",
             "JR_IIN": "pyrates.examples.jansen_rit.population.templates.JansenRitIN",
             "JR_EIN": "pyrates.examples.jansen_rit.population.templates.JansenRitIN"}

    for key, value in nodes.items():
        assert isinstance(template.nodes[key], NodeTemplate)
        assert template.nodes[key] is TemplateLoader.cache[value]
        # test operators in node templates
        for op in template.nodes[key].operators:
            assert op.path in TemplateLoader.cache
            assert isinstance(op, OperatorTemplate)

    # test, whether coupling operator has been loaded correctly
    coupling_path = "pyrates.examples.jansen_rit.edges.LinearCouplingOperator"
    edge_temp = template.edges[0][2]
    assert isinstance(edge_temp, EdgeTemplate)
    assert list(edge_temp.operators)[0] is TemplateLoader.cache[coupling_path]

    assert repr(template) == f"<CircuitTemplate '{path}'>"


def test_circuit_instantiation():
    """Test, if apply() functions all work properly"""
    path = "pyrates.examples.jansen_rit.circuit.JansenRitCircuit"
    from pyrates.frontend.template.circuit import CircuitTemplate

    template = CircuitTemplate.from_yaml(path)

    circuit = template.apply()
    # used to be: test if two edges refer to the same coupling operator by comparing ids
    # this is why we referenced by "operator"
    # now: compare operators directly
    edge_to_compare = circuit.edges[('JR_PC.0', 'JR_EIN.0', 0)]["edge_ir"]
    for op_key, op in circuit.edges[("JR_PC.0", "JR_IIN.0", 0)]["edge_ir"].op_graph.nodes(data=True):
        if op_key in edge_to_compare:
            assert op["operator"].equations == edge_to_compare[op_key].equations


def test_multi_circuit_instantiation():
    """Test, if a circuit with subcircuits is also working."""
    path = "pyrates.examples.jansen_rit.circuit.MultiJansenRitCircuit"
    from pyrates.frontend.template.circuit import CircuitTemplate

    template = CircuitTemplate.from_yaml(path)

    circuit = template.apply()
    assert circuit


def test_equation_alteration():
    """Test, if properties of a template that mean to alter a certain parent equation are treated correctly"""

    path = "pyrates.examples.jansen_rit.population.templates.InstantaneousCPO"
    # this template removes the component "L_m * " from the base equation "L_m * V = k * I"
    from pyrates.frontend.template.operator import OperatorTemplate

    template = OperatorTemplate.from_yaml(path)

    operator, values = template.apply()

    assert operator.equations == ["V = k * I"]


def test_network_def_workaround():
    path = "pyrates.examples.jansen_rit.circuit.JansenRitCircuit"
    from pyrates.frontend.template.circuit import CircuitTemplate

    template = CircuitTemplate.from_yaml(path)

    circuit = template.apply()

    from pyrates.frontend.nxgraph import from_circuit
    nd = from_circuit(circuit, revert_node_names=True)
    operator_order = ['LinearCouplingOperator.3',
                      'LinearCouplingOperator.1',
                      'JansenRitExcitatorySynapseRCO.0',
                      'JansenRitInhibitorySynapseRCO.0',
                      'JansenRitCPO.0',
                      'JansenRitPRO.0']
    inputs = {}

    cpo_i = {'dtype': 'float32',
             'shape': (),
             'vtype': 'state_var',
             'value': 0.}  # 0. is the new default value

    jr_cpo = {'equations': ['V = k * I'],
              'inputs': {'I': {'reduce_dim': True,
                               'sources': ['JansenRitExcitatorySynapseRCO.0',
                                           'JansenRitInhibitorySynapseRCO.0']}},
              'output': 'V'}
    # assert dict(nd.nodes["JR_PC.0"]) == JR_PC
    node = nd.nodes["JR_PC.0"]
    edge = {'delay': 0,
            'source_var': 'JansenRitPRO.0/m_out',
            'target_var': 'LinearCouplingOperator.3/m_out',
            'weight': 1}
    # assert node["operator_order"] == operator_order
    assert node["inputs"] == inputs
    assert node["operator_args"]['JansenRitCPO.0/I'] == cpo_i
    # assert node["operators"]['JansenRitExcitatorySynapseRCO.0']["inputs"]["m_in"]["sources"] == [
    #     'LinearCouplingOperator.3']
    # assert node["operators"]['JansenRitCPO.0'] == jr_cpo
    # assert dict(nd.edges[('JR_EIN.0', 'JR_PC.0', 0)]) == edge


def test_yaml_dump():
    from pyrates.frontend.template.circuit import CircuitTemplate
    circuit = CircuitTemplate.from_yaml("pyrates.examples.jansen_rit.circuit.JansenRitCircuit").apply()
    from pyrates.frontend.yaml import from_circuit
    from_circuit(circuit, "../output/yaml_dump.yaml", "DumpedCircuit")

    # reload saved circuit
    saved_circuit = CircuitTemplate.from_yaml("../output/yaml_dump.yaml.DumpedCircuit").apply()
    assert saved_circuit
