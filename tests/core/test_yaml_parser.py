""" Tests for the parser that translates circuits and components defined in YAML into the intermediate
python representation.
"""
from pyrates.utility import deep_compare

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


@pytest.mark.parametrize("operator", ["pyrates.frontend.axon.axon.PotentialToRateOperator",
                                      "pyrates.frontend.axon.templates.SigmoidPRO",
                                      "pyrates.frontend.axon.templates.JansenRitPRO",
                                      "pyrates.frontend.population.population.CurrentToPotentialOperator",
                                      "pyrates.frontend.synapse.synapse.RateToCurrentOperator",
                                      "pyrates.frontend.coupling.coupling.CouplingOperator",
                                      "pyrates.frontend.coupling.templates.LinearCoupling",
                                      ])
def test_import_operator_templates(operator):
    """test basic (vanilla) YAML parsing using ruamel.yaml (for YAML 1.2 support)"""
    from pyrates.frontend.operator import OperatorTemplateLoader
    from pyrates.frontend.operator import OperatorTemplate

    template = OperatorTemplate.from_yaml(operator)  # type: OperatorTemplate

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

    path = "pyrates.frontend.circuit.templates.JansenRitCircuit"
    from pyrates.frontend.circuit import CircuitTemplate
    from pyrates.frontend.edge.edge import EdgeTemplate
    from pyrates.frontend.node import NodeTemplate
    from pyrates.frontend.operator import OperatorTemplate

    template = CircuitTemplate.from_yaml(path)

    # test, whether circuit is in loader cache
    from pyrates.frontend.yaml_parser import TemplateLoader
    assert template is TemplateLoader.cache[path]

    # test, whether node templates have been loaded successfully
    nodes = {"JR_PC": "pyrates.frontend.population.templates.JansenRitPC",
             "JR_IIN": "pyrates.frontend.population.templates.JansenRitIN",
             "JR_EIN": "pyrates.frontend.population.templates.JansenRitIN"}

    for key, value in nodes.items():
        assert isinstance(template.nodes[key], NodeTemplate)
        assert template.nodes[key] is TemplateLoader.cache[value]
        # test operators in node templates
        for op in template.nodes[key].operators:
            assert op.path in TemplateLoader.cache
            assert isinstance(op, OperatorTemplate)

    # test, whether coupling operator has been loaded correctly
    coupling_path = "pyrates.frontend.edge.templates.LinearCouplingOperator"
    edge_temp = template.edges[0][2]
    assert isinstance(edge_temp, EdgeTemplate)
    assert list(edge_temp.operators)[0] is TemplateLoader.cache[coupling_path]

    assert repr(template) == f"<CircuitTemplate '{path}'>"


def test_circuit_instantiation():
    """Test, if apply() functions all work properly"""
    path = "pyrates.frontend.circuit.templates.JansenRitCircuit"
    from pyrates.frontend.circuit import CircuitTemplate

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
    path = "pyrates.frontend.circuit.templates.MultiJansenRitCircuit"
    from pyrates.frontend.circuit import CircuitTemplate

    template = CircuitTemplate.from_yaml(path)

    circuit = template.apply()
    assert circuit


def test_equation_alteration():
    """Test, if properties of a template that mean to alter a certain parent equation are treated correctly"""

    path = "pyrates.frontend.population.templates.InstantaneousCPO"
    # this template removes the component "L_m * " from the base equation "L_m * V = k * I"
    from pyrates.frontend.operator import OperatorTemplate

    template = OperatorTemplate.from_yaml(path)

    operator, values = template.apply()

    assert operator.equations == ["V = k * I"]


def test_network_def_workaround():
    path = "pyrates.frontend.circuit.templates.JansenRitCircuit"
    from pyrates.frontend.circuit import CircuitTemplate

    template = CircuitTemplate.from_yaml(path)

    circuit = template.apply()

    nd = circuit.network_def()

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


@pytest.mark.skip
def test_differential_equation_reparsing():

    from pyrates.frontend.node import NodeTemplate

    template = NodeTemplate.from_yaml("pyrates.frontend.population.templates.JansenRitIN")
    node = template.apply()

    assert len(node["JansenRitExcitatorySynapseRCO.0"].equations) == 2


def test_yaml_dump():
    from pyrates.frontend.circuit import CircuitTemplate
    circuit = CircuitTemplate.from_yaml("pyrates.frontend.circuit.templates.JansenRitCircuit").apply()
    circuit.to_yaml("../output/yaml_dump.yaml", "DumpedCircuit")

    # reload saved circuit
    saved_circuit = CircuitTemplate.from_yaml("../output/yaml_dump.yaml.DumpedCircuit").apply()
    assert saved_circuit
