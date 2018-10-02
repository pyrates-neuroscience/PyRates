""" Tests for the parser that translates circuits and components defined in YAML into the intermediate
python representation.
"""
from pyrates.circuit import CircuitTemplate
from pyrates.node import NodeTemplate
from pyrates.operator import OperatorTemplate

__author__ = "Daniel Rose"
__status__ = "Development"

import pytest


def setup_module():
    print("\n")
    print("===========================")
    print("| Test Suite: YAML Parser |")
    print("===========================")


@pytest.mark.parametrize("section", ["axon", "population", "synapse"])
def test_parse_operator_templates(section):
    """test basic (vanilla) YAML parsing using ruamel.yaml (for YAML 1.2 support)"""
    from ruamel.yaml import YAML
    import os

    yaml = YAML(typ="safe", pure=True)

    path = f"../pyrates/{section}/"
    base_filename = f"{section}.yaml"
    template_filename = f"templates.yaml"

    # load base template
    filepath = os.path.join(path, base_filename)
    with open(filepath, "r") as file:
        base = yaml.load(file)

    # load further templates
    filepath = os.path.join(path, template_filename)
    with open(filepath, "r") as file:
        templates = yaml.load(file)


@pytest.mark.parametrize("operator", ["pyrates.axon.axon.PotentialToRateOperator",
                                      "pyrates.axon.templates.SigmoidPRO",
                                      "pyrates.axon.templates.JansenRitPRO",
                                      "pyrates.population.population.CurrentToPotentialOperator",
                                      "pyrates.synapse.synapse.RateToCurrentOperator",
                                      "pyrates.coupling.coupling.CouplingOperator",
                                      "pyrates.coupling.templates.LinearCoupling",
                                      ])
def test_import_operator_templates(operator):
    """test basic (vanilla) YAML parsing using ruamel.yaml (for YAML 1.2 support)"""
    from pyrates.operator.operator import OperatorTemplateLoader

    template = OperatorTemplate.from_yaml(operator)  # type: OperatorTemplate

    assert template.path in OperatorTemplateLoader.cache

    cached_template = OperatorTemplateLoader.cache[operator]  # type: OperatorTemplate
    assert template is cached_template
    assert template.path == cached_template.path
    assert template.equation == cached_template.equation
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

    path = "pyrates.circuit.templates.JansenRitCircuit"
    from pyrates.circuit import CircuitTemplate

    template = CircuitTemplate.from_yaml(path)

    # test, whether circuit is in loader cache
    from pyrates.utility.yaml_parser import TemplateLoader
    assert template is TemplateLoader.cache[path]

    # test, whether node templates have been loaded successfully
    nodes = {"JR_PC": "pyrates.population.templates.JansenRitPC",
             "JR_IIN": "pyrates.population.templates.JansenRitIN",
             "JR_EIN": "pyrates.population.templates.JansenRitIN"}

    for key, value in nodes.items():
        assert isinstance(template.nodes[key], NodeTemplate)
        assert template.nodes[key] is TemplateLoader.cache[value]
        # test operators in node templates
        for op in template.nodes[key].operators:
            assert op.path in TemplateLoader.cache
            assert isinstance(op, OperatorTemplate)

    # test, whether coupling operator has been loaded correctly
    coupling_path = "pyrates.coupling.templates.LinearCoupling"

    assert isinstance(template.coupling["LC"], OperatorTemplate)
    assert template.coupling["LC"] is TemplateLoader.cache[coupling_path]

    assert repr(template) == f"<CircuitTemplate '{path}'>"


def test_circuit_instantiation():
    """Test, if apply() functions all work properly"""
    path = "pyrates.circuit.templates.JansenRitCircuit"
    from pyrates.circuit import CircuitTemplate

    template = CircuitTemplate.from_yaml(path)

    circuit = template.apply()
    # test if two edges refer to the same coupling operator by comparing ids
    for op_key, op in circuit.edges[("JR_PC", "JR_IIN", 0)]["operators"].items():
        assert op is circuit.edges[('JR_PC', 'JR_EIN', 0)]["operators"][op_key]


def test_equation_alteration():
    """Test, if properties of a template that mean to alter a certain parent equation are treated correctly"""

    path = "pyrates.population.templates.InstantaneousCPO"
    # this template removes the component "L_m * " from the base equation "L_m * V = k * I"
    from pyrates.operator import OperatorTemplate

    template = OperatorTemplate.from_yaml(path)

    operator, values = template.apply()

    assert operator["equation"] == ["V = k * I"]


def test_network_def_workaround():
    path = "pyrates.circuit.templates.JansenRitCircuit"
    from pyrates.circuit import CircuitTemplate

    template = CircuitTemplate.from_yaml(path)

    circuit = template.apply()

    nd = circuit.network_def()

    operator_order = ['JansenRitExcitatorySynapseRCO:0',
                      'JansenRitInhibitorySynapseRCO:0',
                      'JansenRitCPO:0',
                      'JansenRitPRO:0']
    inputs = {}

    cpo_i = {'dtype': 'float32',
             'shape': (),
             'vtype': 'state_var'}

    jr_cpo = {'equations': ['V = k * I'],
              'inputs': {'I': {'reduce_dim': True,
                               'source': ['JansenRitExcitatorySynapseRCO:0/I',
                                          'JansenRitInhibitorySynapseRCO:0/I']}},
              'output': 'V'}
    # assert dict(nd.nodes["JR_PC:0"]) == JR_PC
    node = nd.nodes["JR_PC:0"]
    edge = {'operator_args': {'LC/c': {'dtype': 'int32', 'shape': (), 'value': 108.0, 'vtype': 'constant'},
                              'LC/m_in': {'name': 'JansenRitExcitatorySynapseRCO:0/m_in', 'vtype': 'target_var'},
                              'LC/m_out': {'name': 'JansenRitExcitatorySynapseRCO:0/m_out', 'vtype': 'source_var'}},
            'operator_order': ['LC'],
            'operators': {'LC': {'equation': ['m_in = c * m_out'],
                                 'inputs': {'m_out': {'reduce_dim': True, 'source': []}},
                                 'output': 'm_in'}}}
    assert node["operator_order"] == operator_order
    assert node["inputs"] == inputs
    assert node["operator_args"]['JansenRitCPO:0/I'] == cpo_i
    assert node["operators"]['JansenRitExcitatorySynapseRCO:0']["inputs"]["m_in"]["source"] == []
    assert node["operators"]['JansenRitCPO:0'] == jr_cpo
    assert dict(nd.edges[('JR_EIN:0', 'JR_PC:0', 0)]) == edge
