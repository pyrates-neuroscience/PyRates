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

    cached_template = OperatorTemplateLoader.cache[operator] # type: OperatorTemplate
    assert template is cached_template
    assert template.path == cached_template.path
    assert template.equation == cached_template.equation
    assert repr(template) == repr(cached_template) == f"<OperatorTemplate '{operator}'>"
    # assert template == TemplateLoader.cache[template.path]


@pytest.mark.parametrize("node", ["pyrates.population.templates.JansenRitIN",
                                  "pyrates.population.templates.JansenRitPC",
                                  "pyrates.population.population.NeuralMass"
                                  ])
def test_import_node_templates(node):
    """test import of node templates"""

    from pyrates.utility.yaml_parser import TemplateLoader
    from pyrates.node.node import NodeTemplateLoader

    template = NodeTemplate.from_yaml(node)  # type: NodeTemplate

    assert template.path in TemplateLoader.cache  # just to check if cache is really shared among subclasses

    cached_template = NodeTemplateLoader.cache[node]
    assert template is cached_template
    assert template.path == cached_template.path
    assert template.operators == cached_template.operators
    assert repr(template) == repr(cached_template) == f"<NodeTemplate '{node}'>"


@pytest.mark.parametrize("circuit", ["pyrates.circuit.templates.JansenRitCircuit",
                                     "pyrates.circuit.circuit.BaseCircuit"
                                     ])
def test_import_circuit_templates(circuit):
    """test import of node templates"""

    from pyrates.circuit.circuit import CircuitTemplateLoader

    template = CircuitTemplate.from_yaml(circuit)  # type: CircuitTemplate

    assert template.path in CircuitTemplateLoader.cache

    cached_template = CircuitTemplateLoader.cache[circuit]  # type: CircuitTemplate

    assert template is cached_template
    assert template.path == cached_template.path
    assert template.nodes == cached_template.nodes
    assert template.edges == cached_template.edges

    for value in template.coupling.values():
        assert isinstance(value, OperatorTemplate)
    for value in template.nodes.values():
        assert isinstance(value, NodeTemplate)
    assert repr(template) == repr(cached_template) == f"<CircuitTemplate '{circuit}'>"
