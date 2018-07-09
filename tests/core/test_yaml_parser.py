""" Tests for the parser that translates circuits and components defined in YAML into the intermediate
python representation.
"""
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
                                      "pyrates.coupling.templates.JansenRitCoupling",
                                      ])
def test_import_operator_templates(operator):
    """test basic (vanilla) YAML parsing using ruamel.yaml (for YAML 1.2 support)"""
    from pyrates.utility.yaml_parser import OperatorTemplateLoader

    template = OperatorTemplateLoader(operator)  # type: OperatorTemplate

    assert template.path in OperatorTemplateLoader.cache

    cached_template = OperatorTemplateLoader.cache[operator]
    assert template is cached_template
    assert template.path == cached_template.path
    assert template.equation == cached_template.equation
    assert repr(template) == repr(cached_template) == f"OperatorTemplate <{operator}>"
    # assert template == TemplateLoader.cache[template.path]


@pytest.mark.parametrize("node", ["pyrates.population.templates.JansenRitIN",
                                  "pyrates.population.templates.JansenRitPC",
                                  "pyrates.population.population.NeuralMass"
                                  ])
def test_import_node_templates(node):
    """test import of node templates"""

    from pyrates.utility.yaml_parser import NodeTemplateLoader, TemplateLoader

    template = NodeTemplateLoader(node)  # type: NodeTemplate

    assert template.path in TemplateLoader.cache

    cached_template = NodeTemplateLoader.cache[node]
    assert template is cached_template
    assert template.path == cached_template.path
    assert template.operators == cached_template.operators
    assert repr(template) == repr(cached_template) == f"NodeTemplate <{node}>"


