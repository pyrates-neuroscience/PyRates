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


@pytest.mark.parametrize("section", ["axon", "population", "synapse"])
def test_parse_basic_templates(section):
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


@pytest.mark.parametrize("operator", ["pyrates.operator.operator.OperatorTemplate",
                                      "pyrates.axon.axon.PotentialToRateOperator",
                                      "pyrates.axon.templates.SigmoidPRO",
                                      "pyrates.axon.templates.JansenRitPRO",
                                      "pyrates.population.population.CurrentToPotentialOperator",
                                      "pyrates.synapse.synapse.RateToCurrentOperator"
                                      ])
def test_import_templates(operator):
    """test basic (vanilla) YAML parsing using ruamel.yaml (for YAML 1.2 support)"""
    from pyrates.utility.yaml_parser import import_template

    import_template(operator)





    assert True
