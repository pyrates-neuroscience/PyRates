
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
import numpy as np


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
    assert template.variables == cached_template.variables
    assert repr(template) == repr(cached_template) == f"<OperatorTemplate '{operator.split('.')[-1]}'>"


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

    # test that all item views work correctly
    for key, value in nodes.items():
        assert template[key] is template_cache[value]
        assert type(template[key]["pro"]) is OperatorTemplate
        assert "input" in template[key]["pro"]["v"]


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
    circuit.add_edges_from_matrix(source_var='phase_op/theta', target_var='phase_op/s_in', nodes=node_names,
                                  weight=weights, template=edge, edge_attr=edge_attr, min_weight=0.0)

    # test whether edges have been added as expected
    edge_attr_tmp = deepcopy(edge_attr)
    edge_attr_tmp['weight'] = weights[1, 2]
    edge_attr_tmp['delay'] = delays[1, 2]
    assert len(circuit.edges) == int(n**2)
    assert ('p3/phase_op/theta', 'p2/phase_op/s_in', edge, edge_attr_tmp) in circuit.edges

    # perform short simulation to ensure that the network has been constructed correctly
    circuit.run(simulation_time=1.0, step_size=1e-4, outputs=['all/phase_op/theta'])


def test_circuit_instantiation():
    """Test, if apply() functions all work properly"""
    path = "model_templates.neural_mass_models.jansenrit.JRC"
    from pyrates import clear_frontend_caches
    from pyrates.frontend import template
    from pyrates.backend.computegraph import ComputeGraph, ComputeVar
    clear_frontend_caches()

    circuit = template.from_yaml(path)

    circuit.apply()
    ir = circuit.intermediate_representation

    # test whether calling apply translated the template into a proper intermediate representation
    assert type(ir.graph) is ComputeGraph
    assert circuit._vectorization_indices['pc/pro/s'] == [0]
    assert len(np.unique(list(circuit._vectorization_indices.values()))) == 2
    assert len(circuit._vectorization_labels) == 3
    assert type(ir._front_to_back['pc/pro/m']) is ComputeVar


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

    assert operator.equations[1] == "x' = h*(m_in + u)/tau - 2*x/tau - v/tau**2"


def test_python_interface():
    """Test, if yaml and Python interface for template definition deliver equivalent results.
    """
    from pyrates import OperatorTemplate, NodeTemplate, CircuitTemplate, clear
    import numpy as np

    # build qif model via python interface
    eqs = [
        "r' = (Delta/(pi*tau) + 2.0*r*v) / tau",
        "v' = (v^2 + eta + I_ext + tau*r_in - (pi*tau*r)^2) / tau"
    ]

    variables = {
        "r": "output(0.01)",
        "v": "variable(-2.0)",
        "Delta": 1.0,
        "tau": 1.0,
        "eta": -5.0,
        "I_ext": "input(0.0)",
        "r_in": "input(0.0)"
    }

    op = OperatorTemplate(name='qif_op', equations=eqs, variables=variables, path=None)
    node = NodeTemplate(name='qif_pop', operators=[op], path=None)
    qif_python = CircuitTemplate(name='qif', nodes={'p': node},
                                 edges=[('p/qif_op/r', 'p/qif_op/r_in', None, {'weight': 15.0})]
                                 )

    # load YAML-based qif model definition from template
    qif_yaml = CircuitTemplate.from_yaml('model_templates.neural_mass_models.qif.qif')

    # compare dynamics of QIF population defined above with QIF population defined via YAML interface
    T = 40.0
    dt = 1e-3
    dts = 1e-2

    in_start = int(np.round(10.0 / dt))
    in_dur = int(np.round(20.0 / dt))
    inp = np.zeros((int(np.round(T / dt)),))
    inp[in_start:in_start + in_dur] = 3.0

    r1 = qif_python.run(simulation_time=T, step_size=dt, sampling_step_size=dts, solver='scipy',
                        outputs={'r': 'p/qif_op/r'}, inputs={'p/qif_op/I_ext': inp})
    clear(qif_python)
    r2 = qif_yaml.run(simulation_time=T, step_size=dt, sampling_step_size=dts, solver='scipy',
                      outputs={'r': 'p/qif_op/r'}, inputs={'p/qif_op/I_ext': inp})
    clear(qif_yaml)

    assert np.mean(r1.values - r2.values) == pytest.approx(0.0, rel=1e-4, abs=1e-4)
