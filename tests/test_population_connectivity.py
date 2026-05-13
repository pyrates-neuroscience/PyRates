# -*- coding: utf-8 -*-
"""Tests for PopulationTemplate and Connectivity.

Each test verifies that the new vector/matrix-native API produces numerically
identical results to the existing scalar node + edge API.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _qif_scalar_circuit(weight: float = 15.0):
    """Build a single-unit QIF circuit using the legacy scalar API."""
    from pyrates.frontend.template import from_yaml
    return from_yaml("model_templates.neural_mass_models.qif.qif")


def _run(circuit, T=10.0, dt=1e-3, solver='euler'):
    return circuit.run(
        simulation_time=T,
        step_size=dt,
        solver=solver,
        outputs={'r': 'p/qif_op/r'},
        clear=True,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# test 1: PopulationTemplate with n=1 matches scalar QIF
# ---------------------------------------------------------------------------

def test_population_n1_matches_scalar():
    """n=1 PopulationTemplate + scalar Connectivity should match the legacy circuit."""
    from pyrates.frontend.template import from_yaml
    from pyrates.frontend.template.population import PopulationTemplate, Connectivity
    from pyrates.frontend.template.circuit import CircuitTemplate
    from pyrates.ir.node import clear_ir_caches

    clear_ir_caches()

    # --- reference: legacy scalar circuit ---
    ref_circuit = _qif_scalar_circuit(weight=15.0)
    ref_result = _run(ref_circuit)

    clear_ir_caches()

    # --- new API ---
    qif_node = from_yaml("model_templates.neural_mass_models.qif.qif_pop")

    pop = PopulationTemplate(name='p', node=qif_node, n=1)

    W = np.array([[15.0]])  # (1, 1) weight matrix

    conn = Connectivity(
        source='p/qif_op/r',
        target='p/qif_op/r_in',
        weights=W,
    )

    circuit = CircuitTemplate(
        name='qif_pop_test',
        populations={'p': pop},
        connections=[conn],
    )

    result = circuit.run(
        simulation_time=10.0,
        step_size=1e-3,
        solver='euler',
        outputs={'r': 'p/qif_op/r'},
        clear=True,
        verbose=False,
    )

    np.testing.assert_allclose(
        result['r'].values,
        ref_result['r'].values,
        rtol=1e-5,
        err_msg="n=1 PopulationTemplate result differs from scalar QIF baseline",
    )


# ---------------------------------------------------------------------------
# test 2: PopulationTemplate with n=4, identity weight matrix
# ---------------------------------------------------------------------------

def test_population_n4_identity_weight():
    """4-unit population with identity weight matrix: each unit is independent,
    so each unit should match the scalar QIF with weight 1.0."""
    from pyrates.frontend.template import from_yaml, CircuitTemplate
    from pyrates.frontend.template.population import PopulationTemplate, Connectivity
    from pyrates.ir.node import clear_ir_caches

    clear_ir_caches()

    # reference: single unit with weight 1.0
    qif_node = from_yaml("model_templates.neural_mass_models.qif.qif_pop")

    ref_pop = PopulationTemplate(name='p', node=qif_node, n=1)
    W_ref = np.array([[1.0]])
    ref_conn = Connectivity(source='p/qif_op/r', target='p/qif_op/r_in', weights=W_ref)
    ref_circuit = CircuitTemplate(name='ref', populations={'p': ref_pop}, connections=[ref_conn])
    ref_result = ref_circuit.run(
        simulation_time=5.0, step_size=1e-3, solver='euler',
        outputs={'r': 'p/qif_op/r'}, clear=True, verbose=False,
    )

    clear_ir_caches()

    # 4-unit population with identity coupling (each unit self-coupled with weight 1)
    pop4 = PopulationTemplate(name='p', node=qif_node, n=4)
    W4 = np.eye(4)
    conn4 = Connectivity(source='p/qif_op/r', target='p/qif_op/r_in', weights=W4)
    circuit4 = CircuitTemplate(name='pop4', populations={'p': pop4}, connections=[conn4])
    result4 = circuit4.run(
        simulation_time=5.0, step_size=1e-3, solver='euler',
        outputs={'r': 'p/qif_op/r'}, clear=True, verbose=False,
    )

    # All 4 units should evolve identically (same initial conditions, same coupling).
    # result4['r'] is a DataFrame with multi-index columns ('r', 0), ('r', 1), ...
    r4 = result4['r'].values  # shape (n_time, 4)
    r_ref = ref_result['r'].values.squeeze()  # shape (n_time,)

    # each column of r4 should match the reference trajectory
    for col in range(4):
        np.testing.assert_allclose(
            r4[:, col],
            r_ref,
            rtol=1e-5,
            err_msg=f"Unit {col} of 4-unit population differs from scalar reference",
        )


# ---------------------------------------------------------------------------
# test 3: heterogeneous eta in PopulationTemplate
# ---------------------------------------------------------------------------

def test_population_heterogeneous_params():
    """Verify that per-unit parameter arrays are stored and applied correctly."""
    from pyrates.frontend.template import from_yaml, CircuitTemplate
    from pyrates.frontend.template.population import PopulationTemplate, Connectivity
    from pyrates.ir.node import clear_ir_caches

    clear_ir_caches()

    qif_node = from_yaml("model_templates.neural_mass_models.qif.qif_pop")

    eta_vals = np.array([-10.0, -5.0, 0.0, 5.0])
    pop = PopulationTemplate(
        name='p',
        node=qif_node,
        n=4,
        params={'qif_op/eta': eta_vals},
    )
    # zero coupling so each unit evolves independently
    W = np.zeros((4, 4))
    conn = Connectivity(source='p/qif_op/r', target='p/qif_op/r_in', weights=W)

    circuit = CircuitTemplate(name='het', populations={'p': pop}, connections=[conn])
    result = circuit.run(
        simulation_time=5.0, step_size=1e-3, solver='euler',
        outputs={'r': 'p/qif_op/r'}, clear=True, verbose=False,
    )

    # result['r'] is a multi-index DataFrame: ('r', 0), ('r', 1), ('r', 2), ('r', 3)
    r = result['r'].values  # shape (n_time, 4)
    # With strongly negative eta (unit 0), firing rate should stay near 0.
    # With positive eta (unit 3), firing rate should be higher.
    assert r[-1, 0] < r[-1, 3], (
        "Expected unit with eta=-10 to have lower firing rate than unit with eta=5"
    )


if __name__ == '__main__':
    test_population_n1_matches_scalar()
    print("test_population_n1_matches_scalar PASSED")
    test_population_n4_identity_weight()
    print("test_population_n4_identity_weight PASSED")
    test_population_heterogeneous_params()
    print("test_population_heterogeneous_params PASSED")
