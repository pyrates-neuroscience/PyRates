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


# ---------------------------------------------------------------------------
# test 4: Connectivity with an identity EdgeTemplate matches plain matvec
# ---------------------------------------------------------------------------

def test_connectivity_edge_identity():
    """EdgeTemplate with identity coupling ('s = theta_s') must produce identical
    results to a plain Connectivity with the same weight matrix."""
    from pyrates.frontend.template.node import NodeTemplate
    from pyrates.frontend.template.edge import EdgeTemplate
    from pyrates.frontend.template.circuit import CircuitTemplate
    from pyrates.frontend.template.population import PopulationTemplate, Connectivity
    from pyrates.ir.node import clear_ir_caches

    clear_ir_caches()

    phase_pop = NodeTemplate.from_yaml('model_templates.oscillators.kuramoto.phase_pop')
    N = 3
    W = np.array([[0., 0.5, 0.5],
                  [0.5, 0., 0.5],
                  [0.5, 0.5, 0.]])
    theta_0 = [0.0, np.pi / 4, np.pi / 2]
    dt = 1e-3
    T = 5 * dt

    def _run_kuramoto(edge, edge_var_map):
        clear_ir_caches()
        pop = PopulationTemplate(name='e', node=phase_pop, n=N,
                                 params={'phase_op/theta': theta_0})
        conn = Connectivity(source='e/phase_op/theta', target='e/phase_op/s_in',
                            weights=W, edge=edge, edge_var_map=edge_var_map)
        circuit = CircuitTemplate(name='ktest', populations={'e': pop}, connections=[conn])
        return circuit.run(simulation_time=T, step_size=dt, solver='euler',
                           outputs={'theta': 'e/phase_op/theta'}, clear=True, verbose=False)

    # Reference: plain matvec (no EdgeTemplate)
    ref = _run_kuramoto(edge=None, edge_var_map=None)

    # Identity EdgeTemplate: coupling function just passes through the source variable
    from pyrates.frontend.template.operator import OperatorTemplate
    id_op = OperatorTemplate(
        name='id_op',
        equations=['s = theta_s'],
        variables={'theta_s': 'input', 's': 'output'},
    )
    identity_edge = EdgeTemplate(name='id_edge', operators=[id_op])
    result = _run_kuramoto(edge=identity_edge, edge_var_map={'theta_s': 'source'})

    max_diff = np.max(np.abs(result['theta'].values - ref['theta'].values))
    assert max_diff < 1e-6, f"Max diff identity vs plain: {max_diff:.2e}"


# ---------------------------------------------------------------------------
# test 5: Kuramoto sine coupling with N=2 — numerical correctness
# ---------------------------------------------------------------------------

def test_connectivity_edge_kuramoto_sine():
    """Matrix Connectivity with sin(theta_s - theta_t) coupling matches hand-computed
    Euler step for N=2 Kuramoto oscillators."""
    from pyrates.frontend.template.node import NodeTemplate
    from pyrates.frontend.template.edge import EdgeTemplate
    from pyrates.frontend.template.circuit import CircuitTemplate
    from pyrates.frontend.template.population import PopulationTemplate, Connectivity
    from pyrates.ir.node import clear_ir_caches

    clear_ir_caches()

    phase_pop = NodeTemplate.from_yaml('model_templates.oscillators.kuramoto.phase_pop')
    sin_edge = EdgeTemplate.from_yaml('model_templates.oscillators.kuramoto.sin_edge')

    N = 2
    W = np.array([[0., 1.], [1., 0.]])
    theta_0 = [0.0, np.pi / 2]
    dt = 1e-3

    pop = PopulationTemplate(name='e', node=phase_pop, n=N,
                             params={'phase_op/theta': theta_0})
    conn = Connectivity(
        source='e/phase_op/theta',
        target='e/phase_op/s_in',
        weights=W,
        edge=sin_edge,
        edge_var_map={'theta_s': 'source', 'theta_t': 'e/phase_op/theta'},
    )
    circuit = CircuitTemplate(name='kuramoto_2', populations={'e': pop}, connections=[conn])

    # Run 2 steps; row at index 1 (t=dt) is the state after the first Euler step.
    result = circuit.run(simulation_time=2 * dt, step_size=dt, solver='euler',
                         outputs={'theta': 'e/phase_op/theta'}, clear=True, verbose=False)

    theta_step1 = result['theta'].values[1]  # state after 1st Euler step

    # Hand computation (omega=10, K=1 from phase_pop template defaults):
    # s_in[0] = W[0,1]*sin(theta_0[1]-theta_0[0]) = 1*sin(pi/2) =  1.0
    # s_in[1] = W[1,0]*sin(theta_0[0]-theta_0[1]) = 1*sin(-pi/2) = -1.0
    # dtheta[0] = omega + K*s_in[0] = 10+1 = 11  →  theta[0] = 0  + dt*11 = 0.011
    # dtheta[1] = omega + K*s_in[1] = 10-1 =  9  →  theta[1] = pi/2 + dt*9
    expected = np.array([theta_0[0] + dt * 11.0, theta_0[1] + dt * 9.0])

    np.testing.assert_allclose(theta_step1, expected, rtol=1e-5,
                               err_msg="Kuramoto sine coupling: numerical result does not match hand-computed Euler step")


# ---------------------------------------------------------------------------
# test 6: dynamic (DE-bearing) EdgeTemplate matches a pure-Python reference loop
# ---------------------------------------------------------------------------

def test_connectivity_dynamic_edge():
    """Lowpass-filter edge u' = (-u + r_pre)/tau_u, s = u, with N=2 and off-diagonal
    weight matrix.  The PyRates Euler trajectory must match a hand-rolled Euler loop."""
    from pyrates.frontend.template.node import NodeTemplate
    from pyrates.frontend.template.edge import EdgeTemplate
    from pyrates.frontend.template.operator import OperatorTemplate
    from pyrates.frontend.template.circuit import CircuitTemplate
    from pyrates.frontend.template.population import PopulationTemplate, Connectivity
    from pyrates.ir.node import clear_ir_caches

    clear_ir_caches()

    # Node: r' = (-r + eta + s_in) / tau_r
    node_op = OperatorTemplate(
        name='rate_op',
        equations=["r' = (-r + eta + s_in) / tau_r"],
        variables={'r': 'output', 'eta': 10.0, 'tau_r': 1.0, 's_in': 'input'},
    )
    rate_node = NodeTemplate(name='rate_node', operators=[node_op])

    # Edge: lowpass filter  u' = (-u + r_pre) / tau_u,  s = u
    edge_op = OperatorTemplate(
        name='lp_op',
        equations=["u' = (-u + r_pre) / tau_u", 's = u'],
        variables={'u': 0.0, 'r_pre': 'input', 'tau_u': 0.1, 's': 'output'},
    )
    lp_edge = EdgeTemplate(name='lp_edge', operators=[edge_op])

    N = 2
    W = np.array([[0., 1.], [1., 0.]])
    eta = np.array([5.0, 15.0])
    tau_r, tau_u = 1.0, 0.1
    dt = 1e-3
    n_steps = 5

    # --- PyRates simulation ---
    pop = PopulationTemplate(name='p', node=rate_node, n=N,
                             params={'rate_op/eta': list(eta)})
    conn = Connectivity(
        source='p/rate_op/r', target='p/rate_op/s_in',
        weights=W, edge=lp_edge, edge_var_map={'r_pre': 'source'},
    )
    circuit = CircuitTemplate(name='dyn_edge_test', populations={'p': pop}, connections=[conn])
    result = circuit.run(simulation_time=n_steps * dt, step_size=dt, solver='euler',
                         outputs={'r': 'p/rate_op/r'}, clear=True, verbose=False)

    r_pyrates = result['r'].values  # shape (n_steps, 2), rows at t=0, dt, 2*dt, ...

    # --- Reference Euler loop (pure Python/NumPy) ---
    r_ref = np.zeros((n_steps, N))
    r = np.zeros(N)
    u = np.zeros((N, N))   # u[target, source]
    for step in range(n_steps):
        r_ref[step] = r
        s_in = np.einsum('ij,ij->i', W, u)
        dr = (-r + eta + s_in) / tau_r
        du = (-u + r[np.newaxis, :]) / tau_u   # broadcast_pre(r) equivalent
        r = r + dt * dr
        u = u + dt * du

    np.testing.assert_allclose(
        r_pyrates, r_ref, rtol=1e-5,
        err_msg="Dynamic edge: PyRates Euler trajectory differs from reference loop",
    )


if __name__ == '__main__':
    test_population_n1_matches_scalar()
    print("test_population_n1_matches_scalar PASSED")
    test_population_n4_identity_weight()
    print("test_population_n4_identity_weight PASSED")
    test_population_heterogeneous_params()
    print("test_population_heterogeneous_params PASSED")
    test_connectivity_edge_identity()
    print("test_connectivity_edge_identity PASSED")
    test_connectivity_edge_kuramoto_sine()
    print("test_connectivity_edge_kuramoto_sine PASSED")
    test_connectivity_dynamic_edge()
    print("test_connectivity_dynamic_edge PASSED")
