"""Test suite for CircuitTemplate.get_jacobian_func."""

import numpy as np
import pytest
from pyrates import CircuitTemplate, OperatorTemplate, clear
from pyrates.ir.node import clear_ir_caches


def setup_module():
    print("\n")
    print("================================")
    print("| Test Suite: Jacobian Function |")
    print("================================")


@pytest.fixture(autouse=True)
def reset_ir_caches():
    """Clear IR/operator caches before every test to prevent inter-test contamination."""
    from pyrates import OperatorTemplate
    from pyrates.ir.node import clear_ir_caches as _clr
    from pyrates.ir.circuit import in_edge_indices, in_edge_vars
    from pyrates.frontend.template import template_cache
    OperatorTemplate.cache.clear()
    _clr()
    in_edge_indices.clear()
    in_edge_vars.clear()
    template_cache.clear()
    yield
    OperatorTemplate.cache.clear()
    _clr()
    in_edge_indices.clear()
    in_edge_vars.clear()
    template_cache.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fd_jacobian(func, t, y, params, eps=1e-6):
    """Central-difference numerical Jacobian of func(t, y, *params)."""
    n = len(y)
    J = np.zeros((n, n))
    for j in range(n):
        yp = y.copy()
        ym = y.copy()
        yp[j] += eps
        ym[j] -= eps
        fp = np.asarray(func(t, yp, *params), dtype=float).ravel()
        fm = np.asarray(func(t, ym, *params), dtype=float).ravel()
        J[:, j] = (fp - fm) / (2 * eps)
    return J


def _vdp_circuit():
    """Return a fresh Van der Pol circuit (no extrinsic input)."""
    return CircuitTemplate.from_yaml("model_templates.oscillators.vanderpol.vdp")


# ---------------------------------------------------------------------------
# Test 1: ODE Jacobian — dimensions
# ---------------------------------------------------------------------------

def test_jacobian_ode_dimensions(backend):
    """Generated ODE Jacobian has shape (n, n)."""
    vdp = _vdp_circuit()
    jac_func, jac_args, _, sv_indices = vdp.get_jacobian_func(
        func_name='vdp_jac', step_size=1e-3, solver='euler', in_place=False, clear=True,
        vectorize=False, backend=backend,
    )
    clear_ir_caches()

    t0, y0 = float(jac_args[0]), np.array(jac_args[1], dtype=float)
    J = jac_func(t0, y0, *jac_args[2:])
    n = len(y0)
    assert J.shape == (n, n), f"Expected ({n}, {n}), got {J.shape}"


# ---------------------------------------------------------------------------
# Test 2: ODE Jacobian vs finite difference
# ---------------------------------------------------------------------------

def test_jacobian_ode_vs_finite_difference(backend):
    """Symbolic Jacobian matches finite-difference approximation for Van der Pol.

    The run_func writes into a float32 `dy` buffer, so we use a perturbation
    large enough to stay above float32 machine epsilon (~1.2e-7) while still
    being small enough for an accurate FD estimate.  eps=1e-3 gives ~1e-3
    relative error in the FD, which comfortably passes atol=0.01.
    """
    vdp_run = _vdp_circuit()
    run_func, run_args, _, _ = vdp_run.get_run_func(
        func_name='vdp_run', step_size=1e-3, solver='euler', in_place=False, clear=True,
        vectorize=False, backend=backend,
    )
    clear_ir_caches()

    vdp_jac = _vdp_circuit()
    jac_func, jac_args, _, _ = vdp_jac.get_jacobian_func(
        func_name='vdp_jac_fd', step_size=1e-3, solver='euler', in_place=False, clear=True,
        vectorize=False, backend=backend,
    )
    clear_ir_caches()

    t0 = float(run_args[0])
    y0 = np.array(run_args[1], dtype=float)
    params = run_args[2:]

    J_sym = np.asarray(jac_func(t0, y0, *jac_args[2:]), dtype=float)
    # use eps=1e-3 to avoid float32 quantization artefacts
    J_fd = fd_jacobian(run_func, t0, y0, params, eps=1e-3)

    np.testing.assert_allclose(J_sym, J_fd, atol=0.01,
                               err_msg="Symbolic Jacobian differs from finite-difference approximation")


# ---------------------------------------------------------------------------
# Test 3: ODE Jacobian exact analytical values
# ---------------------------------------------------------------------------

def test_jacobian_ode_analytical_values(backend):
    """Verify exact entries of Van der Pol Jacobian at y = [0, 1] with mu = 1.

    f = [z,  mu*(1-x^2)*z - x + inp]
    J = [[0,  1],
         [-1, 1]]   at (x=0, z=1, mu=1, inp=0)
    """
    vdp = _vdp_circuit()
    jac_func, jac_args, _, sv_indices = vdp.get_jacobian_func(
        func_name='vdp_jac_val', step_size=1e-3, solver='euler', in_place=False, clear=True,
        vectorize=False, backend=backend,
    )
    clear_ir_caches()

    t0, y0 = float(jac_args[0]), np.array(jac_args[1], dtype=float)

    # find state-variable indices
    x_idx = sv_indices.get('p/vdp_op/x')
    z_idx = sv_indices.get('p/vdp_op/z')

    # set y to x=0, z=1
    if x_idx is not None and z_idx is not None:
        y0[x_idx] = 0.0
        y0[z_idx] = 1.0

    J = np.asarray(jac_func(t0, y0, *jac_args[2:]), dtype=float)

    assert J[x_idx, z_idx] == pytest.approx(1.0, abs=1e-8), \
        f"J[x, z] = {J[x_idx, z_idx]}, expected 1.0 (dx/dz = 1)"
    assert J[z_idx, x_idx] == pytest.approx(-1.0, abs=1e-8), \
        f"J[z, x] = {J[z_idx, x_idx]}, expected -1 (-2*mu*x*z - 1 at x=0, z=1)"
    assert J[x_idx, x_idx] == pytest.approx(0.0, abs=1e-8), \
        f"J[x, x] = {J[x_idx, x_idx]}, expected 0"
    assert J[z_idx, z_idx] == pytest.approx(1.0, abs=1e-8), \
        f"J[z, z] = {J[z_idx, z_idx]}, expected mu*(1-x^2) = 1 at x=0"


# ---------------------------------------------------------------------------
# Test 3b: ODE Jacobian on a system with transcendentals (exp / sin / cos)
# ---------------------------------------------------------------------------

def test_jacobian_ode_transcendentals(backend):
    """Symbolic Jacobian differentiates through ``exp``, ``sin``, ``cos`` etc.

    Regression for the rebind in ``_node_to_expr`` that used to replace
    ``sp.exp`` with ``Function('exp')`` (an UndefinedFunction). After that
    rebind, ``sp.diff(exp(x), x)`` returned the unevaluated
    ``Derivative(exp(x), x)`` — the analytical-Jacobian path
    (``_compute_symbolic_jacobian``) then produced uncompilable Fortran
    (``Derivative(...)`` written verbatim) for any model with a
    transcendental in its RHS. Now the rebind only fires when the backend's
    call name actually differs from the sympy one, so stdlib functions keep
    their built-in derivative rules.
    """
    from pyrates.frontend.template.operator import OperatorTemplate
    from pyrates.frontend.template.node import NodeTemplate
    from pyrates.frontend.template.circuit import CircuitTemplate

    def _build_circuit(circuit_name: str, op_name: str, node_name: str):
        # Fresh templates per call — both get_jacobian_func and get_run_func
        # mutate the circuit internally (see test_jacobian_ode_vs_finite_difference
        # for the same pattern).
        op = OperatorTemplate(
            name=op_name,
            equations=[
                "u' = -u + exp(-a*u) + cos(w)",
                "w' = -w + sin(u)",
            ],
            variables={'u': 'output(0.1)', 'w': 'variable(0.2)', 'a': 1.5},
        )
        node = NodeTemplate(name=node_name, operators=[op])
        return CircuitTemplate(name=circuit_name, nodes={'p': node})

    circuit_jac = _build_circuit('trans_jac_circ', 'trans_op_jac', 'trans_pop_jac')
    jac_func, jac_args, _, _ = circuit_jac.get_jacobian_func(
        func_name='trans_jac', step_size=1e-3, solver='euler', in_place=False, clear=True,
        vectorize=False, backend=backend,
    )
    clear_ir_caches()

    circuit_run = _build_circuit('trans_run_circ', 'trans_op_run', 'trans_pop_run')
    run_func, run_args, _, _ = circuit_run.get_run_func(
        func_name='trans_run', step_size=1e-3, solver='euler', in_place=False, clear=True,
        vectorize=False, backend=backend,
    )
    clear_ir_caches()

    t0 = float(run_args[0])
    y0 = np.array(run_args[1], dtype=float)
    J_sym = np.asarray(jac_func(t0, y0, *jac_args[2:]), dtype=float)
    J_fd = fd_jacobian(run_func, t0, y0, run_args[2:], eps=1e-3)

    np.testing.assert_allclose(
        J_sym, J_fd, atol=0.01,
        err_msg="Symbolic Jacobian disagrees with finite difference on transcendental-RHS system",
    )


# ---------------------------------------------------------------------------
# Test 3b: ODE Jacobian for circuits with stateful (plastic) edges
# ---------------------------------------------------------------------------

def test_jacobian_stateful_edge(backend):
    """Symbolic Jacobian compiles when the circuit contains a stateful edge.

    Regression for the ``identity()`` derivative bug in
    ``_compute_symbolic_jacobian``.  Plastic edges route their pre-synaptic
    state through the internal ``identity()`` marker before it reaches the
    target's ``s_in`` input.  Because ``identity`` is registered as a bare
    ``sympy.Function`` with no ``fdiff`` defined, chain-rule applications
    used to leave behind ``Subs(Derivative(identity(_xi), _xi), _xi, expr)``
    wrappers — and bare ``identity(...)`` calls in Jacobian entries — that
    the Fortran printer emitted verbatim, producing uncompilable output
    (``Integer too big for its kind`` on the dummy-symbol hash and
    ``Function 'identity' has no IMPLICIT type``).  ``_resolve_derivatives``
    now collapses leftover ``Subs`` wrappers via ``.doit()`` and strips
    ``identity(arg)`` markers down to ``arg`` at the sympy level so the
    Jacobian stringifier never sees them.
    """
    from pyrates.frontend.template.operator import OperatorTemplate
    from pyrates.frontend.template.node import NodeTemplate
    from pyrates.frontend.template.edge import EdgeTemplate
    from pyrates.frontend.template.circuit import CircuitTemplate

    def _build_circuit(circuit_name: str, op_suffix: str):
        qif_op = OperatorTemplate(
            name=f'qif_op_{op_suffix}',
            equations=[
                "r' = D + 2.0*r*v",
                "v' = v*v + J*s_in + eta - (pi*r)*(pi*r)",
            ],
            variables={'r': 'output(0.1)', 'v': 'variable(-2.0)',
                       'eta': -5.0, 'J': 1.0, 'D': 0.1,
                       's_in': 'input(0.0)'},
        )
        sd_op = OperatorTemplate(
            name=f'sd_op_{op_suffix}',
            equations=["s' = r - s/tau_s"],
            variables={'s': 'output(0.0)', 'tau_s': 0.5, 'r': 'input(0.0)'},
        )
        # stateful edge: w' = -w/tau_w + a*pre_s, s_out = pre_s * w
        weight_op = OperatorTemplate(
            name=f'weight_op_{op_suffix}',
            equations=["w' = -w/tau_w + a*pre_s",
                       "s_out = pre_s*w"],
            variables={'s_out': 'output(0.0)', 'w': 'variable(0.5)',
                       'tau_w': 10.0, 'a': 0.1, 'pre_s': 'input(0.0)'},
        )
        node = NodeTemplate(name=f'qif_node_{op_suffix}',
                            operators=[qif_op, sd_op])
        edge = EdgeTemplate(name=f'plastic_edge_{op_suffix}',
                            operators=[weight_op])
        return CircuitTemplate(
            name=circuit_name,
            nodes={'p0': node, 'p1': node},
            edges=[
                (f'p0/sd_op_{op_suffix}/s', f'p1/qif_op_{op_suffix}/s_in', edge, {'weight': 1.0}),
                (f'p1/sd_op_{op_suffix}/s', f'p0/qif_op_{op_suffix}/s_in', edge, {'weight': 1.0}),
            ],
        )

    circuit_jac = _build_circuit('stateful_jac_circ', 'jac')
    jac_func, jac_args, _, _ = circuit_jac.get_jacobian_func(
        func_name='stateful_jac', step_size=1e-3, solver='euler',
        in_place=False, clear=True, vectorize=False, backend=backend,
    )
    clear_ir_caches()

    circuit_run = _build_circuit('stateful_run_circ', 'run')
    run_func, run_args, _, _ = circuit_run.get_run_func(
        func_name='stateful_run', step_size=1e-3, solver='euler',
        in_place=False, clear=True, vectorize=False, backend=backend,
    )
    clear_ir_caches()

    t0 = float(run_args[0])
    y0 = np.array(run_args[1], dtype=float)
    J_sym = np.asarray(jac_func(t0, y0, *jac_args[2:]), dtype=float)
    J_fd = fd_jacobian(run_func, t0, y0, run_args[2:], eps=1e-3)

    np.testing.assert_allclose(
        J_sym, J_fd, atol=0.01,
        err_msg="Symbolic Jacobian disagrees with finite difference on stateful-edge circuit",
    )


# ---------------------------------------------------------------------------
# Test 4: Sparse ODE Jacobian
# ---------------------------------------------------------------------------

def test_jacobian_sparse(backend):
    """sparse=True returns scipy.sparse.csr_matrix with identical numerical content."""
    from scipy.sparse import issparse

    vdp_d = _vdp_circuit()
    jac_dense, jargs_d, _, _ = vdp_d.get_jacobian_func(
        func_name='vdp_jac_dense', step_size=1e-3, solver='euler', sparse=False,
        in_place=False, clear=True, vectorize=False, backend=backend,
    )
    clear_ir_caches()

    vdp_s = _vdp_circuit()
    try:
        jac_sparse, jargs_s, _, _ = vdp_s.get_jacobian_func(
            func_name='vdp_jac_sparse', step_size=1e-3, solver='euler', sparse=True,
            in_place=False, clear=True, vectorize=False, backend=backend,
        )
    except NotImplementedError as e:
        pytest.skip(f"backend `{backend}` does not support sparse Jacobian: {e}")
    clear_ir_caches()

    t0 = float(jargs_d[0])
    y0_d = np.array(jargs_d[1], dtype=float)
    y0_s = np.array(jargs_s[1], dtype=float)

    J_dense = np.asarray(jac_dense(t0, y0_d, *jargs_d[2:]), dtype=float)
    J_sparse_result = jac_sparse(t0, y0_s, *jargs_s[2:])

    assert issparse(J_sparse_result), "sparse=True should return a sparse matrix"
    np.testing.assert_allclose(J_sparse_result.toarray(), J_dense, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 5: DDE Jacobian — structure and finite-difference for J0
# ---------------------------------------------------------------------------

def test_jacobian_dde_structure(backend):
    """DDE Jacobian returns (J0, [J_hist]) with correct shapes."""
    vdp = _vdp_circuit()
    k, tau = 0.5, 0.1
    vdp.update_template(
        edges=[('p/vdp_op/x', 'p/vdp_op/inp', None, {'weight': k, 'delay': tau})],
        in_place=True
    )

    jac_func, jac_args, _, sv_indices = vdp.get_jacobian_func(
        func_name='vdp_dde_jac', step_size=1e-3, solver='scipy', in_place=False, clear=True,
        vectorize=False, backend=backend,
    )
    clear_ir_caches()

    t0, y0 = float(jac_args[0]), np.array(jac_args[1], dtype=float)
    n = len(y0)

    result = jac_func(t0, y0, *jac_args[2:])
    assert isinstance(result, tuple) and len(result) == 2, \
        "DDE Jacobian should return (J0, [J_hist]) tuple"
    J0, J_hist_list = result

    J0 = np.asarray(J0, dtype=float)
    assert J0.shape == (n, n), f"J0 shape {J0.shape} != ({n}, {n})"

    assert isinstance(J_hist_list, list) and len(J_hist_list) >= 1, \
        "J_hist should be a non-empty list"
    J_hist = np.asarray(J_hist_list[0], dtype=float)
    assert J_hist.shape == (n, n), f"J_hist shape {J_hist.shape} != ({n}, {n})"


def test_jacobian_dde_j0_vs_finite_difference(backend):
    """J0 from DDE Jacobian matches finite-difference Jacobian of the instantaneous part."""
    k, tau = 0.5, 0.1

    vdp_run = _vdp_circuit()
    vdp_run.update_template(
        edges=[('p/vdp_op/x', 'p/vdp_op/inp', None, {'weight': k, 'delay': tau})],
        in_place=True
    )
    run_func, run_args, _, _ = vdp_run.get_run_func(
        func_name='vdp_dde_run', step_size=1e-3, solver='scipy', in_place=False, clear=True,
        vectorize=False, backend=backend,
    )
    clear_ir_caches()

    vdp_jac = _vdp_circuit()
    vdp_jac.update_template(
        edges=[('p/vdp_op/x', 'p/vdp_op/inp', None, {'weight': k, 'delay': tau})],
        in_place=True
    )
    jac_func, jac_args, _, _ = vdp_jac.get_jacobian_func(
        func_name='vdp_dde_jac2', step_size=1e-3, solver='scipy', in_place=False, clear=True,
        vectorize=False, backend=backend,
    )
    clear_ir_caches()

    t0 = float(run_args[0])
    y0 = np.array(run_args[1], dtype=float)

    # use eps=1e-3 to avoid float32 quantization artefacts (same as ODE FD test)
    J_fd = fd_jacobian(run_func, t0, y0, run_args[2:], eps=1e-3)

    result = jac_func(t0, y0, *jac_args[2:])
    J0 = np.asarray(result[0], dtype=float)

    np.testing.assert_allclose(J0, J_fd, atol=0.01,
                               err_msg="J0 differs from finite-difference approximation")
