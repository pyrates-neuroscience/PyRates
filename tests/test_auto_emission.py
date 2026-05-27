"""Test suite for the auto-07p Fortran emission path.

Covers the YAML-order preservation (issue: param-order scrambling resulted in
`parnames = {1: 'p4', 2: 'p2', 3: 'p1', 4: 'p3'}` instead of the declared
`{1: 'p1', 2: 'p2', 3: 'p3', 4: 'p4'}`) and the sub-routine signature / auto
wrapper consistency.
"""

import os
import re
import tempfile

import pytest

from pyrates import CircuitTemplate, OperatorTemplate
from pyrates.frontend.template.node import NodeTemplate


def setup_module():
    print("\n")
    print("==============================")
    print("| Test Suite: auto Emission |")
    print("==============================")


@pytest.fixture(autouse=True)
def reset_ir_caches():
    """Clear IR / operator caches between tests."""
    from pyrates import OperatorTemplate
    from pyrates.ir.node import clear_ir_caches
    from pyrates.ir.circuit import in_edge_indices, in_edge_vars
    from pyrates.frontend.template import template_cache
    OperatorTemplate.cache.clear()
    clear_ir_caches()
    in_edge_indices.clear()
    in_edge_vars.clear()
    template_cache.clear()
    yield
    OperatorTemplate.cache.clear()
    clear_ir_caches()
    in_edge_indices.clear()
    in_edge_vars.clear()
    template_cache.clear()


def _build_ops_circuit(name_suffix: str = ''):
    """Programmatic build of the auto-07p `ops` demo (FitzHugh-Nagumo style),
    with parameters declared in YAML-style order ``p1, p2, p3, p4``."""
    op = OperatorTemplate(
        name=f'ops_op{name_suffix}',
        equations=[
            "x1' = (-p4*(x1^3/3 - x1) + (x3 - x1)/p2 - x2) / p1",
            "x2' = x1 - p3",
            "x3' = -(x3 - x1)/p2",
        ],
        variables={
            'x1': 'output(0.9)',
            'x2': 'variable(1.314)',
            'x3': 'variable(0.9)',
            'p1': 0.5,
            'p2': 4.0,
            'p3': 0.9,
            'p4': 2.0,
        },
        path=None,
    )
    node = NodeTemplate(name=f'ops_pop{name_suffix}', operators=[op], path=None)
    return CircuitTemplate(name=f'ops{name_suffix}', nodes={'p': node})


def test_auto_parnames_preserves_yaml_order(tmp_path):
    """`parnames` in the generated c.* file follows the YAML `variables:` dict
    order, not the equation-walk order.

    Regression for the param-order scrambling fix: the equation-walk order is
    `p4, p2, p1, p3` (first equation references p4 first, then p2 then p1;
    second equation introduces p3). Before the fix, that ordering propagated
    into `parnames` and the auto-07p Fortran subroutine signature, so integer-
    keyed UZSTOP/ICP arguments quietly addressed the wrong PAR slot.
    """
    work = tmp_path / 'work'
    work.mkdir()
    os.chdir(work)

    circuit = _build_ops_circuit()
    circuit.get_run_func(
        'vfx', step_size=1e-3, file_name='ops_par_order',
        backend='fortran', float_precision='float64',
        auto=True, vectorize=False, solver='scipy',
    )

    c_ivp = (work / 'c.ivp').read_text()
    m = re.search(r"parnames\s*=\s*\{([^}]+)\}", c_ivp)
    assert m, f"no parnames line in c.ivp:\n{c_ivp}"
    # Parse the dict, e.g. "{1: 'p1', 2: 'p2', ...}" → ordered list of names.
    pairs = re.findall(r"(\d+)\s*:\s*'([^']+)'", m.group(1))
    parnames = {int(i): name for i, name in pairs}
    assert parnames == {1: 'p1', 2: 'p2', 3: 'p3', 4: 'p4'}, (
        f"parnames out of YAML order: {parnames}"
    )


def test_auto_subroutine_signature_matches_parnames(tmp_path):
    """The generated subroutine declares parameters in the same order that the
    auto-07p wrapper passes them in (i.e. PAR-slot order from `parnames`).

    If these get out of sync, the subroutine's named ``intent(in)``
    declarations end up bound to the wrong PAR values at runtime — silent
    miscomputation, not a crash. Pinned here against both the subroutine
    declaration line and the auto-07p ``call vfx(..., args(1), args(2), ...)``
    wrapper line.
    """
    work = tmp_path / 'work'
    work.mkdir()
    os.chdir(work)

    circuit = _build_ops_circuit('_sig')
    circuit.get_run_func(
        'vfx', step_size=1e-3, file_name='ops_sig',
        backend='fortran', float_precision='float64',
        auto=True, vectorize=False, solver='scipy',
    )

    src = (work / 'ops_sig.f90').read_text()
    # Subroutine declaration line: should be `subroutine vfx(t,y,dy,p1,p2,p3,p4)`
    sig = re.search(r'subroutine vfx\(([^)]+)\)', src)
    assert sig is not None, f"no `subroutine vfx(...)` line in generated source:\n{src[:1000]}"
    args = [a.strip() for a in sig.group(1).split(',')]
    assert args == ['t', 'y', 'dy', 'p1', 'p2', 'p3', 'p4'], (
        f"subroutine signature in equation-walk order rather than declaration order: {args}"
    )

    # auto wrapper call: `call vfx(args(14), y, dy, args(1), args(2), args(3), args(4))`.
    # Collapse line-continuations (Fortran's `&` mid-line break) before matching.
    src_unwrapped = re.sub(r'&\s*\n\s*&?', '', src)
    # Manual paren-balanced extraction (a simple regex stops at the first `)`,
    # which is inside the nested `args(N)`).
    start = src_unwrapped.find('call vfx(')
    assert start >= 0, f"no `call vfx(...)` line in generated source:\n{src[:2000]}"
    depth = 0
    inner_start = start + len('call vfx(')
    i = inner_start
    while i < len(src_unwrapped):
        ch = src_unwrapped[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            if depth == 0:
                break
            depth -= 1
        i += 1
    call_args = [a.strip() for a in src_unwrapped[inner_start:i].split(',')]
    # PAR(14) is auto's time slot; args(1..4) are p1..p4 (per YAML order).
    assert call_args == [
        'args(14)', 'y', 'dy', 'args(1)', 'args(2)', 'args(3)', 'args(4)',
    ], f"auto wrapper call doesn't index params in YAML order: {call_args}"
