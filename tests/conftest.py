"""Pytest module-wide configuration file.

Adds a ``--backends`` CLI option that lets users select which PyRates
backend(s) the per-backend test suites should run against.  Tests that take a
``backend`` fixture argument are automatically parametrized over the selected
list.

Examples:
    pytest tests/                                # default backend only
    pytest tests/ --backends jax                 # JAX only
    pytest tests/ --backends default,torch,jax   # three backends per test
"""

import os
import pytest


VALID_BACKENDS = {'default', 'numpy', 'torch', 'jax', 'fortran', 'julia', 'matlab'}


def pytest_addoption(parser):
    """Register the ``--backends`` CLI option."""
    parser.addoption(
        "--backends",
        action="store",
        default="default",
        help="Comma-separated list of backends to run per-backend tests against. "
             f"Valid choices: {sorted(VALID_BACKENDS)}. "
             "Example: --backends default,jax",
    )


def pytest_configure(config):
    """Validate the ``--backends`` selection once at session start."""
    raw = config.getoption("--backends")
    backends = [b.strip() for b in raw.split(",") if b.strip()]
    bad = [b for b in backends if b not in VALID_BACKENDS]
    if bad:
        raise pytest.UsageError(
            f"--backends contains unknown backend(s): {bad}. "
            f"Valid choices: {sorted(VALID_BACKENDS)}."
        )
    config._pyrates_backends = backends


def pytest_generate_tests(metafunc):
    """Auto-parametrize any test that takes a ``backend`` fixture argument.

    A test of the form ``def test_xxx(backend):`` is expanded into
    one test instance per backend selected via ``--backends``.  Failures are
    reported per-backend (e.g. ``test_xxx[jax]``).
    """
    if "backend" in metafunc.fixturenames:
        backends = getattr(metafunc.config, "_pyrates_backends", ["default"])
        metafunc.parametrize("backend", backends)


@pytest.fixture(autouse=True)
def run_around_tests():
    """Preserve the user's working directory across tests."""
    cwd = os.getcwd()
    yield
    os.chdir(cwd)
