"""Pytest module-wide configuration file."""

import pytest


@pytest.fixture(autouse=True)
def run_around_tests():
    # Code that will run before your test
    # Find and save current (test) working directory
    import os
    import warnings
    cwd = os.getcwd()
    if not cwd.endswith("tests"):
        os.chdir("tests/")
    cwd = os.getcwd()
    warnings.warn(f"before: {cwd}")
    # A test function will be run at this point
    yield
    # Code that will run after the test
    warnings.warn(f"\n after: {os.getcwd()}")
    # revert changes to working directory
    os.chdir(cwd)
