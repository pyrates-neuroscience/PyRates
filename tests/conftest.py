"""Pytest module-wide configuration file."""

import pytest


@pytest.fixture(autouse=True)
def run_around_tests():
    # Code that will run before your test
    # Find and save current (test) working directory
    import os
    cwd = os.getcwd()
    print(f"before: {cwd}")
    # A test function will be run at this point
    yield
    # Code that will run after the test
    print(f"after: {os.getcwd()}")
    # revert changes to working directory
    os.chdir(cwd)
