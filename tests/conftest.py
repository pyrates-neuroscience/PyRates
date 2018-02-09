""" pytest configuration file based on example from https://github.com/dropbox/pyannotate
This file configures py.test so that pyannotate is used to collect types in the tested module.
The results are saved as type_info.json.

Note that only files are checked for types that are in the current working directory. In order to check other files,
move the test files to the respective folder.
"""

# Configuration for pytest to automatically collect types.
# Thanks to Guilherme Salgado.

# import pytest
# from pyannotate_runtime import collect_types
#
# collect_types.init_types_collection()
#
#
# @pytest.fixture(autouse=True)
# def collect_types_fixture():
#     collect_types.resume()
#     yield
#     collect_types.pause()
#
#
# def pytest_sessionfinish(session, exitstatus):
#     collect_types.dump_stats("type_info.json")
