PyRates.tests
=============

Documentation of all test functions that are included in PyRates.
YAML-based model definitions are loaded from `model_templates.test_resources`.

Notes on Testing
----------------

**Static type checking**

MyPy is used to statically check types. To test, if everything works out, run:

`MYPYPATH=./stubs/ mypy --strict-optional --ignore-missing-imports core`

If you get no output, all type checks are successful. Some issues are ignored using the comment tag

`# type: ignore`

These issues may be too complicated for mypy to recognise them properly - or too complicated to fix immediately,
but might need fixing, nevertheless.


**Running tests with py.test**

We use `py.test` for testing.

Make sure you have `pytest` installed.

Usage:

- from within PyCharm
    - select py.test as default testing framework
    - right-click on tests and select "run py.test in tests"
- from the console
    - navigate to the PyRates base directory
    - run `pytest tests`

Backend Parser Tests
--------------------

.. automodule:: tests.test_backend_parser
    :members:
    :undoc-members:
    :show-inheritance:

Backend Simulations Tests
-------------------------

.. automodule:: tests.test_backend_simulations
    :members:
    :undoc-members:
    :show-inheritance:

Intermediate Representation Tests
---------------------------------

.. automodule:: tests.test_IR
    :members:
    :undoc-members:
    :show-inheritance:

Frontend YAML Parser Tests
--------------------------

.. automodule:: tests.test_frontend_yaml_parser
    :members:
    :undoc-members:
    :show-inheritance:

Frontend Graph Parser Tests
---------------------------

.. automodule:: tests.test_frontend_graph_parser
    :members:
    :undoc-members:
    :show-inheritance:

File I/O Tests
--------------

.. automodule:: tests.test_file_io
    :members:
    :undoc-members:
    :show-inheritance:

Frontend-to-Backend Tests
-------------------------

.. automodule:: tests.test_front_to_back
    :members:
    :undoc-members:
    :show-inheritance:

Pre-defined Models Tests
------------------------

.. automodule:: tests.test_implemented_models
    :members:
    :undoc-members:
    :show-inheritance:

Cluster Compute Tests
---------------------

.. automodule:: tests.test_cluster_compute
    :members:
    :undoc-members:
    :show-inheritance:

Operator Caching Tests
----------------------

.. automodule:: tests.test_implemented_models
    :members:
    :undoc-members:
    :show-inheritance:
