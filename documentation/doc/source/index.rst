.. pyrates documentation master file, created by
   sphinx-quickstart on Wed Oct  3 12:55:59 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


PyRates Documentation
=====================

Documentation of `PyRates <https://github.com/pyrates-neuroscience/PyRates>`_, an open-source Python toolbox for dynamical systems modeling.

Basic features:
---------------

- Frontend:
   - implement models via a frontend of your choice: *YAML* or *Python*
   - create basic mathematical building blocks (i.e. differential equations and algebraic equations) and use them to define a networks of nodes connected by edges
   - create hierarchical networks by connecting networks via edges
- Backend:
   - choose from a number of different backends
   - `NumPy` backend for dynamical systems modeling on CPUs via *Python*
   - `Tensorflow` and `PyTorch` backends for parameter optimization via gradient descent and dynamical systems modeling on GPUs
   - `Julia` backend for dynamical system modeling in *Julia*, via tools such as `DifferentialEquations.jl`
   - `Fortran` backend for dynamical systems modeling via *Fortran 90* and interfacing the parameter continuation software *Auto-07p*
- Other features:
   - perform quick numerical simulations via a single function call
   - choose between different numerical solvers
   - perform parameter sweeps over multiple parameters at once
   - generate backend-specific run functions that evaluate the vector field of your dynamical system
   - Implement dynamic edge equations that include scalar dealys or delay distributions (delay distributions are automatically translated into :math:`\gamma`-kernel convolutions)
   - choose from various pre-implemented dynamical systems that can be directly used for simulations or integrated into custom models

Installation:
-------------

PyRates can be installed via the `pip` command. We recommend to use `Anaconda` to create a new python environment with Python >= 3.6 and then simply run the following line from a terminal with the environment being activated:

.. code-block:: bash

   pip install pyrates


You can install optional (non-default) packages by specifying one or more options in brackets, e.g.:

.. code-block:: bash

   pip install pyrates[tf,plot]


Available options are `backends`, `dev`, and `all`.
The latter includes all optional packages.

Alternatively, it is possible to clone this repository and run one of the following lines
from the directory in which the repository was cloned:

.. code-block:: bash

   python setup.py install

or

.. code-block:: bash

   pip install .[<options>]

Finally, a singularity container of the most recent version of this software can be found [here](https://singularity.gwdg.de/containers/3).
This container provides a stand-alone version of PyRates including all necessary Python tools to be run, independent of local operating systems.
To be able to use this container, you need to install `Singularity <https://singularity.lbl.gov/>`_ on your local machine first.
Follow these `instructions <https://singularity.lbl.gov/quickstart>`_ to install singularity and run scripts inside the PyRates container.

Reference
---------

If you use PyRates, please cite:

`Gast, R., Rose, D., Salomon, C., Möller, H. E., Weiskopf, N., & Knösche, T. R. (2019). PyRates-A Python framework for rate-based neural simulations. PloS one, 14(12), e0225900. <https://doi.org/10.1371/journal.pone.0225900>`_

Contact
-------

If you have questions, problems or suggestions regarding PyRates, please contact `Richard Gast <https://www.cbs.mpg.de/person/59190/376039>`_ or `Daniel Rose <https://www.cbs.mpg.de/person/51141/374227>`_.

Contribute
----------

PyRates is an open-source project that everyone is welcome to contribute to. Check out our `GitHub repository <https://github.com/pyrates-neuroscience/PyRates>`_
for all the source code, open issues etc. and send us a pull request, if you would like to contribute something to our software.

Examples Gallery
================

For a comprehensive overview over the PyRates basics, have a look at the jupyter notebook `Tutorial_PyRates_Basics.ipynb` that can be found in `documentation`.
More specific use examples can be found in the galleries below.

.. include::
   auto_introductions/index.rst

.. include::
   auto_implementations/index.rst

.. include::
   auto_analysis/index.rst

API
===

.. toctree::
   :maxdepth: 4

   pyrates
   model_templates
   tests

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
