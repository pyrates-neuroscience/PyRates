.. pyrates documentation master file, created by
   sphinx-quickstart on Wed Oct  3 12:55:59 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


PyRates Documentation
=====================

Documentation of `PyRates <https://github.com/pyrates-neuroscience/PyRates>`_, an open-source Python toolbox for neurodynamical systems modeling.

Basic features:
---------------

- Different backends: `Numpy` for fast simulations of small- to medium-sized networks. `Tensorflow` for efficient parallelization on GPUs/CPUs, `Fortran` for parameter continuations.
- Each model is internally represented by a `networkx` graph of nodes and edges, with the former representing the model units (i.e. single cells, cell populations, ...) and the latter the information transfer between them. In principle, this allows to implement any kind of dynamic neural system that can be expressed as a graph via PyRates.
- Solutions of initial value problems via different numerical solvers (e.g. full interface to `scipy.integrate.solve_ivp`)
- Parameter continuations and bifurcation analysis via PyAuto, an interface to `auto-07p`
- Storage of solutions in `pandas.DataFrame`
- Efficient parameter sweeps on single and multiple machines via the `grid_search` module
- Model optimization via genetic algorithms
- Visualization of results via `seaborn`
- Post-processing of simulation results via `scipy` and `MNE Python`
- The user has full control over the mathematical equations that nodes and edges are defined by.
- Model configuration and simulation can be done within a few lines of code.
- Various templates for rate-based population models are provided that can be used for neural network simulations imediatly.

Installation:
-------------

PyRates can be installed via the `pip` command. We recommend to use `Anaconda` to create a new python environment with Python >= 3.6 and then simply run the following line from a terminal with the environment being activated:

.. code-block:: bash

   pip install pyrates


You can install optional (non-default) packages by specifying one or more options in brackets, e.g.:

.. code-block:: bash

   pip install pyrates[tf,plot]


Available options are `tf`, `plot`, `proc`, `cluster`, `numba` and `all`.
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

.. include::
   auto_mod/index.rst

.. include::
   auto_def/index.rst

.. include::
   auto_sweep/index.rst

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
