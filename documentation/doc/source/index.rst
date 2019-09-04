.. PyRates documentation master file, created by
   sphinx-quickstart on Wed Jun 26 11:14:39 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the PyRates documentation!
=====================================

PyRates is a Python 3 tool for building rate-based neural models and performing numerical simulations of their dynamic behavior.

Among its core features are:

* Models can be configured via YAML files or Python dictionaries
* You can choose between _numpy and _tensorflow as your backend for simulations
* Every network comes with a _networkx graph representation
* Output are easily visualized via _seaborn and _mnepython
* Functionalities for multi-dimensional parameter sweeps on single machines and clusters are provided

Installation
------------

PyRates requires an installation of Python >=3.6.1. We recommend to install PyRates into a separate virtual environment.
If using `Anaconda`, such a virtual environment can be set up via the command line::

	conda create -n pyrates python>=3.6.1

And activated by calling::

	conda activate pyrates
  
PyRates can either be installed via `pip`::

	pip install pyrates

Or via the `setup.py` provided on `GitHub <https://github.com/pyrates-neuroscience/PyRates>`::

	python setup.py install

This installation contains merely the minimum of required Python packages to build and run models in PyRates. 
Additional packages that may be installed include:

* matplotlib (for visualization)
* seaborn (for visualization)
* mne (for visualization and post-processing)
* tensorflow2.0 (for an alternative backend with GPU support)
* paramiko and h5py (for cluster computation support)
* numba (for CPU parallelization and just-in-time compilation of simulations via the NumPy backend)

All of these packages are available at `PyPI` and can thus be installed via::

	pip install <name>

HowTo
-----

Here, we provide examples of how to use the major user interfaces and links to instructive jupyter notebooks on our github repository.

.. toctree::
   :maxdepth: 4

   minimal_example
   model_setup
   simulations
   parameter_sweeps
   model_optimization
   visualization
   post_processing

Contents
--------
.. toctree::
   :maxdepth: 4
   
   pyrates.frontend
   pyrates.backend
   pyrates.ir
   pyrates.utility

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
