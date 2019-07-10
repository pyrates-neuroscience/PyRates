.. PyRates documentation master file, created by
   sphinx-quickstart on Wed Jun 26 11:14:39 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyRates's documentation!
===================================

PyRates is a Python 3 tool for building rate-based neural models and performing numerical simulations of their dynamic behavior.

Among its core features are:

* Models can be configured via YAML files or Python dictionaries
* You can choose between _numpy and _tensorflow as your backend for simulations
* Every network comes with a _networkx graph representation
* Output are easily visualized via _seaborn and _mnepython
* Functionalities for multi-dimensional parameter sweeps on single machines and clusters are provided

Example
-------
Building a model and simulating its behavior via PyRates can look as simple as in the cell below::

	from pyrates.backend import ComputeGraph
	
	net = ComputeGraph('model_templates.montbrio.simple_montbrio.Net1', 
			   backend='numpy',
			   dt=dt) 
	results = net.run(simulation_time=10.0, outputs={'r': 'Pop1.0/Op_e.0/r'})
	results.plot()

Installation
------------

PyRates can either be installed via `pip`::

	pip install pyrates

Or via the `setup.py` provided on `GitHub <https://github.com/pyrates-neuroscience/PyRates>`::

	python setup.py install

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
