PyRates
=======

[![License](https://img.shields.io/github/license/pyrates-neuroscience/PyRates.svg)](https://github.com/pyrates-neuroscience/PyRates) 
[![CircleCI](https://circleci.com/gh/pyrates-neuroscience/PyRates/tree/master.svg?style=svg)](https://circleci.com/gh/pyrates-neuroscience/PyRates/tree/master)
[![PyPI version](https://badge.fury.io/py/pyrates.svg)](https://badge.fury.io/py/pyrates)
[![Documentation Status](https://readthedocs.org/projects/pyrates/badge/?version=latest)](https://pyrates.readthedocs.io/en/latest/?badge=latest)
[![Python](https://img.shields.io/pypi/pyversions/pyrates.svg?style=plastic)](https://badge.fury.io/py/pyrates)

<img src="https://github.com/pyrates-neuroscience/PyRates/blob/master/PyRates_logo_color.png" width="20%" heigth="20%" align="right">

PyRates is a framework for dynamical systems modeling, developed by Richard Gast and Daniel Rose. 
It is an open-source project that everyone is welcome to contribute to.

Basic features
===============

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
   - `Matlab` backend for differential equation solving via Matlab
- Other features:
   - perform quick numerical simulations via a single function call
   - choose between different numerical solvers
   - perform parameter sweeps over multiple parameters at once
   - generate backend-specific run functions that evaluate the vector field of your dynamical system
   - Implement dynamic edge equations that include scalar delays or delay distributions (delay distributions are automatically translated into gamma-kernel convolutions)
   - choose from various pre-implemented dynamical systems that can be directly used for simulations or integrated into custom models

Installation
============

Stable release (PyPI)
---------------------

PyRates can be installed via the `pip` command. We recommend to use `Anaconda` to create a new python environment with Python >= 3.6 and then simply run the following line from a terminal with the environment being activated:
```
pip install pyrates
```

You can install optional (non-default) packages by specifying one or more options in brackets, e.g.:
```
pip install pyrates[backends]
```

Available options are `backends`, `dev`, and `all` at the moment. 
The latter includes all optional packages. 
Furthermore, the option `tests` includes all packages necessary to run tests found in the github repository.

Development version (github)
----------------------------

Alternatively, it is possible to clone this repository and run one of the following lines 
from the directory in which the repository was cloned:
```
python setup.py install
```
or
```
pip install '.[<options>]'
```

Documentation
=============

For a full API of PyRates, see https://pyrates.readthedocs.io/en/latest/.
For examplary simulations and model configurations, please have a look at the jupyter notebooks provided in the documenation folder.

Reference
=========

If you use this framework, please cite:
[Gast, R., Rose, D., Salomon, C., Möller, H. E., Weiskopf, N., & Knösche, T. R. (2019). PyRates-A Python framework for rate-based neural simulations. PloS one, 14(12), e0225900.](https://doi.org/10.1371/journal.pone.0225900)

Contact
=======

If you have questions, problems or suggestions regarding PyRates, please contact [Richard Gast](https://www.richardgast.me).
