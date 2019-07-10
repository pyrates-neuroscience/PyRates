[![](https://img.shields.io/github/license/pyrates-neuroscience/PyRates.svg)](https://github.com/pyrates-neuroscience/PyRates) 
[![Build Status](https://travis-ci.com/pyrates-neuroscience/PyRates.svg?branch=master)](https://travis-ci.com/pyrates-neuroscience/PyRates)
<img src="https://github.com/pyrates-neuroscience/PyRates/blob/master/PyRates_logo_color.svg" width="20%" heigth="20%" align="right">
[![PyPI version](https://badge.fury.io/py/pyrates.svg)](https://badge.fury.io/py/pyrates)
 
# PyRates
PyRates is a framework for neural modeling and simulations, developed by Richard Gast and Daniel Rose at the Max Planck Institute of Human Cognitive and Brain Sciences, Leipzig, Germany. 

Basic features:
---------------
- Different backends: `Numpy` for fast simulations of small- to medium-sized networks. `Tensorflow` for large networks that can be efficiently parallelized on GPUs/CPUs.
- Each model is internally represented by a `networkx` graph of nodes and edges, with the former representing the model units (i.e. single cells, cell populations, ...) and the latter the information transfer between them. In principle, this allows to implement any kind of dynamic neural system that can be expressed as a graph via PyRates.
- The user has full control over the mathematical equations that nodes and edges are defined by. 
- Model configuration and simulation can be done within a few lines of code.  
- Various templates for rate-based population models are provided that can be used for neural network simulations imediatly.
- Visualization and data analysis tools are provided.
- Tools for fast and parallelized exploration of model parameter spaces are provided.

Installation
------------
PyRates can be installed via the `pip` command. We recommend to use `Anaconda` to create a new python environment with Python >= 3.6 and then simply run the following line from a terminal with the environment being activated:
```
pip install pyrates
```
Alternatively, it is possible to clone this repository and run the following line from the directory in which the repository was cloned:
```
python setup.py install
```

Documentation
-------------
For a full API of PyRates, see https://pyrates.readthedocs.io/en/latest/.
For examplary simulations and model configurations, please have a look at the jupyter notebooks provided in the documenation folder.

Reference
---------

If you use this framework, please cite:
Gast, R., Daniel, R., Moeller, H. E., Weiskopf, N. and Knoesche, T. R. (2019). “PyRates – A Python Framework for rate-based neural Simulations.” bioRxiv (https://www.biorxiv.org/content/10.1101/608067v2).

Contact
-------

If you have questions, problems or suggestions regarding PyRates, please contact [Richard Gast](https://www.cbs.mpg.de/person/59190/376039) or [Daniel Rose](https://www.cbs.mpg.de/person/51141/374227).
