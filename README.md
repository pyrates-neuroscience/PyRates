[![](https://img.shields.io/github/license/pyrates-neuroscience/PyRates.svg)](https://github.com/pyrates-neuroscience/PyRates) 
[![Build Status](https://travis-ci.com/pyrates-neuroscience/PyRates.svg?branch=master)](https://travis-ci.com/pyrates-neuroscience/PyRates)
<img src="https://github.com/pyrates-neuroscience/PyRates/blob/master/PyRates_logo.png" width="20%" heigth="20%" align="right">

# PyRates
PyRates is a framework for neural modeling and simulations, developed by Richard Gast and Daniel Rose at the Max Planck Institute of Human Cognitive and Brain Sciences, Leipzig, Germany. 

Basic features:
---------------
- Every model implemented in PyRates is translated into a tensorflow graph, a powerful compute engine that provides efficient CPU and GPU parallelization. 
- Each model is internally represented by a networkx graph of nodes and edges, with the former representing the model units (i.e. single cells, cell populations, ...) and the latter the information transfer between them. In principle, this allows to implement any kind of dynamic neural system that can be expressed as a graph via PyRates.
- The user has full control over the mathematical equations that nodes and edges are defined by. 
- Model configuration and simulation can be done within a few lines of code.  
- Various templates for rate-based population models are provided that can be used for neural network simulations imediatly.
- Visualization and data analysis tools are provided.
- Tools for the exploration of model parameter spaces are provided.

Documentation
-------------
For a full API of PyRates, see READTHEDOCSLINK.
For examplary simulations and model configurations, please have a look at the jupyter notebooks in the documenation folder.

Reference
---------

If you use this framework, please cite:
Gast, R., Knoesche, T. R., Daniel, R., Moeller, H. E., and Weiskopf, N. (2018). “P168 pyrates: A python framework for rate-based neural simulations.” BMC Neuroscience. 27th Annual Computational Neuroscience Meeting (CNS*2018): Part One.

Contact
-------

If you have questions, problems or suggestions regarding PyRates, please contact [Richard Gast](https://www.cbs.mpg.de/person/59190/376039) or [Daniel Rose](https://www.cbs.mpg.de/person/51141/374227).
