Model Definition
----------------

In PyRates, neurodynamic models are defined via sets of differential equations, for which solutions with respect to time
can be found numerically. Such models can be defined via two different interfaces in PyRates: via YAML files or via the
Python frontend. In this gallery, we will provide walk-throughs for all features and options that you have when
defining a model in PyRates.

This includes instructions for:

* how to create YAML templates for operators, nodes, edges and circuits in PyRates,
* how to handle the template classes of the PyRates frontend,
* how to define models via the Python-based frontend interface,
* how to build hierarchically organized circuits with multiple levels of nodes
