"""
Numerical Simulations
=====================

In this tutorial, you will learn the different options that PyRates provides for performing numerical simulations.
Given a dynamical system with state vector :math:`\\mathbb{y}`, the evolution of which is given by

.. math::
    \\dot \\mathbb{y} = \\mathbb{f}(\\mathbb{y}, t)

with vector field :math:`f`, we are interested in the state of the system at each time point :math:`t`.
Given an initial time :math:`t_0` and an initial state :math:`y_0`, this problem can be solved by evaluating

.. math::
    \\mathbb{y}(t) = \\int_{t_0}^{t} \\mathbb{f}(\\mathbb{y}, t') dt'.

This is known as the initial value problem (IVP) and there exist various algorithms for approximating the solution to
the IVP for systems where an analytic solution is intractable.
Many of these algorithms are available via PyRates, and we will demonstrate below how to use a sub-set of them.
To this end, we will solve the IVP for the QIF mean-field model, for which a detailed model introduction exists in the
model introduction gallery.
"""