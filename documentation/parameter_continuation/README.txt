Parameter Continuations
=======================

In this gallery, you will find tutorials that demonstrate how to perform parameter continuations (1D, 2D and 3D)
and bifurcation anaylses via the aut0-07p interface of PyRates.

**RESTRICTION:** This feature only works for scalar networks, as of now. This means, that the model has to be defined
inside a single operator, as a set of coupled differential equations.

PyRates will automatically generate the fortran files required to run auto-07p [1]_. You will then be able to access
every feature of auto-07p via the `pyrates.utility.pyauto` module.

**REQUIREMENT:** You will have to install `auto-07p <https://github.com/auto-07p/auto-07p>`_ on your machine and follow
these `installation instructions <https://github.com/auto-07p/auto-07p/tree/master/doc>`_ for any of the examples below
to work.

References
----------

.. [1] E.J. Doedel, T.F. Fairgrieve, B. Sandstede, A.R. Champneys, Y.A. Kuznetsov and W. Xianjun (2007) *Auto-07p:
       Continuation and bifurcation software for ordinary differential equations.* Technical report,
       Department of Computer Science, Concordia University, Montreal, Quebec.
