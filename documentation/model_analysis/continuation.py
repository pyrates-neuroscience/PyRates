"""
Parameter continuation and bifurcation detection
================================================

In this tutorial, you will learn how to perform a 1D
`numerical parameter continuation <http://www.scholarpedia.org/article/Numerical_analysis#Numerical_solution_of_
differential_and_integral_equations>`_ in PyRates with automatic fold `bifurcation
<http://www.scholarpedia.org/article/Bifurcation>`_ detection.
Furthermore, you will learn how to plot a simple bifurcation diagram. Throughout this example, we will use
the quadratic integrate-and-fire population model [1]_, a detailed introduction of which is given in the model
introductions example gallery. The dynamic equations of this model read the following:

.. math::
    \\tau \\dot r = \\frac{\\Delta}{\\pi\\tau} + 2 r v, \n
    \\tau \\dot v = v^2 +\\bar\\eta + I(t) + J r \\tau - (\\pi r \\tau)^2,

where :math:`r` is the average firing rate and :math:`v` is the average membrane potential of the QIF population.
It is governed by 4 parameters:

    - :math:`\\tau` --> the population time constant
    - :math:`\\bar \\eta` --> the mean of a Lorenzian distribution over the neural excitability in the population
    - :math:`\\Delta` --> the half-width at half maximum of the Lorenzian distribution over the neural excitability
    - :math:`J` --> the strength of the recurrent coupling inside the population

In this tutorial, we will demonstrate how to (1) , (2) perform a simple 1D parameter continuation in
:math:`\\bar \\eta`, and (3) plot the corresponding bifurcation diagram.
The latter has also been done in [1]_, so you can compare the resulting plot with the results reported by Montbrió et
al. For parts (2) and (3) of the tutorial, it is required that you have
`PyAuto <https://github.com/pyrates-neuroscience/PyAuto>`_ installed in the Python environment you are using.

References
----------
.. [1] E. Montbrió, D. Pazó, A. Roxin (2015) *Macroscopic description for networks of spiking neurons.* Physical
       Review X, 5:021028, https://doi.org/10.1103/PhysRevX.5.021028.
.. [2] E.J. Doedel, T.F. Fairgrieve, B. Sandstede, A.R. Champneys, Y.A. Kuznetsov and W. Xianjun (2007) *Auto-07p:
       Continuation and bifurcation software for ordinary differential equations.* Technical report,
       Department of Computer Science, Concordia University, Montreal, Quebec.
"""

import matplotlib.pyplot as plt
from pyrates import CircuitTemplate
from pyauto import PyAuto
import sys
sys.path.append('../')

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# %%
# Part 1: Creating a PyAuto Instance
# ==================================
#
# In this first part, we will be concerned with how to create a model representation that is compatible with auto-07p,
# which is the software that is used for parameter continuations and bifurcation analysis in PyRates [2]_.

# %%
# Step 1: Load the model into PyRates
# -----------------------------------
#
# As a first step, we have to load the model into PyRates. This is done the usual way. If you are not familiar with
# this, check out the example galleries for model definitions.

qif = CircuitTemplate.from_yaml("model_templates.neural_mass_models.qif.qif")

# %%
# Step 2: Generate the Fortran routines required by Auto-07p
# ----------------------------------------------------------
#
# In the next step, we will translate our model into a Fortran file containing all subroutines required by
# :code:`auto-07p`. In short, :code:`auto-07p` requires a fortran file with the model equations and initial values [2]_.
# This will require using the Fortran backend of PyRates and turning the :code:`vectorize` option off:

qif.get_run_func(func_name='qif_rhs', file_name='qif', step_size=1e-4, auto=True, backend='fortran', solver='pyauto',
                 vectorize=False, float_precision='float64')

# %%
# Calling the :code:`CircuitTemplate.get_run_func` method with :code:`auto=True` creates two files, which we will
# inspect below. The first file is a :code:`.f90` file containing all the Fortran subroutines required by
# :code:`auto-07p`:

f = open('qif.f90', 'r')
print('')
print(f.read())

# %%
# The second file is a textfile containing all the :code:`auto-07p` parameters that determine how it performs parameter
# continuations and automated bifurcation detection:

f = open('c.ivp', 'r')
print('')
print(f.read())

# %%
# The default parameters written out by PyRates allow to solve the initial value problem, i.e. perform simple numerical
# simulations via :code:`auto-07p`. For a detailed explanation of these parameters, see the :code:`auto-07p`
# `documentation<https://github.com/auto-07p/auto-07p>`_.

# %%
# Step 3: Generate a PyAuto instance
# ----------------------------------
#
# Now that the model equations are compiled, we can generate an instance of :code:`pyauto.PyAuto`, a Python
# `tool<https://github.com/pyrates-neuroscience/PyAuto>`_ that provides and interface to :code:`auto-07p`.

qif_auto = PyAuto(working_dir=None, auto_dir=auto_dir)

# %%
# Now, we can use all the tools provided by auto-07p to investigate how the model reacts to changes in its
# parameterization.

# %%
# Part 2: Performing Parameter Continuations
# ==========================================
#
# In this part, we will demonstrate how to perform simple 1D parameter continuations via the :code:`PyAuto.run()`
# method.

# %%
# Step 1: Time continuation
# -------------------------
#
# In parameter continuations, it is required that you start continuing the parameters from an
# `equilibrium <http://www.scholarpedia.org/article/Equilibrium>`_ or
# `periodic orbit <http://www.scholarpedia.org/article/Periodic_orbit>`_, i.e. that the solution to the
# `initial value problem <http://www.scholarpedia.org/article/Initial_value_problems>`_ would be constant or periodic
# in time. To achieve this, you can either set the initial values and parameters of your system to a known solution in
# the model definition (e.g. the YAML template), or choose a model parameterization for which a finite solution exists
# and then calculate the solution of the model in time until it converges to a equilibrium (or periodic orbit).
# If you go with the latter, you can then start to perform parameter continuations using the values of the state
# variables at the end point of your solution in time. This can be simply achieved by a call to the
# :code:`CircuitIR.run()` method, before calling the :code:`to_pyauto()` method. Then, PyRates will automatically use
# the values of the state variables from the last simulation step. Alternatively, you can perform simulations in time
# via PyAuto as follows:

t_sols, t_cont = qif_auto.run(
    e='qif', c='ivp', name='time', DS=1e-4, DSMIN=1e-10, EPSL=1e-08, EPSU=1e-08, EPSS=1e-06,
    DSMAX=1e-2, NMX=1000, UZR={14: 4.0}, STOP={'UZ1'})

qif_auto.plot_continuation('PAR(14)', 'U(1)', cont='time')
plt.show()

# %%
# In this function call, you see how the general interface of the :code:`PyAuto.run()` method works. In every first call
# of this method, the name of a fortran equations files needs to be specified by the keyword argument :code:`e`. The
# standard name of this function is :code:`rhs_func` when the file was automatically generated via PyRates. Also, we
# declare a constants file via :code:`c='ivp'`. This file has been automatically generated by PyRates as well and
# contains the constants that are required by auto-07p for time continuations. Both files can be expected in the build
# directory of PyRates (*pyrates_build/qif* in this case). All other arguments just overwrite important auto-07p
# constants that are also declared in the constants file :code:`c.ivp`. For a detailed explanation of those constants,
# please have a look at the `auto documentation <https://github.com/auto-07p/auto-07p/tree/master/doc>`_.
# Here, we will provide a short explanation of what each specific parameter does in our context:
#
#   - :code:`DS=1e-3` defines the initial step-size of the time continuation (in ms)
#   - :code:`DSMIN=1e-4` defines the minimal step-size of the time continuation (in ms)
#   - :code:`DSMAX=1.0` defines the maximal step-size of the time continuation (in ms)
#   - :code:`NMX=10000` defines the maximum number of continuation steps to perform
#   - :code:`UZR={14: 1000.0}` tells auto-07p to create a user-specified marker when the parameter 14, which is the
#     default parameter field in auto-07p in which time is stored, reaches a value of 1000.0 (ms)
#   - :code:`STOP={'UZ1'}` tells auto-07p to stop the continuation ones it hits the first user-specified marker
#
# The output of this call to :code:`.run()` is a tuple, with the following two entries:
#
#   - :code:`dict` --> A dictionary that contains a summary of the parameter, variables and other characteristics of
#     each solution along the branch that has been generated by the continuation.
#   - :code:`branch` --> An auto-07p object that has been generated during the :code:`.run()` call and is required
#     for subsequent parameter continuations.

# %%
# Step 2: Continuation of :math:`\bar \eta`
# -----------------------------------------
#
# If you look at the auto-07p output in the terminal, you will see that the values for :code:`U(1)` and :code:`U(2)`
# converged to certain values. These two values represent the current values of our state variables :math:`r` and
# :math:`v`. Thus, our model converged to an equilibrium and we are now save to perform the continuation in our
# parameter of interest: :math:`\bar \eta`. This follows a very similar syntax:

eta_sols, eta_cont = qif_auto.run(
    origin=t_cont, starting_point='UZ1', name='eta', bidirectional=True,
    ICP=4, RL0=-20.0, RL1=20.0, IPS=1, ILP=1, ISP=2, ISW=1, NTST=400,
    NCOL=4, IAD=3, IPLT=0, NBC=0, NINT=0, NMX=2000, NPR=10, MXBF=5, IID=2,
    ITMX=40, ITNW=40, NWTN=12, JAC=0, EPSL=1e-06, EPSU=1e-06, EPSS=1e-04,
    DS=1e-4, DSMIN=1e-8, DSMAX=5e-2, IADS=1, THL={}, THU={}, UZR={}, STOP={}
)

# %%
# In this call, we specified the full set of auto-07p constants. Don't worry, usually, you do not have to bother with
# most of them. It is common practice to specify most of them in constants files that you would refer to the same way as
# in the previous call to the :code:`.run()` method, via :code:`c=name`. In such a case, you would specify all auto-07p
# constants that do not change between calls to the :code:`.run()` method in a file with the name *c.name* and only
# provide the constants that need to be altered between :code:`.run()` calls directly to the :code:`.run()` method.
#
# While it is out of the scope of this tutorial to explain all auto-07p constants here. Instead, we will provide an
# intuitive explanation of the most important ones and refer to the
# `auto documentation <https://github.com/auto-07p/auto-07p/tree/master/doc>`_ for the rest:
#
#   - :code:`origin='t_cont'` is a keyword argument specific to `PyAuto`. It tells auto-07p from which branch of
#     solutions to start the parameter continuation from. This needs to be specified for every call to the
#     :code:`.run()` method, except for the first (since their is no solutions branch at this point). Here, we
#     specified the solution branch from our initial continuation in time.
#   - :code:`starting_point='UZ1'` is a keyword argument specific to `PyAuto`. It tells auto-07p to start the
#     continuation from the first user-specified marker of the provided origin
#   - :code:`name='eta'` is a keyword argument specific to `PyAuto`. It tells PyAuto to store the results of the
#     continuation using this particular name. We can use this name for later continuations or for plotting, to indicate
#     which continuation to start from or to plot the results of.
#   - :code:`bidirectional=True` is a keyword argument specific to `PyAuto`. It tells PyAuto to change :math:`\bar\eta`
#     both in the positive and the negative direction.
#   - :code:`ICP=4` tells auto-07p to perform a 1D continuation over parameter number 4 (you can check in the fortran
#     file *rhs_func.f* that :math:`\bar\eta` is indeed the 4th parameter)
#   - :code:`RL0=-20.0` and :code:`RL1=20.0` specify the boundaries of the parameter continuation. If :math:`\bar\eta`
#     is continued beyond any of these borders, auto-07p stops the parameter continuation.
#   - :code:`IPS=1` indicates auto-07p that it is supposed to continue an equilibrium of an ODE system
#   - :code:`ILP=1` turns on the detection of fold bifurcations in auto-07p
#   - :code:`ISP=2` turns on full automatic bifurcation detection in auto-07p
#
# Checking the terminal output of auto-07p, you will realize that the output in column *TY* shows *LP* for two of the
# solutions we computed along our branch in :math:`\bar\eta`. These indicate the detection of *limit point* or
# `fold bifurcations <http://www.scholarpedia.org/article/Saddle-node_bifurcation>`_. We can visualize the full
# bifurcation diagram via the following call:

qif_auto.plot_continuation('PAR(4)', 'U(1)', cont='eta')
plt.show()

# %%
# The curve in this plot represents the value of :math:`r` (y-axis) at the equilibrium solutions that exist for each
# value of :math:`\bar\eta` (x-axis). A solid line indicates that the equilibrium is stable, whereas a dotted line
# indicated that the equilibrium is unstable. The triangles mark the points at which auto-07p detected fold
# bifurcations. At a fold bifurcation, the critical eigenvalue of the vector field defined by the right-hand sides of
# our model's ODEs crosses the imaginary axis (i.e. its real part changes the sign). This indicates a change of
# `stability <http://www.scholarpedia.org/article/Bifurcation_diagram>`_ of an equilibrium solution, which happens
# because an unstable and a stable equilibiurm approach and annihilate each other. This behavior can be read from the
# plot as well. The solid and the dotted line approach each other towards the fold bifurcation marks. After they meet
# at the fold bifurcation, both cease to exist.

# %%
# Final step: Clean up all temporary files
# ----------------------------------------
#
# As a last step, it is good practice to clean up all temporary files created by PyAuto and PyRates. This can be
# achieved with the following simple call:

qif.clear()
