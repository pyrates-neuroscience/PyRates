"""
A Mean-Field Model of a Quadratic Integrate-and-Fire (QIF) Neuron Population
============================================================================

Here, we will introduce the QIF population mean-field model, which has been derived from a population of all-to-all
coupled QIF neurons in [1]_. The model equations are given by:

.. math::

    \\tau \\dot r = \\frac{\\Delta}{\\pi\\tau} + 2 r v, \n
    \\tau \\dot v = v^2 +\\bar\\eta + I(t) + J r \\tau - (\\pi r \\tau)^2,

where :math:`r` is the average firing rate and :math:`v` is the average membrane potential of the QIF population [1]_.
It is governed by 4 parameters:
    - :math:`\\tau` --> the population time constant
    - :math:`\\bar \\eta` --> the mean of a Lorenzian distribution over the neural excitability in the population
    - :math:`\\Delta` --> the half-width at half maximum of the Lorenzian distribution over the neural excitability
    - :math:`J` --> the strength of the recurrent coupling inside the population
This mean-field model is an exact representation of the macroscopic firing rate and membrane potential dynamics of a
spiking neural network consisting of QIF neurons with Lorentzian distributed background excitabilities.
While the mean-field derivation is mathematically only valid for all-to-all coupled populations of infinite size,
it has been shown that there is a close correspondence between the mean-field model and neural populations with
sparse coupling and population sizes of a few thousand neurons [2]_. In the same work, it has been demonstrated how to
extend the model by adding synaptic dynamics or additional adaptation currents to the single cell network, that can be
carried through the mean-field derivation performed in [1]_. For example, a QIF population with spike-frequency
adaptation would be given by the following 4D system:

.. math::

    \\tau \\dot r = \\frac{\\Delta}{\\pi\\tau} + 2 r v, \n
    \\tau \\dot v = v^2 +\\bar\\eta + I(t) + J r \\tau - A - (\\pi r \\tau)^2, \n
    \\tau_A \\dot A = B, \n
    \\tau_A \\dot B = \\alpha r - 2 B - A,

where the evolution equations for :math:`A` and :math:`B` express a convolution of :math:`r` with an alpha kernel, with
adaptation strength :math:`\\alpha` and time constant :math:`\\tau_A`.

In the sections below, we will demonstrate for each model how to load the model template into pyrates, perform
simulations with it and visualize the results.

References
----------

.. [1] E. Montbrió, D. Pazó, A. Roxin (2015) *Macroscopic description for networks of spiking neurons.* Physical
       Review X, 5:021028, https://doi.org/10.1103/PhysRevX.5.021028.

.. [2] R. Gast, H. Schmidt, T.R. Knösche (2020) *A Mean-Field Description of Bursting Dynamics in Spiking Neural
       Networks with Short-Term Adaptation.* Neural Computation (in press).

"""

# %%
# Basic QIF Model
# ===============
#
# We will start out by a step-by-step tutorial of how to use the QIF model without adaptation.

# %%
# Step 1: Importing the frontend class for defining models
# --------------------------------------------------------
#
# As a first step, we import the :code:`pyrates.frontend.CircuitTemplate` class, which allows us to set up a model
# definition in PyRates.

from pyrates.frontend import CircuitTemplate

# %%
# Step 2: Loading a model template from the `model_templates` library
# -------------------------------------------------------------------
#
# In the second step, we load the model template for an excitatory QIF population that comes with PyRates via the
# :code:`from_yaml()` method of the :code:`CircuitTemplate`. This method returns a :code:`CircuitTemplate` instance
# which provides the method :code:`apply()` for turning it into a graph-based representation, i.e. a
# :code:`pyrates.ir.CircuitIR` instance. Have a look at the yaml definition of the model that can be found at the path
# used for the :code:`from_yaml()` method. You will see that all variables and parameters are already defined there.
# These are the basic steps you perform, if you want to load a model that is
# defined inside a yaml file. To check out the different model templates provided by PyRates, have a look at
# the :code:`PyRates.model_templates` module.

qif_circuit = CircuitTemplate.from_yaml("model_templates.montbrio.simple_montbrio.QIF_exc").apply()

# %%
# Step 3: Loading the model into the backend
# ------------------------------------------
#
# In this example, we directly load the :code:`CircuitIR` instance into the backend via the  :code:`compile()` method
# without any further changes to the graph. This way, a :code:`pyrates.backend.NumpyBackend` instance is created.
# After this step, structural modifications of the network are not possible anymore.

qif_compiled = qif_circuit.compile(backend='numpy', step_size=1e-3)

# %%
# Step 4: Numerical simulation of a the model behavior in time
# ------------------------------------------------------------
#
# After loading the model into the backend, numerical simulations can be performed via the :code:`run()` method.
# Calling this function will solve the initial value problem of the above defined differential equations for a time
# interval from 0 to the given simulation time.
# This solution will be calculated numerically by a differential equation solver in the backend, starting with a defined
# step-size.

results = qif_compiled.run(simulation_time=40.0, outputs={'r': 'p/Op_e/r'})

# %%
# Step 5: Visualization of the solution
# -------------------------------------
#
# The output of the :code:`run()` method is a :code:`pandas.Dataframe`, which comes with a :code:`plot()` method for
# plotting the timeseries it contains.
# This timeseries represents the numerical solution of the initial value problem solved in step 4 with respect to the
# state variable :math:`r` of the model.

results.plot()

# %%
# QIF SFA Model
# =============
#
# Now, lets have a look at the QIF model with spike-frequency adaptation. We will follow the same steps as outlined
# above.

qif_sfa_circuit = CircuitTemplate.from_yaml("model_templates.montbrio.simple_montbrio.QIF_sfa").apply()
qif_sfa_compiled = qif_sfa_circuit.compile(backend='numpy', step_size=1e-3)
results = qif_sfa_compiled.run(simulation_time=40.0, outputs={'r': 'p/Op_sfa/r'})
results.plot()

# %%
# you can see that, by adding the adaptation variable to the model, we introduced synchronized bursting behavior to
# the model. Check out [2]_ if you would like to test out some different parameter regimes and would like to know what
# kind of model behavior to expect if you make changes to the adaptation parameters. To change the parameters, you need
# to derive a new operator template from the given operator template in a yaml file and simply set the parameter you
# would like to change. For a detailed introduction on how to handle model definitions via YAML files, have a look at
# the model definition gallery.
