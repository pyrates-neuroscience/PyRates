"""
Minimal Example
===============

In this minimal example, we load a model from the `model_templates` module, simulate its behavior and plot the resulting time series.
The model represents the macroscopic dynamics of a population of quadratic integrate-and-fire (QIF) neurons and we record the average firing rate of the population.

The model equations are given by:

.. math::

    \\tau \\dot r = \\frac{\\Delta}{\\pi\\tau} + 2 r v, \n
    \\tau \\dot v = v^2 +\\bar\\eta + I(t) + J r \\tau - (\\pi r \\tau)^2.

where `r` is the average firing rate and `v` is the average membrane potential of the QIF population [1].

References
----------

.. [1] Montbrió et al. (2015) Phys Rev X.

"""

# %%
# Step 1: Importing the frontend class for defining models
# --------------------------------------------------------
#
# As a first step, we import the `pyrates.frontend.CircuitTemplate` class, which allows us to set up a model definition in PyRates.

from pyrates.frontend import CircuitTemplate

# %%
# Step 2: Loading a model template from the `model_templates` library
# -------------------------------------------------------------------
#
# In the second step, we load one of the model templates that comes with PyRates via the `from_yaml()` method of the `CircuitTemplate`.
# This method returns a `CircuitTemplate` instance which provides the method `apply()` for turning it into a graph-based representation, i.e. a `pyrates.ir.CircuitIR` instance.
# These are the basic steps you perform, if you want to load a model that is defined inside a yaml file.
# To check out the different model templates provided by PyRates, have a look at the `PyRates.model_templates` module.

circuit = CircuitTemplate.from_yaml("model_templates.montbrio.simple_montbrio.Net1").apply()

# %%
# Step 3: Loading the model into the backend
# ------------------------------------------
#
# In this example, we directly transform the `CircuitIR` instance into a `ComputeGraph` instance via the `compile()`
# method without any further changes to the graph.
# This way, our network is loaded into the backend, i.e. a `pyrates.backend.NumpyBackend` instance is created.
# After this step, structural modifications of the network are not possible anymore.

compute_graph = circuit.compile(backend='numpy', step_size=1e-3)

# %%
# Step 4: Numerical simulation of a the model behavior in time
# ------------------------------------------------------------
#
# After loading the model into the backend, numerical simulations can be performed via the `run()` method.
# Calling this function will solve the initial value problem of the above defined differential equations for a time
# interval from 0 to the given simulation time.
# This solution will be calculated numerically by a differential equation solver in the backend, starting with a defined
# step-size.

results = compute_graph.run(simulation_time=40.0, outputs={'r': 'Pop1/Op_e/r'})

# %%
# Step 5: Visualization of the solution
# -------------------------------------
#
# The output of the `run()` method is a `pandas.Dataframe`, which comes with a `plot()` method for plotting the
# timeseries it contains.
# This timeseries represents the numerical solution of the initial value problem solved in step 4 with respect to the
# state variable `r` of the model.

results.plot()
