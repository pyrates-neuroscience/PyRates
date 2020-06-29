"""
Differential Evolution
======================

In this tutorial, you will learn how to optimize PyRates models via the
`differential evolution <https://en.wikipedia.org/wiki/Differential_evolution>`_ strategy introduced in [1]_.
It will be based on the same model and the same parameter as the single parameter grid search example. So it will be
worthwhile to first have a look at that example, before proceeding.

Shortly, we will use the Jansen-Rit model (check out the model introduction for the Jansen-Rit model, to learn
about the mathematics behind the model and about its implementation in PyRates) [2]_.
We will perform a 1D evolutionary optimization over its connectivity scaling parameter :math:`C`. This parameter scales
all synaptic strengths in the Jansen-Rit neural mass model and has a critical influence on its behavior [2]_.

The general idea behind evolutionary model optimization strategies is to optimize a set of model parameters with
respect to some objective function that defines the fitness of a certain model parametrization. Initially, a number of
different model parameterizations are sampled from a defined parameter space. Then, the iterative optimization starts.
At each iteration of the optimization algorithm, the following two steps are performed:

    1. The objective function is evaluated for each parameterization, resulting in a fitness value for each model
       parameterization
    2. The fitness values are used to sample new model parameterizations, mutate the old parameterizations, or create
       new parameterizations via combinations of the old parameterizations

These iterations are then repeated until a fitness criterion is reached. The exact functional relationships that are
used to translate fitness values and old parameterizations into new parameterizations depend on the type of evolutionary
model optimization strategy that is used. For a summary of differential evolution, have a look at
`this article <https://en.wikipedia.org/wiki/Differential_evolution>`_. In PyRates, we simply provide an interface to
the differential evolution optimization provided by SciPy [3]_.

References
^^^^^^^^^^

.. [1] R. Storn and K. Price (1997) *Differential Evolution - a Simple and Efficient Heuristic for Global Optimization
       over Continuous Spaces.* Journal of Global Optimization, 11: 341-359.

.. [2] B.H. Jansen & V.G. Rit (1995) *Electroencephalogram and visual evoked potential generation in a mathematical
       model of coupled cortical columns.* Biological Cybernetics, 73(4): 357-366.

.. [3] P. Virtanen, R. Gommers et al. (2020) *SciPy 1.0: fundamental algorithms for scientific computing in Python.*
       Nature Methods, 17: 261-272. https://doi.org/10.1038/s41592-019-0686-2.
"""

# %%
# First, let's import the :code:`DifferentialEvolutionAlgorithm` class from PyRates

from pyrates.utility.genetic_algorithm import DifferentialEvolutionAlgorithm
import numpy as np

# %%
# Definition of the optimization details
# --------------------------------------
#
# (1) To optimize our parameter :math:`C`, we will have to define the parameter boundaries within which the optimization
# should be performed:

params = {'C': {'min': 1.0, 'max': 1000.0}}

# %%
# (2) Furthermore, we need to define the model and the model parameter that the :math:`C` refers to:

model_template = "model_templates.jansen_rit.simple_jansenrit.JRC_simple"
param_map = {'C': {'vars': ['JRC_op/c'], 'nodes': ['JRC']}}

# %%
# (3) Finally, we have to define the objective function that should be optimized. This objective function always needs
# to calculate a scalar fitness, based on model output. Thus, we first define the model output:

output = {'V_pce': 'JRC/JRC_op/PSP_pc_e', 'V_pci': 'JRC/JRC_op/PSP_pc_i'}

# %%
# ...and then the objective function:


def loss(data, min_amp=6.0, max_amp=10.0):
    """Calculates the difference between the value range in the data and the
    range defined by min_amp and max_amp.
    """

    # calculate the membrane potential of the PC population
    data = data['V_pce'] - data['V_pci']

    # calculate the difference between the membrane potential range
    # of the model and the target membrane potential range
    data_bounds = np.asarray([np.min(data), np.max(data)]).squeeze()
    target_bounds = np.asarray([min_amp, max_amp])
    diff = data_bounds - target_bounds

    # return the sum of the squared errors
    return diff @ diff.T

# %%
# The value of this loss function depends on the minimum and the maximum value of the average membrane potential of the
# pyramidal cell population of the Jansen-Rit model [1]_. Depending on the :code:`min_amp` and :code:`max_amp`
# arguments of that function, the differential evolution algorithm should optimize the parameter :math:`C` of our model
# such that the minimum and maximum membrane potential fluctuations of the PC population are as close to those values
# as possible. Therefore, this function should suffice to find model parameterizations that express oscillatory behavior
# with different oscillation amplitudes or non-oscillatory behavior.
#
# Performing the model optimization
# ---------------------------------
#
# Now, we have prepared everything to start the optimization. This will be done via an instance of
# :code:`pyrates.utility.genetic_algorithm.DifferentialEvolutionAlgorithm`:

diff_eq = DifferentialEvolutionAlgorithm()

# %%
# To start the optimization, simply use its :code:`run()` method:

winner = diff_eq.run(initial_gene_pool=params,
                     gene_map=param_map,
                     template=model_template,
                     compile_kwargs={'solver': 'scipy', 'backend': 'numpy', 'step_size': 1e-4, 'verbose': False},
                     run_kwargs={'step_size': 1e-4, 'simulation_time': 3., 'sampling_step_size': 1e-2, 'verbose': False,
                                 'outputs': {'V_pce': 'JRC/JRC_op/PSP_pc_e', 'V_pci': 'JRC/JRC_op/PSP_pc_i'}},
                     loss_func=loss,
                     loss_kwargs={'min_amp': 6.0, 'max_amp': 10.0},
                     workers=-1)

# %%
# This function provides an interface to :code:`scipy.optimize.differential_evolution`, for which a detailed
# documentation can be found
# `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html>`_. All
# arguments that :code:`scipy.optimize.differential_evolution` takes can also be provided as keyword arguments to the
# :code:`run()` method. By providing a :code:`template`, :code:`compile_kwargs` and :code:`run_kwargs`, the
# :code:`run()` method knows that it should use the provided model template, load it into the backend via
# :code:`CircuitIR.compile(**compile_kwargs)` and then simulate its behavior via :code:`CircuitIR.run(**run_kwargs)`.
# The resulting timeseries is then forwarded to the :code:`loss_func` together with the keyword arguments in
# :code:`loss_kwargs`.
#
# The return value of the :code:`run()` method contains the winning parameter set and its loss function value.
# Let's check out, whether this model parameter indeed produces the behavior we optimized for:

from pyrates.frontend import CircuitTemplate
from matplotlib.pyplot import show

jr_temp = CircuitTemplate.from_yaml(model_template).apply()
jr_temp.set_node_var('JRC/JRC_op/c', winner.at[0, 'C'])
jr_comp = jr_temp.compile(solver='scipy', backend='numpy', step_size=1e-4)
results = jr_comp.run(simulation_time=3.0, sampling_step_size=1e-2,
                      outputs={'V_pce': 'JRC/JRC_op/PSP_pc_e', 'V_pci': 'JRC/JRC_op/PSP_pc_i'})

results = results['V_pce'] - results['V_pci']
results.plot()
show()

# %%
# As can be seen, the model shows oscillatory behavior with minimum and maximum membrane potential amplitudes that
# are close to our target values of 6.0 and 10.0 mV.
