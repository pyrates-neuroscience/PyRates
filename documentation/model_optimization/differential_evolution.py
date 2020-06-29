"""
Differential Evolution
======================

In this tutorial, you will learn how to optimize PyRates models via the
`differential evolution <https://en.wikipedia.org/wiki/Differential_evolution>`_ strategy introduced in [1]_.

References
^^^^^^^^^^

.. [1] R. Storn and K. Price (1997) *Differential Evolution - a Simple and Efficient Heuristic for Global Optimization
       over Continuous Spaces.* Journal of Global Optimization, 11: 341-359.
"""

from pyrates.utility.genetic_algorithm import DifferentialEvolutionAlgorithm
import numpy as np


def fitness(data, min_amp=6.0, max_amp=10.0):
    data = data['V_pce'] - data['V_pci']
    data_bounds = np.asarray([np.min(data), np.max(data)]).squeeze()
    target_bounds = np.asarray([min_amp, max_amp])
    diff = data_bounds - target_bounds
    return np.sqrt(diff @ diff.T)


diff_eq = DifferentialEvolutionAlgorithm()
diff_eq.run(initial_gene_pool={'C': {'min': 1.0, 'max': 1000.0}},
            gene_map={'C': {'vars': ['JRC_op/c'], 'nodes': ['JRC']}},
            template="model_templates.jansen_rit.simple_jansenrit.JRC_simple",
            compile_kwargs={'solver': 'scipy', 'backend': 'numpy', 'step_size': 1e-4},
            run_func_kwargs={'step_size': 1e-4, 'simulation_time': 3.0, 'sampling_step_size': 1e-2,
                             'outputs': {'V_pce': 'JRC/JRC_op/PSP_pc_e', 'V_pci': 'JRC/JRC_op/PSP_pc_i'}},
            fitness_func=fitness,
            fitness_func_kwargs={'min_amp': 6.0, 'max_amp': 10.0},
            workers=-1)
