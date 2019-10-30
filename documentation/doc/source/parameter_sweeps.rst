
Parameter Sweeps
================

This section demonstrates how to use the `grid_search()` method and the `ClusterGridSearch` class from the `pyrates.utility` module to perform parameter sweeps on models in PyRates.
Below, you will find the code to run a parameter sweep on the Montbrio model described in more detail in "Gast, R. et al. (2018) PyRates - A Python framework for neural modeling and simulations on parallel hardware". The Montbrio model is a mean-field model of a globally coupled population of quadratic integrate-and-fire neurons, initially proposed by Montbrio and colleagues in 2015: "Montbrio et al. (2015) ....". The PyRates model implementation can be found under templates/montbrio.

grid_search()
-------------

The `grid_search()` function is a dedicated tool for parameter space investigations. It is optimized to run multiple instances of the same model with different parametrizations in parallel.
In the following example we simulate two coupled montbrio populations, representing a pyramid cell (PC) and an inhibitory inter neuron (IIN), respectively, for different excitability levels :math:`\eta_{pc}` and :math:`\eta_{iin}` of the respective population.

1) Parameter definition: In the following cell, the different conditions and basic simulation parameters are defined::

    import numpy as np
    from pyrates.utility.grid_search import grid_search

    circuit_template = "model_templates.montbrio.simple_montbrio.Net3"
    dt = 1e-4                                     # integration step size in s
    dts = 1e-3                                    # variable storage sub-sampling step size in s
    T = 5.                                        # total simulation time in s
    inp = np.ones((int(T/dt), 1))                 # external input to the population
    inputs = {"PC/Op_e/inp": inp}
    outputs = {'r': 'PC/Op_e/r'}                  # model output parameters

2) In the cell below, multiple parametrizations of the model are defined and the simulation will be run. In the parameter grid, multiple values can be specified for a certain model parameter, whereas in the parameter map the parameter grid keys are mapped to explicit model parameters defined in the circuit template. Additional arguments can be added to customize the computation. The model output for each parametrization is stored in the 'results' DataFrame, whereas the 'result_map' yields the parameter combinations of each parametrization.:::

    param_grid = {'eta_op': np.linspace(-5, 5, 5),
                  'eta_iin': np.linspace(-5, 5, 5)}

    param_map = {'eta_op': {'vars': ['Op_e/eta'],
                            'nodes': ['PC']},
                 'eta_iin': {'vars': ['Op_i/eta'],
                             'nodes': ['IIN']}}

    # Execution
    results, result_map, t_ = grid_search(
        circuit_template=circuit_template,
        param_grid=param_grid,
        param_map=param_map,
        simulation_time=T,
        dt=dt,
        sampling_step_size=dts,
        inputs=inputs.copy(),
        outputs=outputs.copy(),
        permute_grid=True,                            # Creates all permutations of the param_grid values if set to True
        init_kwargs={                                 # Additional keyword arguments for the compute graph
            'backend': 'numpy',                       # Can be either 'numpy' or 'tensorflow'
            'solver': 'euler'                         # Solver for the differential equation approximation.
        },
        profile='t',                                  # Automated runtime tracking
        njit=True,                                    # Uses Numba's njit compiler if set to True
        parallel=False
    )
    results.plot()

ClusterGridSearch
-----------------

The ClusterGridSearch class extends PyRates' tools for parameter space investigations with the ability to distribute computations among multiple workstation in a computer network. This is especially useful when investigating high-dimensional parameter spaces that require large parameter grids which might exceed the hardware capacities of a single workstation. In the following example we present the same parameter space invetigation as presented above, yet utilizing PyRates' ClusterGridSearch module.

1) Initialization: A cluster computation consist of two components, the cluster initialization and the actual model computation. During its initialization a ClusterGridSearch (CGS) object expects a list of computer names in the computer network that will be utilized for the computation. Optionally, a compute directory can be specified where all result files, transfer data and configuration files will be stored. If no explicit path is provided, a compute directory is created at the same location of the calling python script.::

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    from pyrates.utility.grid_search import ClusterGridSearch
    %matplotlib inline

    nodes = [
        'animals',
        'spanien'
    ]

    compute_dir = "/nobackup/spanien1/salomon/CGS/Benchmark_jup"

    cgs = ClusterGridSearch(nodes, compute_dir=compute_dir)

2) Execution: In the cell below, first all simulation and cluster parameters are set. For a detailed description of the CGS specific parameters please see the function documentation. The CGS.run() method does not return DataFrames, but filepaths to result files of the cluster computation. These result files can be then easily loaded into DataFrames using pandas' read_hdf() function.::

    # Simulation/grid search parameters
    circuit_template = "model_templates.montbrio.simple_montbrio.Net3"
    dt = 1e-4                                     # integration step size in s
    dts = 1e-3                                    # variable storage sub-sampling step size in s
    T = 5.                                        # total simulation time in s
    inp = np.ones((int(T/dt), 1))                 # external input to the population
    inputs = {"PC/Op_e/inp": inp}
    outputs = {'r': 'PC/Op_e/r'}                  # model output parameters

    param_grid = {'eta_op': np.linspace(-5, 5, 5),
                  'eta_iin': np.linspace(-5, 5, 5)}

    param_map = {'eta_op': {'vars': ['Op_e/eta'],
                            'nodes': ['PC']},
                 'eta_iin': {'vars': ['Op_i/eta'],
                             'nodes': ['IIN']}}

    # CGS specific parameters
    chunk_size = 5  # [10,5]
    worker_env = "/data/u_salomon_software/anaconda3/envs/PyRates/bin/python3"
    worker_file = '/data/hu_salomon/PycharmProjects/PyRates/pyrates/utility/worker_template.py'
    add_template_info = False
    config_kwargs = {
        "init_kwargs": {
            'backend': 'numpy',
            'solver': 'euler'
        }
    }

    # Simulation run
    res_file = cgs.run(
        circuit_template=circuit_template,
        params=param_grid,
        param_map=param_map,
        simulation_time=T,
        dt=dt,
        permute=True,
        sampling_step_size=dts,
        inputs=inputs,
        outputs=outputs,
        chunk_size=chunk_size,
        worker_env=worker_env,
        worker_file=worker_file,
        add_template_info=add_template_info,
        config_kwargs=config_kwargs)

    results = pd.read_hdf(res_file, key=f'Results/results')
    result_map = pd.read_hdf(res_file, key='Results/result_map')

    results.plot()


3) CGS postprocessing
