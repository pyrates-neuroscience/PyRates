
Simulations
===========

The examples below teach how to use the `run()` method of the `CircuitIR`. As an exemplary model, the minimal example of the `HowTo` section is used, i.e. we assume the following code to be executed already::

	from pyrates.frontend import CircuitTemplate
	
	dt = 1e-3
	circuit = CircuitTemplate.from_yaml("model_templates.montbrio.simple_montbrio.Net1").apply()
	compute_graph = circuit.compile(dt=dt)

Basic Run Interface
-------------------

In this example, the basic interface of the `CircuitIR.run` method is explained::
	
	import numpy as np
	
	T = 10.0
	inp = np.zeros((int(T/dt), 1)) + 3.0

	results = compute_graph.run(simulation_time=T, 
				    inputs={'Pop1/Op_e/inp': inp},
				    outputs={'r': 'Pop1/Op_e/r'})

In the above code snippet, we performed a numerical simulation with a step-size of `dt` for a simulation time of `T`.

We defined an external input `inp`, which we passed to the `run()` function via the `inputs` argument.
This argument takes a dictionary as value and the keys of the dictionary need to be pointers to the variables in the network that input should be provided to.
The values of the inputs dictionary take the form of `numpy.ndarray`. They need to be defined for each simulation step (1. dimension of the array).

Furthermore, we defined an output via the `outputs` argument, which takes a dictionary as argument as well. 
The keys of this dictionary represent the names under which the output variables should be returned from the `run` method.
The values of this dictionary contain pointers to network variables that should be recorded over time.

Importantly, in hierarchical networks with multiple sub-circuits of nodes, inputs and outputs can also point to groups of nodes.
For example, if we would have a network of of 2 sub-circuits (`Circ1` and `Circ2`), each containing 2 populations (`Pop1` and `Pop2`),
we could either define inputs and outputs for specific node variables as shown above::

	results = compute_graph.run(simulation_time=T, 
				    inputs={'Circ1/Pop1/Op_e/inp': inp},
				    outputs={'r': 'Circ2/Pop2/Op_e/r'})

Or for groups of nodes using `all` as a pointer to all nodes of a certain hierarchical level::

	results = compute_graph.run(simulation_time=T,
			    inputs={'all/Pop1/Op_e/inp': inp}
			    outputs={'r': 'Circ2/all/Op_e/r'})

In the example above, we defined inputs for `Pop1` of both sub-circuits and recorded the firing rate `r` of both populations in `Circ2`.

To save memory, it can be convenient to reduce the sampling rate of the output storage. This can be done as follows::

	results = compute_graph.run(simulation_time=T, 
				    inputs={'Pop1/Op_e/inp': inp},
				    outputs={'r': 'Pop1/Op_e/r'},
				    sampling_step_size=1e-2)

Where `sampling_step_size` indicates the step size between sampling points. This value should always be larger than `dt`.


Simulation Time Optimization
----------------------------

In this example, we go through various arguments to the `CircuitIR.run()` and `CircuitIR.compile()` methods as well as configuration changes of external tools that may change the simulation performance of PyRates. First of all, the size of the network and the number of simulation steps are the major determinants of simulation durations in PyRates.

The effect of the network size on the simulation performance can be counteracted to some degree by network representation optimizations. A particularly important optimization is the vectorization of network nodes with identical mathematical structure. This optimization can be turned on and off via the `vectorization` argument of the `compile` method::

   
	compute_graph = circuit.compile(dt=dt, vectorization=True)

This vectorization allows for effective parallelization of network operations. Different backends of PyRates have different parallelization capacities. While the NumPy backend only provides CPU-based parallelization, tensorflow adds GPU-based parallelization to the mix. To switch between the two, use the `backend` argument of the `compile` method::

	compute_graph = circuit.compile(backend='tensorflow', dt=dt)

When using the NumPy backennd, parallelization and run times in general can be further optimized and controlled via the Numba package. This package provides various function decorators that can be passed to the `CircuitIR.run` method together with their arguments. Simple just-in-time compilation of the PyRates operations can be enabled the following way::

	from numba import jit

	results = compute_graph.run(simulation_time=T, 
				    inputs={'Pop1/Op_e/inp': inp},
				    outputs={'r': 'Pop1/Op_e/r'},
				    decorator=jit, nopython=True)

In this example, `nopython` is an option of the decorator `jit`, which can be enabled or disabled. Another such option is `parallel`, which tells the decorator whether it should attempt to parallelize the decorated functions or not::

	from numba import njit

	results = compute_graph.run(simulation_time=T, 
				    inputs={'Pop1/Op_e/inp': inp},
				    outputs={'r': 'Pop1/Op_e/r'},
				    decorator=njit, parallel=True)

For a more comprehensive documentation of the numba function decorators, see https://numba.pydata.org/.

When using the tensorflow backend, decorators are used in the backend automatically, so these arguments are disabled. However, some control over the performance of tensorflow is possible by changing the parallelism within and between operations::

	from tensorflow import config

	config.threading.set_inter_op_parallelism_threads(4)
	config.threading.set_intra_op_parallelism_threads(2)

Optimal settings depend strongly on the running system. For further tensorflow internal optimization options see https://www.tensorflow.org/guide/performance/overview.

Finally, densely connected networks can benefit from their edge operations being realized via a vector product between an edge weight matrix and the source variables.
This is done automatically in PyRates. However, it is only allowed for if the edge weight matrix is dense enough.
The criterion for how much sparseness is allowed in edge weight matrices can be controlled via the `matrix_sparseness` keyword argument
of the `CircuitIR.compile` method::

    compute_graph = circuit.compile(backend='tensorflow', dt=dt, matrix_sparseness=0.8)

Note, however, that this only affects edges with zero delays.
