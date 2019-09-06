
Minimal Example
===============

In this minimal example, we load a model from the `model_templates` module, simulate its behavior and plot the resulting time series.
The model represents the macroscopic dynamics of a population of quadratic integrate-and-fire neurons and we record the average firing rate of the population:: 

	from pyrates.frontend import CircuitTemplate
	
	circuit = CircuitTemplate.from_yaml("model_templates.montbrio.simple_montbrio.Net1").apply()
	compute_graph = circuit.compile(backend='numpy', dt=dt) 
	results = compute_graph.run(simulation_time=10.0, outputs={'r': 'Pop1/Op_e/r'})
	results.plot()

In the first step, we loaded one of the model templates that come with PyRates via the `from_yaml()` method.
This method returns a `CircuitTemplate` instance which provides the method `apply()` for turning it into a graph-based representation, i.e. a `CircuitIR` instance. 
In this example, we directly transform the `CircuitIR` instance into a `ComputeGraph` instance via the `compile()` method without any further changes to the graph.
This way, our network is loaded into the backend. After this step, structural modifications of the network are not possible anymore.
However, from this point on, numerical simulations can be performed via the `run()` method. 
The output of the `run()` method is a `pandas.Dataframe`, which comes with a `plot()` method for plotting the timeseries it contains.
