Changelog
=========

1.0
---

1.0.7
-----

- updated pytests to account for recent updates to sympy and other python packages
- fixed a bug in the documentation use example `continuation.py`
- improved support for complex variables in Fortran backend. The global variable `I = sqrt(-1.0)` is now defined in each fortran script. Also, initial conditions for complex variables are properly set via the parentheses notation, e.g. `v = (1.0, 0.5)`.
- fixed a bug with the recognition of complex-valued variables in the `OperatorTemplate` class
- resolved bug with edge template vectorization where edge source and target indices were not applied correctly

1.0.6
-----

- fixed a bug that caused vectorization to fail if the same operator was used multiple times on a single node
- fixed a bug that caused an error in the generation of unique variable names on nodes with more than 10 operators defined on them
- fixed a bug that caused `CircuitTemplate.clear()` calls to not clear all attributes on a `CircuitTemplate` instace, causing issues with multiple calls of `CircuitTemplate.get_run_func`
- updated the fortran backend to work with the recent changes to the `numpy.f2py` module for generating a modulate that can be imported into python from a fortran file
- fixed a bug in the `ComputeGraph` class of computegraph.py that caused function names to not be updated properly for backend-specific function definitions

1.0.5
-----

- adjusted the call of the max/min functions: Use `maxi` and `mini` in the equations. Both functions take two input arguments, and return the larger/smaller one, respectively
- updated the PyRates reference in the readme and on the documentation website (using the PLOS CB paper now instead of the arxiv preprint)
- removed a bug where differential equations with a constant right-hand side were not properly handled by the automated compute graph optimization
- resolved an issue with the fortran backend where complex data types were not properly processed during the code generation

1.0.4
-----

- updated readthedocs configuration file
- added keyword argument `adaptive` to the `CircuitTemplate.get_run_func` method, which allows to indicate whether the generated equation file is expected to be called with an adaptive step-size solver (`adaptive=True`) or not
- reduced computational overhead for the creation and simulation of delayed differential equation systems
- removed a bug where edge attribute dictionaries were changed by mistake during the `CircuitIR` instantiation
- improved working directory management in the backend

1.0.3
-----

- simplified automated generation of unique variable names (recursive calls etc. were replaced with look-up tables)
- improved variable passing between different operators within a node. Less additional variables are now created, thus reducing the memory load

1.0.2
-----

- fixed bug in fortran backend where the NPAR parameter for Auto-07p files was not properly set
- improved code readability in fortran backend
- moved selection of output variables from the results of a numerical simulation from the backend to the computegraph, thus reducing the amount of variables that had to be passed between the different classes
- after each simulation, the value of all state variables in the compute graph is updated to the value at the final simulation step
- added functionalities to the `CircuitTemplate` that allow to remember the state of all network variables from a previous simulation, even if a new backend is chosen for function generation or more simulations

1.0.1
-----

- added a background input parameter to the izhikevich population template
- updated the documentation example for parameter sweeps to account for recent changes in the keyword arguments to the `grid_search` function
- changed keyword argument `vectorization` of the function `grid_search` to `vectorize`, to be consistent with the naming of the same argument in `CircuitTemplate.run`
- updated the `CircuitTemplate.add_edges_from_matrix` method to allow for edges that connect separate network nodes

1.0.0
-----

This official release is a combination of all the bug fixes and improvements in the
pre 1.0 versions.

Minor improvements since 0.17.4:

- removed typos in documentation
- improved layout of the online documentation
- updated documentation to account for latest changes
- removed bug where a delayed differential equation model would lead to a KeyError when trying to generate the run function

0.17
----

0.17.4
~~~~~~

- added sign function to the backend that returns the sign of its input (1 or -1)
- improved readthedocs documentation (removed bug with display of math, added new use example for edge templates)
- added a safe guard for defining edge templates: An error is raised now when edge template input variables have the
  same name as their source variable.

0.17.3
~~~~~~

- minor debugging of the model introduction use examples
- adjustments of the template cheat sheet `template_specification.rst`
- debugged issue in base backend, where file names specified by the users that contained a file ending wre not handled properly
- debugged issue with fortran backend where file names that contained a directory path were not handled properly for module imports
- debugged issue with fortran backend where adjustments of the default auto meta parameters were not applied correctly

0.17.2
~~~~~~

- the state variable indices and parameter names returned as the fourth and third return values of `CircuitTemplate.get_run_func`, respectively, now use the frontend variable names instead of the backend variable names
- implemented a method `CircuitIR.get_frontend_varname` that returns the frontend variable name given a backend variable name

0.17.1
~~~~~~

- changed the theme of the readthedocs documentation website
- added documentation for all supported backend functions
- added documentation for dependencies and requirements
- added documentation for YAML template structure to the documentation website
- added documentation for mathematical syntax
- added the changelog to the documentation website

0.17.0
~~~~~~

-  added ``__getitem__`` methods on all frontend template classes that
   allow for a less convoluted examination of the major properties of
   the template classes
-  added pytests that test these new features
-  users can now quickly access each node on ``CircuitTemplate``, each
   operator on ``NodeTemplate`` and ``EdgeTemplate``, and each variable
   on ``OperatorTemplate``

0.16
----

0.16.0
~~~~~~

-  added class for interactive grid search results visualization to
   utility
-  changed organization of the pandas DataFrames that ``grid-search``
   returns: Each different parameterization of the model appears only
   once in the ``param_grid.index`` and the ``results`` DataFrame uses a
   full hierarchical column organization.
-  The pandas DataFrame returned by ``CircuitTemplate.run`` uses a fully
   hierarchical column organization now: Every node hierarchy level is a
   separate level in the column index hierarchy.
-  minor docstring improvements
-  fixed bug in edge equation setup where a wrong index was provided to
   the target variable sometimes
-  fixed bug in variable updating that occurred for ``numpy.ndarray``
   variables where the ``shape`` attribute was an empty tuple
-  applied all changed to the gallery examples in the documentation

0.15
----

0.15.1
~~~~~~

-  added generic method for state variable indexing to circuit.py that
   is used for all edge-related indexing operations now (replacing
   multiple, slightly different implementations at various places in
   circuit.py)
-  added an alternative compute graph class that can be used to generate
   function files that do not perform in-place manipulations of the
   vectorfield ``dy`` but instead just create a new variable. This is
   relevant for gradient-based optimization.
-  improved the modularity of the ``ComputeGraph``
-  added a method ``add_import`` to the backend that allows adding
   import statements to the top of a function file
-  added a backend function ``concatenate`` that can be used in equation
   strings now in order to combine vectorized variables
-  removed a bug where calling ``clear_frontend_caches`` did not clear
   all IR caches properly

0.15.0
~~~~~~

-  added support for models with vectorized state-variables
-  improved performance of edge operations
-  more detailed output about returned function arguments when calling
   ``CircuitTemplate.get_run_func``
-  improved memory consumption during model initialization
-  complex-valued models use complex variable types for all variables
   and parameters now, to prevent type conversions
-  added a new method ``CircuitTemplate.get_var`` that allows users to
   access backend variables after calling
   ``CircuitTemplate.get_run_func``
-  added automated reduction of vectorized constants, if all constants
   are identical
-  added possibility to pass iterables to
   ``CircuitTemplate.update_var``, thus allowing to update vectorized
   variables in one go
-  updated ``CircuitTemplate.add_edges_from_matrix`` such that only
   edges with non-zero weights are added to the ``CircuitTemplate``
   instance

0.14
----

0.14.3
~~~~~~

-  run-function generating method of ComputeGraph now returns the keys
   of the function arguments together with the arguments
-  implemented a method in CircuitTemplate that allows to get the
   indices of state variables within the system state vector

0.14.2
~~~~~~

-  updated changelog

0.14.1
~~~~~~

-  added different versions of the Izhikevich mean-field model (the
   dimensionless model, the biophysical model with distributed
   background currents, and the biophysical model with distributed spike
   thresholds)
-  improved documentation gallery examples (debugged equations, added
   images, added Izhikevich model references)

0.14.0
~~~~~~

-  added Heun’s method as a new differential equation solver method
-  Heun’s method was integrated with all backends
-  a test was added that ensures correct functionality of Heun’s method
-  the usage of the method is demonstrated in the simulations gallery
   example
-  added hyperlinks to websites explaining the different numerical
   solvers in the gallery example
-  improved the backend implementation of choosing between different
   solvers (less code overlap between backends now)

0.13
----

0.13.0
~~~~~~

-  added support for delayed differential equation (DDE) systems
-  a function ``past(y, tau)`` is now available for any backend that
   allows to evaluate a state variable ``y`` at time ``t-tau``
-  edges with discrete delays that are to be used in combination with an
   adaptive step-size solver are translated into ``past`` calls
-  a gallery example was added that demonstrates how to interface the
   Python package ``ddeint`` via a DDE system generated by PyRates
-  the Julia backend received support for performing DDE simulations
   from within PyRates via its interface to ``DifferentialEquations.jl``

0.12
----

0.12.2
~~~~~~

-  debugged latex equation error in Izhikevich model gallery example
-  bugfix in julia backend where a wrong file ending was provided
-  added new pytests for the izhikevich model, the python model
   definition interface and the CircuitIR translation
-  updated the readme
-  added a new QIF model template that includes conductance-based
   synapses

0.12.1
~~~~~~

-  added gallery example for the izhikevich mean-field model
-  updated readme
-  updated changelog
-  updated default parameterization of the izhikevich model

0.12.0
~~~~~~

-  added a matlab backend (mainly for code generation, since simulations
   are very slow due to array conversion between numpy and matlab)
-  added a mean-field model of the Izhikevich neuron
-  small bug fixes

   -  removed an issue of the fortran interface to Auto-07p that led to
      wrong function argument indices being generated
   -  removed an issue with synaptic weights of -1 being converted to 1
   -  removed a compatibility issue between old and new versions of the
      ‘to_yaml’ methods

-  added the natural logarithm ‘log’ as backend function

0.11
----

0.11.1
~~~~~~

-  removed bug where vectorized circuits with multiple edges to the same
   target wre not resolved correctly
-  removed bug where creating deepcopies of a ``CircuitTemplate`` raised
   an error for scalar-valued models
-  added a new gallery example demonstrating different ways of adding
   delays to models
-  added a new gallery example demonstrating the different options to
   optimize run times of numerical simulations

0.11.0
~~~~~~

-  added support for complex-valued systems
-  added model templates for the kuramoto order parameter and the theta
   neuron model
-  added model templates for the van der pol oscillator and the
   stuart-landau oscillator
-  added support for Python 3.9
-  added new example galleries
-  extended pytest library
-  added the ``CircuitTemplate.to_yaml`` method that allows to save a
   given ``CircuiTemplate`` instance to a YAML definition file
-  added the ``CircuitTemplate.add_edges_from_matrix`` method that
   allows to connect nodes in a ``CircuiTemplate`` instance via
   connectivity matrices
-  deleted old, deprecated code fragments
-  removed the dependecy on pyparsing

0.10
----

0.10.1
~~~~~~

-  updates to changelog and setup.py

0.10.0
~~~~~~

-  reworked features:

   -  Restructured backend

      -  new backends (torch, Julia)
      -  sympy-based equation parsing
      -  improved compute graph
      -  improved generation of run functions from compute graphs

   -  Improved frontend

      -  easier imports
      -  additional convenience functions for simulations
      -  less steps from model definition to simulation
      -  reduced syntax for model definitions

   -  Removed utility package

      -  utility packages for parameter optimization, signal analysis
         and visualization have been removed from the pyrates main
         package
      -  most utility functionalities have been moved to separate
         repositories of the pyrates-neuroscience organization
      -  less package requirements

   -  new model templates

      -  improved structure of the model templates
      -  New model templates and documentation examples
      -  new example galleries and jupyter notebooks with hands-on use
         examples

0.9
---

0.9.6
~~~~~

-  Reworked features:

   -  ``CircuitIR._add_edge_buffer()`` was re-worked, such that the
      algorithm that translates gamma-kernel convolutions for edges into
      ODE systems is more transparent and computationally less expensive
   -  additionally improved the source code documentation of
      ``CircuitIR._add_edge_buffer()``
   -  removed unnecessary copying/indexing operations of original edge
      source variable

0.9.5
~~~~~

-  Bug fixes:

   -  fixed a bug in ``CircuitIR._add_edge_buffer()`` that caused a
      mix-up between edges when data was transferred from the originial
      output into the buffer variables.

-  Performance improvements:

   -  zero-weight edges are now removed much earlier in the compilation
      process, thus reducing compilation time.

0.9.4
~~~~~

-  Bug fixes:

   -  fixed a bug in ``CircuitIR._add_edge_buffer()`` that caused a
      mix-up between edges when some outputs of a node had delays while
      others had not.

-  Usability improvements:

   -  changed ``CircuitIR.vectorize_edges()`` in circuit.py such that
      zero-weight edges are removed during the vectorization, even if
      they have a delay defined on them (previously, defining a delay on
      a zero-weight edge kept that edge in the graph).

0.9.3
~~~~~

-  Documentation changes:

   -  corrected mistake in the documentation of
      ``pyrates.ir.circuit.CircuitIR.add_edge_buffer()``, where
      arguments that refer to the source variable of an edge, where
      erroneously described as target variable information.

-  Bug fixes:

   -  fixed bug in ``pyrates.ir.circuit.CircuitIR.add_edge_buffer()``
      where the conversion from discrete delays to gamma-kernel
      convolutions led to a mix-up between different edges in some
      special cases.
   -  fixed bug in
      ``pyrates.utility.pyauto.PyAuto._start_from_solution()`` where
      certain special solution branches from Auto-07p could not be
      properly handled

-  Usability improvements:

   -  changed ``pyrates.utility.grid_search.adapt_circuit()`` such that
      node properties are always deep-copied before they are changed.
      This allows users to change the values of parameters on specific
      node operators, even though that exact same operator has been used
      to define multiple nodes in the network. Previously, changing the
      value of the parameter on one node led to changes on all other
      nodes as well.
   -  improved stability and usability of
      ``pyrates.utility.visualization.Interactive2DParamPlot``. A title
      for the 2D plot can now be passed, a colorbar is added, and the
      location of the axis ticks of the 2D plot was improved

0.9.2
~~~~~

-  Documentation updates:

   -  all Jansen-Rit model introductions where changed to track the
      excitatory and inhibitory post-synaptic potentials of the
      pyramidal cell population as output variables. Their difference
      provides the average membrane potential of the pyramidal cells.
   -  Changed documentation jupyter notebooks etc. to account for
      Jansen-Rit model definition change (see below).
   -  adjusted ``qif_fold.py`` to delete all temporary files created by
      auto-07p

-  model templates updates:

   -  added a 3 population model to the qif model templates in
      ``simple_montbrio.yaml``
   -  added qif population template with mono-exponential synaptic
      depression to ``simple_montbrio.yaml``
   -  added a new model template to ``simple_montbrio.yaml`` which
      provides a QIF population with mono-exponential spike-frequency
      adaptation
   -  added bi-exponential short-term adaptation descriptions to QIF
      models in ``simple_montbrio.yaml``
   -  small change to the Jansen-Rit model definition: I removed the
      observer operator. To investigate the PC membrane potential,
      please record both PSP variables at the PC population and plot
      their sum. This has been changed accordingly in all corresponding
      examples.

-  PyAuto related updates:

   -  altered the ``pyrates.utility.pyauto.PyAuto.to_file`` method.
      Additional keyword arguments that are provided by the user are now
      stored in a dictionary under ``additional_attributes``. Loading a
      pyauto instance via ``from_file`` will thus create an attribute
      ``additional_attributes`` on the instances, which will contain all
      the keyword arguments as a dictionary.
   -  debugged the ``pyrates.utility.pyauto.get_from_solutions`` method.
      Previously, providing more than one attribute key resulted in the
      method using an erroneous list comprehension style. This was fixed
      now. Providing multiple keys now results in the method returning a
      list of lists.
   -  changed the way automatic re-runs of starting points computed by
      auto are detected by ``pyrates.utility.pyauto.PyAuto``
   -  fixed problem with extracting a solution from auto via the method
      ``pyrates.utility.pyauto.PyAuto.get_solution()``. Apparently,
      sometimes the function call ``solution_branch(solution_key)`` does
      not work and throws an attribute error. I implemented a work
      around for this inconsistency in the Python interface for
      auto-07p.
   -  changed ``pyrates.utility.pyauto.continue_period_doubling_bf`` to
      return a list that contains the names of all period doubling
      continuations performed with the pyauto instance that is returned
      as a second return value
   -  now catching an error in the plotting-related method
      ``pyrates.utility.pyauto.PyAuto._get_line_collection``, if the
      ``x`` argument is a vector of length 1
   -  debugged ``pyrates.utility.pyauto.PyAuto.get_point_idx()``.
      Sometimes, when auto-07p failed to locate the new fixed point of a
      steady-state solution, it retries the previous step. PyAuto could
      not recognize the auto-07p diagnostic output for such cases. Now
      it can.
   -  improved period doubling continuation in
      ``pyrates.utility.pyauto.py``. Only solution branches with new PD
      bifurcations are saved for plotting etc.
   -  adjusted ``pyrates.utility.pyauto.PyAuto.plot_continuation``
      method such that it can be used to plot continuations of the time
      parameter “PAR(14)”
   -  adjusted ``pyrates.utility.pyauto.PyAuto.plot_trajectory`` to be
      able to plot phase space trajectories of explicit time
      continuations (continuations in “PAR(14)”)
   -  adjusted the return values of the
      ``pyrates.utility.pyauto.fractal_dimension`` method for its
      extreme cases. If the sum of the lyapunov spectrum is positive,
      return the number of lyapunov exponents. If the largest lyapunov
      exponent is smaller or equal to zero, use the normal formula.
   -  added a ``cutoff`` argument to the
      ``pyrates.utility.pyauto.PyAuto.plot_trajectory`` method that
      allows to cut off initial transients within the time window from
      ``t=0`` until ``t=cutoff``.
   -  implemented speed-up of
      ``pyrates.utility.pyauto.PyAuto.get_eigenvalues()`` method and
      fixed two bugs with the method that (1) led to an empty list being
      returned, and (2) caused the method to fail when applied to a
      steady-state solution
   -  improved continuation of period doubling cascades via
      ``pyrates.utility.pyauto.continue_period_doubling_bf()``: It
      recognizes now which branches it had already switched to at period
      doubling bifurcations. Reduces the number of overall continuations
   -  added the possibility to pass the installation directory of
      auto-07p to ``pyrates.utility.pyauto.PyAuto``,
      ``pyrates.utility.pyauto.PyAuto.from_file`` and
      ``pyrates.ir.circuit.CircuitIR.to_pyauto()``. This makes it easier
      to install auto-07p, since the users do not have to manupilate
      system path variables themselfes anymore
   -  debugged counting of already calculated parameter continuations in
      ``pyrates.utility.pyauto.PyAuto``
   -  adjusted the ``pyrates.ir.circuit.CircuitIR.clear()`` method
      together with the
      ``pyrates.backend.fortran_backend.FortranBackend.clear()`` method
      to remove all temporary files created by us or auto-07p during the
      model compilation and execution.

-  grid-search updates:

   -  added a warning to the
      ``pyrates.utility.grid_search.grid_search()`` function if a
      certain parameter is not found in the model
   -  improved interface between
      ``pyrates.utility.grid_search.grid_search()`` function and
      ``pyrates.utility.grid_search.ClusterGridsearch`` class
   -  added a keyword argument ``clear`` to ``grid_search`` that
      prevents removal of temporary files if set to ``False``

-  visualization updates:

   -  improved the interactive 2D plot in
      ``pyrates.utility.visualization.py``
   -  Debugging of
      ``pyrates.utility.visualization.Interactive2DParamPlot``:
      retrieving the column index of each column name now handles
      multi-column Dataframes correctly.

-  backend updates:

   -  replaced “is” comparisons with “==” comparisons where appropriate

-  evolutionary optimization updates:

   -  changed the way model ids are sampled in
      ``pyrates.utility.genetic_algorithm.DifferentialEvolutionAlgorithm``.
      With the old method, multiple workers sometimes generated models
      with equal IDs, leading to errors.
   -  added an argument to
      ``pyrates.utility.genetic_algorithm.DifferentialEvolutionAlgorithm.run()``
      that allows to suppress runtime warnings.

-  intermediate representation updates:

   -  fixed a bug in ``pyrates.ir.circuit.CircuitIR._add_edge_buffer()``
      that led to a wrong association between node indices and node
      variables in cases where multiple delayed edges with different
      delay profiles had to be handled. This mostly affected
      grid-searches over delay distribution parameters.
   -  passed the ``verbose`` argument of
      ``pyrates.ir.circuit.CircuitIR.run()`` to the backend run
      function. Now all printed output of PyRates can be muted.

0.9.1
~~~~~

-  Updated documentation
-  Removed conversion function register, because the functions were not
   used and made the code unnecessarily complicated

   -  might be replaced by a graph-based conversion path-finder in the
      future, if necessary

-  Extended support for loading circuits from and saving to files

   -  supported formats: ``yaml``, ``pickle``
   -  supported classes: templates

-  Removed all imports in ``pyrates.utility.__init__.py`` for increased
   stability. Previously, importing something from ``pyrates.utility``,
   would have required a user to install optional packages that might
   not have been needed. Now all utility functions need to be imported
   from sub-files in the ``pyrates.utility`` module instead of directly
   from the module.
-  Added optional install collection ``tests`` that includes all
   packages necessary to run the tests. Also restricted the travis CI
   build to use only the tests installation instead of the full
   installation.
-  Added feature to pass a dictionary to ``CircuitTemplate.apply()`` in
   order to adapt values of variables on the fly. This behaviour was
   already supported by all other parts of the hierarchy, only circuits
   missed out until now.

0.9.0
~~~~~

-  Added experimental support for multiple source variables per edge

   -  edges can either have multiple input variable from the same input
      node, or
   -  they can have additional (“modulating”) input from any node in the
      network

-  Added experimental support for Fortran code creation backend
-  Edge delays can now be transformed into delay distributions via
   convoluted Gamma-Kernels based on differential equation using a mean
   and spread parameter for the delay
-  various performance improvements

0.8
---

0.8.2 Included bug fixes from jajcayn:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Allow to initialise CircuitTemplate with instances of
   ``EdgeTemplate`` instead of a template path, previous behaviour is
   unaffected.
-  Fix writing graph to the file by passing ``_format`` along until the
   end

0.8.1 Improved cluster distribution and bug fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  updated tensorflow dependency to >=2.0, fixes some dependency
   problems
-  Improved cluster distribution system, available under
   ``pyrates.utility.grid_search``
-  New feature: model optimization with genetic algorithms, available
   under ``pyrates.utility.genetic_algorithm``
-  Miscellaneous bug fixes

0.8.0
~~~~~

-  removed version ID numbers of operator/node instances in the
   intermediate representation. I.e. a node label ``mynode`` was
   previously renamed to ``mynode.0`` and will now keep it’s original
   label.
-  moved all functionality of ComputeGraph into CircuitIR, which is now
   the main interface for the backend.

   -  ``CircuitIR`` now has a ``.compile`` method that performs all
      vectorization and transformation into the computable backend form.

-  vectorization will transform all nodes into instances of
   ``VectorizedNodeIR`` that have labels like ``vector_nodeX`` with X
   being a integer index. The map between old nodes and vectorized nodes
   with respective index is saved in the ``label_map`` dictionary
   attribute of the ``CircuitIR``
-  When adding input or sampling output of a network with multiple
   stacked levels of circuits, you can now use ``all`` to get all nodes
   within that particular level. For example
   ``mysubcircuit1/all/mynode`` will get all nodes with label ``mynode``
   that are in one level of sub-circuits below ``mysubcircuit``.
-  Tensorflow support now relies on the current 2.0 release candidate
   ``tensorflow-2.0-rc``
-  Added optional install requirements via ``extras_require`` in
   setup.py
