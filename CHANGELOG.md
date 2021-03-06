# Changelog

## 0.9

### 0.9.2

- Documentation updates:
    - all Jansen-Rit model introductions where changed to track the excitatory and inhibitory post-synaptic potentials of the pyramidal cell population as
    output variables. Their difference provides the average membrane potential of the pyramidal cells.
    - Changed documentation jupyter notebooks etc. to account for Jansen-Rit model definition change (see below).
    - adjusted `qif_fold.py` to delete all temporary files created by auto-07p
- model templates updates:
    - added a 3 population model to the qif model templates in `simple_montbrio.yaml`
    - added qif population template with mono-exponential synaptic depression to `simple_montbrio.yaml`
    - added a new model template to `simple_montbrio.yaml` which provides a QIF population with mono-exponential spike-frequency adaptation
    - added bi-exponential short-term adaptation descriptions to QIF models in `simple_montbrio.yaml`
    - small change to the Jansen-Rit model definition: I removed the observer operator. To investigate the PC membrane potential, please record both PSP 
    variables at the PC population and plot their sum. This has been changed accordingly in all corresponding examples.
- PyAuto related updates:
    - altered the `pyrates.utility.pyauto.PyAuto.to_file` method. Additional keyword arguments that are provided by the user are now stored in a dictionary
    under `additional_attributes`. Loading a pyauto instance via `from_file` will thus create an attribute `additional_attributes` on the instances, which
    will contain all the keyword arguments as a dictionary.
    - debugged the `pyrates.utility.pyauto.get_from_solutions` method. Previously, providing more than one attribute key resulted in the method using an
    erroneous list comprehension style. This was fixed now. Providing multiple keys now results in the method returning a list of lists.
    - changed the way automatic re-runs of starting points computed by auto are detected by `pyrates.utility.pyauto.PyAuto`
    - fixed problem with extracting a solution from auto via the method `pyrates.utility.pyauto.PyAuto.get_solution()`. Apparently, sometimes the function
    call `solution_branch(solution_key)` does not work and throws an attribute error. I implemented a work around for this inconsistency in the Python 
    interface for auto-07p.
    - changed `pyrates.utility.pyauto.continue_period_doubling_bf` to return a list that contains the names of all period doubling continuations performed
    with the pyauto instance that is returned as a second return value
    - now catching an error in the plotting-related method `pyrates.utility.pyauto.PyAuto._get_line_collection`, if the `x` argument is a vector of length
    1
    - debugged `pyrates.utility.pyauto.PyAuto.get_point_idx()`. Sometimes, when auto-07p failed to locate the new fixed point of a steady-state solution,
    it retries the previous step. PyAuto could not recognize the auto-07p diagnostic output for such cases. Now it can.
    - improved period doubling continuation in `pyrates.utility.pyauto.py`. Only solution branches with new PD bifurcations are saved for plotting etc.
    - adjusted `pyrates.utility.pyauto.PyAuto.plot_continuation` method such that it can be used to plot continuations of the time parameter "PAR(14)"
    - adjusted `pyrates.utility.pyauto.PyAuto.plot_trajectory` to be able to plot phase space trajectories of explicit time continuations (continuations
    in "PAR(14)")
    - adjusted the return values of the `pyrates.utility.pyauto.fractal_dimension` method for its extreme cases. If the sum of the lyapunov spectrum is
    positive, return the number of lyapunov exponents. If the largest lyapunov exponent is smaller or equal to zero, use the normal formula.
    - added a `cutoff` argument to the `pyrates.utility.pyauto.PyAuto.plot_trajectory` method that allows to cut off initial transients within the time
    window from `t=0` until `t=cutoff`.
    - implemented speed-up of `pyrates.utility.pyauto.PyAuto.get_eigenvalues()` method and fixed two bugs with the method that (1) led to an empty list
    being returned, and (2) caused the method to fail when applied to a steady-state solution
    - improved continuation of period doubling cascades via `pyrates.utility.pyauto.continue_period_doubling_bf()`: It recognizes now which branches it
    had already switched to at period doubling bifurcations. Reduces the number of overall continuations
    - added the possibility to pass the installation directory of auto-07p to `pyrates.utility.pyauto.PyAuto`, 
    `pyrates.utility.pyauto.PyAuto.from_file` and `pyrates.ir.circuit.CircuitIR.to_pyauto()`. This makes it easier to install auto-07p, since the users do
    not have to manupilate system path variables themselfes anymore
    - debugged counting of already calculated parameter continuations in `pyrates.utility.pyauto.PyAuto`
    - adjusted the `pyrates.ir.circuit.CircuitIR.clear()` method together with the `pyrates.backend.fortran_backend.FortranBackend.clear()` method to 
    remove all temporary files created by us or auto-07p during the model compilation and execution.
- grid-search updates:
    - added a warning to the `pyrates.utility.grid_search.grid_search()` function if a certain parameter is not found in the model
    - improved interface between `pyrates.utility.grid_search.grid_search()` function and `pyrates.utility.grid_search.ClusterGridsearch` class
    - added a keyword argument `clear` to `grid_search` that prevents removal of temporary files if set to `False`
- visualization updates:
    - improved the interactive 2D plot in `pyrates.utility.visualization.py`
    - Debugging of `pyrates.utility.visualization.Interactive2DParamPlot`: retrieving the column index of each column name now handles multi-column
    Dataframes correctly.
- backend updates:
    - replaced "is" comparisons with "==" comparisons where appropriate
- evolutionary optimization updates:
    - changed the way model ids are sampled in `pyrates.utility.genetic_algorithm.DifferentialEvolutionAlgorithm`. With the old method, multiple workers
    sometimes generated models with equal IDs, leading to errors.
    - added an argument to `pyrates.utility.genetic_algorithm.DifferentialEvolutionAlgorithm.run()` that allows to suppress runtime warnings.
- intermediate representation updates:
    - fixed a bug in `pyrates.ir.circuit.CircuitIR._add_edge_buffer()` that led to a wrong association between node indices and node variables in cases
    where multiple delayed edges with different delay profiles had to be handled. This mostly affected grid-searches over delay distribution parameters.
    - passed the `verbose` argument of `pyrates.ir.circuit.CircuitIR.run()` to the backend run function. Now all printed output of PyRates can be muted.
    

### 0.9.1

- Updated documentation
- Removed conversion function register, because the functions were not used and made the code unnecessarily complicated
    - might be replaced by a graph-based conversion path-finder in the future, if necessary
- Extended support for loading circuits from and saving to files
    - supported formats: `yaml`, `pickle`
    - supported classes: templates
- Removed all imports in `pyrates.utility.__init__.py` for increased stability. 
  Previously, importing something from `pyrates.utility`, would have required a user to install optional packages that 
  might not have been needed. Now all utility functions need to be imported from sub-files in the `pyrates.utility` 
  module instead of directly from the module.
- Added optional install collection `tests` that includes all packages necessary to run the tests. 
  Also restricted the travis CI build to use only the tests installation instead of the full installation.
- Added feature to pass a dictionary to `CircuitTemplate.apply()` in order to adapt values of variables on the fly. This 
  behaviour was already supported by all other parts of the hierarchy, only circuits missed out until now.  

### 0.9.0

- Added experimental support for multiple source variables per edge
  - edges can either have multiple input variable from the same input node, or
  - they can have additional ("modulating") input from any node in the network
- Added experimental support for Fortran code creation backend
- Edge delays can now be transformed into delay distributions via convoluted Gamma-Kernels based on differential equation using a mean and spread parameter for the delay
- various performance improvements


## 0.8

### 0.8.2 Included bug fixes from jajcayn:

- Allow to initialise CircuitTemplate with instances of `EdgeTemplate` instead of a template path, previous behaviour is unaffected. 
- Fix writing graph to the file by passing `_format` along until the end

### 0.8.1 Improved cluster distribution and bug fixes

- updated tensorflow dependency to >=2.0, fixes some dependency problems
- Improved cluster distribution system, available under `pyrates.utility.grid_search`
- New feature: model optimization with genetic algorithms, available under `pyrates.utility.genetic_algorithm`
- Miscellaneous bug fixes

### 0.8.0 

- removed version ID numbers of operator/node instances in the intermediate representation. I.e. a node label `mynode` 
  was previously renamed to `mynode.0` and will now keep it's original label.
- moved all functionality of ComputeGraph into CircuitIR, which is now the main interface for the backend. 
  - `CircuitIR` now has a `.compile` method that performs all vectorization and transformation into the computable 
    backend form.
- vectorization will transform all nodes into instances of `VectorizedNodeIR` that have labels like `vector_nodeX` with 
  X being a integer index. The map between old nodes and vectorized nodes with respective index is saved in the 
  `label_map` dictionary attribute of the `CircuitIR`
- When adding input or sampling output of a network with multiple stacked levels of circuits, you can now use `all` to 
  get all nodes within that particular level. For example `mysubcircuit1/all/mynode` will get all nodes with label 
  `mynode` that are in one level of sub-circuits below `mysubcircuit`.
- Tensorflow support now relies on the current 2.0 release candidate `tensorflow-2.0-rc`
- Added optional install requirements via `extras_require` in setup.py 
