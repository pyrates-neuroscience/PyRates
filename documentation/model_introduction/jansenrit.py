"""

The Jansen-Rit Neural Mass Model
================================

Here, we will introduce the Jansen-Rit model, a neural mass model of the dynamic interactions between 3 populations:

    - pyramidal cells (PCs)
    - excitatory interneurons (EINs)
    - inhibitory interneurons (IINs)

Originally, the model has been developed to describe the waxing-and-waning of EEG activity in the alpha frequency range
(8-12 Hz) in the visual cortex [1]_. In the past years, however, it has been used as a generic model to describe the
macroscopic electrophysiological activity within a cortical column [2]_. A graphic representation of such a model,
placed inside a brain network, can be found in the figure below.

.. _fig1:

.. figure:: ../../../pyrates_interfaces/images/pyrates_model.png
   :width: 700

   Figure 1

The structure of the full Jansen-Rit model is depicted in :ref:`fig1` B. As visualized for the pyramidal cell
population in :ref:`fig1`C, the model can be decomposed into a number of generic mathematical operators that can be
used to build the dynamic equations for the Jansen-Rit model. Essentially, the membrane potential deflections
that are caused at the somata of a population by synaptic input, are modeled by a convolution operation with an alpha
kernel. This choice has been shown to reflect the dynamic process of polarization propagation from the synapse via the
dendritic tree to the soma [3]_. The convolution operation can be expressed via a second-order differential equation:

.. math::
        \\dot V &= I, \n
        \\dot I &= \\frac{H}{\\tau} m_{in} - \\frac{2 I}{\\tau} - \\frac{V}{\\tau^2},

where :math:`V` represents the average post-synaptic potential and :math:`H` and :math:`\tau` are the efficacy and
the time-scale of the synapse, respectively. As a second operator, the translation of the average membrane potential
deflection at the soma to the average firing of the population is given by a sigmoidal function:

.. math::
        m_{out} = S(V) = \\frac{m_{max}}{1 + e^{(r (V_{thr} - V))}}.

In this equation, :math:`m_{out}` and :math:`V` represent the average firing rate and membrane potential, respectively,
while :math:`m_{max}`, :math:`r` and :math:`V_{thr}` are constants defining the maximum firing rate, firing threshold
variance and average firing threshold within the modeled population, respectively.

By using the linearity of the convolution operation, the dynamic interactions between PCs, EINs and IINs can be
expressed via 6 coupled ordinary differential equations that are composed of the two operators defined above:

.. math::

        \\dot V_{pce} &= I_{pce}, \n
        \\dot I_{pce} &= \\frac{H_e}{\\tau_e} c_4 S(c_3 V_{in}) - \\frac{2 I_{pce}}{\\tau_e} - \\frac{V_{pce}}{\\tau_e^2}, \n
        \\dot V_{pci} &= I_{pci}, \n
        \\dot I_{pci} &= \\frac{H_i}{\\tau_i} c_2 S(c_1 V_{in}) - \\frac{2 I_{pci}}{\\tau_i} - \\frac{V_{pci}}{\\tau_i^2}, \n
        \\dot V_{in} &= I_{in}, \n
        \\dot I_{in} &= \\frac{H_e}{\\tau_e} S(V_{pce} - V_{pci}) - \\frac{2 I_{in}}{\\tau_e} - \\frac{V_{in}}{\\tau_e^2},

where :math:`V_{pce}`, :math:`V_{pci}`, :math:`V_{in}` are used to represent the average membrane potential deflection
caused by the excitatory synapses at the PC population, the inhibitory synapses at the PC population, and the excitatory
synapses at both interneuron populations, respectively.

Below, we will demonstrate how to load this a model into pyrates and perform numerical simulations with it.

References
^^^^^^^^^^

.. [1] B.H. Jansen & V.G. Rit (1995) *Electroencephalogram and visual evoked potential generation in a mathematical
       model of coupled cortical columns.* Biological Cybernetics, 73(4): 357-366.

.. [2] A. Spiegler, S.J. Kiebel, F.M. Atay, T.R. Knösche (2010) *Bifurcation analysis of neural mass models: Impact of
       extrinsic inputs and dendritic time constants.* NeuroImage, 52(3): 1041-1058,
       https://doi.org/10.1016/j.neuroimage.2009.12.081.

.. [3] P.A. Robinson, C.J. Rennie, J.J. Wright (1997) *Propagation and stability of waves of electrical activity in the
       cerebral cortex.* Physical Review E, 56(826), https://doi.org/10.1103/PhysRevE.56.826.

"""

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
# In the second step, we load a model template for the Jansen-Rit model that comes with PyRates via the
# :code:`from_yaml()` method of the :code:`CircuitTemplate`. This method returns a :code:`CircuitTemplate` instance
# which provides the method :code:`apply()` for turning it into a graph-based representation, i.e. a
# :code:`pyrates.ir.CircuitIR` instance. Have a look at the yaml definition of the model that can be found at the path
# used for the :code:`from_yaml()` method. You will see that all variables and parameters are already defined there.
# These are the basic steps you perform, if you want to load a model that is
# defined inside a yaml file. To check out the different model templates provided by PyRates, have a look at
# the :code:`PyRates.model_templates` module.

jrc = CircuitTemplate.from_yaml("model_templates.jansen_rit.simple_jansenrit.JRC_simple").apply()

# %%
# Step 3: Loading the model into the backend
# ------------------------------------------
#
# In this example, we directly load the :code:`CircuitIR` instance into the backend via the  :code:`compile()` method
# without any further changes to the graph. This way, a :code:`pyrates.backend.NumpyBackend` instance is created.
# After this step, structural modifications of the network are not possible anymore. Here, we choose scipy as a solver
# for our differential equation system. The default is the forward Euler method that is implemented in PyRates itself.
# Generally, the scipy solver is both more accurate and faster and thus the recommended solver in PyRates.

jrc_compiled = jrc._compile(backend='numpy', step_size=1e-4, solver='scipy')

# %%
# Step 4: Numerical simulation of a the model behavior in time
# ------------------------------------------------------------
#
# After loading the model into the backend, numerical simulations can be performed via the :code:`run()` method.
# Calling this function will solve the initial value problem of the above defined differential equations for a time
# interval from 0 to the given simulation time.
# This solution will be calculated numerically by a differential equation solver in the backend, starting with a defined
# step-size.

results = jrc_compiled.run(simulation_time=2.0,
                           step_size=1e-4,
                           sampling_step_size=1e-3,
                           outputs={'V_pce': 'JRC/JRC_op/PSP_pc_e',
                                    'V_pci': 'JRC/JRC_op/PSP_pc_i'})

# %%
# Step 5: Visualization of the solution
# -------------------------------------
#
# The output of the :code:`run()` method is a :code:`pandas.Dataframe`, which comes with a :code:`plot()` method for
# plotting the timeseries it contains.
# This timeseries represents the numerical solution of the initial value problem solved in step 4 with respect to the
# state variables :math:`V_{pce}` and :math:`V_{pci}` of the model.

results.plot()

# %%
# To visualize the average membrane potential at the PC somata, simply plot the difference between :math:`V_{pce}` and
# :math:`V_{pci}`:

v_pc = results['V_pce'] - results['V_pci']
v_pc.plot()
from matplotlib.pyplot import show
show()