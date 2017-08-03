"""Running tests based on legacy code by Thomas Knösche. This represents the original interface.
author: Daniel Rose
"""

import numpy as np
from legacy.thomas_knoesche.init import init_jr

###############################################################################
# Running the test on the original interface
###############################################################################

###############################################################################
# Integration scheme / time axis
###############################################################################

start_time = 0  # all in seconds
end_time = 1
step_size = 5e-4
T_ = np.arange(start_time, end_time + step_size, step_size)

time_axis = {'start_time': start_time, 'end_time': end_time, 'step_size': step_size, 'T_': T_}
# this is a dictionary

###############################################################################
# lattice description
###############################################################################

# Summary
# mass index:   1     2     3
# positions:    1     1     1
# layer:        1     1     1
# neuron type:  PC    EIN   IIN

# ------------------------------------------------------------------------------
# list of positions of cortical profiles on cortical (or non-cortical) sheet
# ------------------------------------------------------------------------------

NP = 1  # number of positions (= local areas)
pos = np.zeros((NP, 2))  # (x,y) coordinates of positions
# (the actual positions are currently not relevant)
# -----
# Remark on use of positions (not implemented yet):
# The positions are used to compute distances and delays for local connectivity.
# Distances above 0.1m are not used for local connectivity. Therefore it is
# possible to code different detached areas (e.g. cortex and thalamus) by
# referencing them to origins at least 1 m apart. For example:
# pos = [0,0;0,1;1,0;1,1;1000,1000;1000,1001;1001,1000;1001,1001]^T
# the first 4 columns belong, e.g., to the cortex, the other 4 to the
# thalamus - local connectivity will be used only within each area.
# -----

# ------------------------------------------------------------------------------
# list of layers, each characterized by a label and a vertical (z) position
# ------------------------------------------------------------------------------

NL = 1  # number of cortical layers
lab_layers = ["I"]  # labels of layers, here just "I"
z_layers = np.zeros(NL)  # The z-positions of the layer centers are measured from the GM-WM interface.
# (here we use no layer specific modeling)

# -----
# Remark:
# Here, also layers from different cortical and subcortical areas can be
# listed, for example:
# {'I_BA6','II_BA6','III_BA6','IV_BA6','V_BA6','VI_BA6','I_BA4','II_BA4','I_BA4',II_BA4','IV_BA4','V_BA4','VI_BA4','Thal'};
# -----

# ------------------------------------------------------------------------------
# list of neuronal types
# ------------------------------------------------------------------------------

NT = 3  # number of different types of neurons
lab_types = ["PC", "EIN", "IIN"]  # we use the standard configuration with
# pyramidal cells (PC), excitatory
# and inhibitory interneurons (EIN, IIN))

# ------------------------------------------------------------------------------
# list of neural masses, each characterized by position, layer and type
# ------------------------------------------------------------------------------

N = 3  # number of masses
mass_pos = np.zeros(N, dtype=np.intp)  # positions of masses, entries refer to indices in pos
# here they all belong to the sole position "1"

mass_layers = np.zeros(N, dtype=np.intp)  # layers of masses, entries refer to indices in *_layers
# here they all belong to the sole layer "I"

mass_types = np.array([0, 1, 2], dtype=np.intp)  # types of masses, entries refer to indices in *_types

# ------------------------------------------------------------------------------
# arrange lattice
# ------------------------------------------------------------------------------

lattice = {'NP': NP, 'pos': pos, 'NL': NL, 'lab_layers': lab_layers, 'z_layers': z_layers, 'NT': NT,
           'lab_types': lab_types, 'N': N, 'mass_pos': mass_pos, 'mass_layers': mass_layers, 'mass_types': mass_types}

###############################################################################
# Parameterization of NT types of neural masses
###############################################################################

# ------------------------------------------------------------------------------
# membrane properties (for leak current)
# ------------------------------------------------------------------------------

tau_m = np.zeros(NT)  # passive membran time constants in s
tau_m[:] = 0.016  # assuming one passive membrane constant for all cell types

C_m = np.zeros(NT)  # membrane capacitance
C_m[:] = 1e-12  # soma diameter 19 um, capacitance 2uF/cm² (Wikipedia) results in ca. 1pF

E_L = np.zeros(NT)  # reversal potentials in V
E_L[:] = -0.075  # assumed to be the same for all neuronal types

# ------------------------------------------------------------------------------
# parameters of the sigmoid
# ------------------------------------------------------------------------------

sigm_max = np.zeros(NT)  # maximum firing rates
sigm_max[:] = 5

sigm_thresh = np.zeros(NT)  # lumped thresholds
sigm_thresh[:] = -0.069

sigm_spread = np.zeros(NT)  # spread of sigmoid
sigm_spread[:] = 0.0018

# (The cells are assumed all identical. Differences are coded by the connectivity.)

# ------------------------------------------------------------------------------
# parameters of transmitter-receptor systems
# ------------------------------------------------------------------------------

NR = 2  # number of transmitter-receptor systems
NR_names = ["AMPA", "GABA_A"]  # only current based synapses
# allowed_rec = np.zeros([NR, NT])  # codes which cell types allow which receptors
allowed_rec = np.array([[1, 0], [1, 0], [0, 1]])  # PC and EIN support AMPA, and IIN supports GABA_A

E_k = np.array(
    [0, -0.075])  # reversal potentials: 0 for AMPA and -75 mV for GABA (not applicable to current based synapses)

kernel_length = 0.05
max_samples = int(np.floor(kernel_length / step_size))  # 50 ms maximum kernel length

# ------------------------------------------------------------------------------
# arrange neuronal parameters
# ------------------------------------------------------------------------------

neurons = {'C_m': C_m, 'tau_m': tau_m, 'E_L': E_L, 'sigm_max': sigm_max, 'sigm_thresh': sigm_thresh,
           'sigm_spread': sigm_spread, 'NR': NR, 'NR_names': NR_names, 'E_k': E_k, 'allowed_rec': allowed_rec,
           'kernel_length': kernel_length, 'max_samples': max_samples}

###############################################################################
# Connectivity parameters
###############################################################################

# -----
# Remark:
# The connectivity between masses is characterized by strength and delay.
# For the delay, indexing is based on source and target mass, for the
# strength, also the transmitter-receptor system plays a role.
# -----

# ------------------------------------------------------------------------------
# strength for chemical synapses
# ------------------------------------------------------------------------------

conn_strength = np.zeros([NR, N, N])
conn_strength_AMPA = np.array([[0, 0.8 * 135, 0], [135, 0, 0], [0.25 * 135, 0, 0]])
conn_strength_GABA = np.array([[0, 0, 0.25 * 135], [0, 0, 0], [0, 0, 0]])
conn_strength[0] = conn_strength_AMPA
conn_strength[1] = conn_strength_GABA

# (Local connectivities are according to Jansen & Rit)

# ------------------------------------------------------------------------------
# delays for chemical synapses
# ------------------------------------------------------------------------------

conn_delay = np.zeros([N, N])  # all connections local - no delays

# ------------------------------------------------------------------------------
# gap junctions
# ------------------------------------------------------------------------------

conn_gap = np.zeros([N, N])  # currently not considered

# ------------------------------------------------------------------------------
# arrange connectivity parameters
# ------------------------------------------------------------------------------

connectivity = {'conn_strength': conn_strength, 'conn_delay': conn_delay, 'conn_gap': conn_gap}

###############################################################################
# External input: noise and systematic
###############################################################################

I_ext_exp = np.zeros(N)  # noise expectation values, in A

I_ext_std = np.ones(N)  # noise standard deviations, in A

NTime = int(round((end_time - start_time) / step_size) + 1)
ext_input = np.zeros([N, NR, NTime])  # for every mass, receptor type and time step

start_stim = 300.0  # ms
len_stim = 50.0  # ms
mag_stim = 300.0  # 1/s

stimpos = int(round(start_stim / 1000 / step_size))
ext_input[1, 0, stimpos:stimpos + int(round(len_stim / 1000 / step_size))] = mag_stim

external = {'I_ext_exp': I_ext_exp, 'I_ext_std': I_ext_std, 'ext_input': ext_input}

###############################################################################
# Initial values for state variables
###############################################################################
# V      -    membrane potential
# Zeta   -    markov process for synaptic conductivity
# Glut   -    gliogenic glutamate
# ATP    -    Adenosin triphophate
# Ca     -    intraglial calcium
# K      -    extracellular potassium
# IP3    -    glial IP3

init_V = np.zeros(N)  # membrane potentials
init_Zeta = np.zeros([N, NR])  # noise per receptor type
init_Glut = np.zeros(N)  # glutamate
init_ATP = np.zeros(N)  # ATP
init_Ca = np.zeros(N)  # calcium
init_K = np.zeros(N)  # potassium
init_IP3 = np.zeros(N)

initstate = {'V': init_V, 'Zeta': init_Zeta, 'Glut': init_Glut, 'ATP': init_ATP, 'Ca': init_Ca, 'K': init_K,
             'IP3': init_IP3}

###############################################################################
# Options
###############################################################################

syn_type = 'current'  # can be "current" or "conductance"
options = {'syn_type': syn_type}


def legacy_interface_jrm():
    ###############################################################################
    # Initialize neural network
    ###############################################################################

    nw = init_jr.nmm_net(lattice, neurons, connectivity, external, time_axis, initstate, options)

    ###############################################################################
    # Compute synaptic kernels
    ###############################################################################

    kernel_length = 0.05  # 50 ms maximum kernel length
    max_samples = int(kernel_length / step_size)  # maximum samples in kernel
    nw.precompute_kernels(max_samples, np.arange(0, kernel_length, step_size))
    # plt.plot(np.transpose(nw.kernels))


    ###############################################################################
    # Execute integration
    ###############################################################################
    # Euler solver

    Vm_now = nw.initstates['V']  # initialize current potential vector
    nw.Vm[:, 0] = Vm_now  # set first time sample to initial values
    i = 1  # set counter
    for t in nw.time_axis['T_']:
        dVm = nw.right_side(t)  # compute right side of differential equation
        Vm_now = Vm_now + nw.time_axis['step_size'] * dVm  # carry out integration
        if i < np.size(nw.Vm, 1):
            nw.Vm[:, i] = Vm_now  # store result
        i += 1  # increment counter

    Vm = nw.Vm
    return Vm


def legacy_interface_on_subclass_jrm():
    from base.nmm_network import NeuralMassModelLegacyInterface
    ###############################################################################
    # Initialize neural network
    ###############################################################################

    nw = NeuralMassModelLegacyInterface(lattice, neurons, connectivity, external, time_axis, initstate, options)

    ###############################################################################
    # Compute synaptic kernels
    ###############################################################################

    kernel_length = 0.05  # 50 ms maximum kernel length
    max_samples = int(kernel_length / step_size)  # maximum samples in kernel
    nw.precompute_kernels(max_samples, np.arange(0, kernel_length, step_size))
    # plt.plot(np.transpose(nw.kernels))


    ###############################################################################
    # Execute integration
    ###############################################################################
    # Euler solver

    Vm_now = nw.initstates['V']  # initialize current potential vector
    nw.Vm[:, 0] = Vm_now  # set first time sample to initial values
    #i = 1  # set counter
    for i, t in enumerate(nw.time_axis['T_'], start=1):
        dVm = nw.right_side(t)  # compute right side of differential equation
        Vm_now = Vm_now + nw.time_axis['step_size'] * dVm  # carry out integration
        if i < np.size(nw.Vm, 1):
            nw.Vm[:, i] = Vm_now  # store result
        #i += 1  # increment counter

    Vm = nw.Vm
    return Vm


def new_interface_jrm():
    from base.nmm_network import NeuralMassModel
    ###############################################################################
    # Initialize neural network
    ###############################################################################

    nw = NeuralMassModel(lattice, neurons, connectivity, external, time_axis, initstate, options)

    ###############################################################################
    # Compute synaptic kernels
    ###############################################################################

    kernel_length = 0.05  # 50 ms maximum kernel length
    max_samples = int(kernel_length / step_size)  # maximum samples in kernel
    nw.precompute_kernels(max_samples, np.arange(0, kernel_length, step_size))
    # plt.plot(np.transpose(nw.kernels))


    ###############################################################################
    # Execute integration
    ###############################################################################
    # Euler solver

    Vm_now = nw.initstates['V'].copy()  # initialize current potential vector
    nw.Vm[:, 0] = Vm_now.copy()  # set first time sample to initial values
    #i = 1  # set counter
    for i, t in enumerate(nw.time_axis['T_'], start=1):
        dVm = nw.right_side(t)  # compute right side of differential equation
        Vm_now = Vm_now + nw.time_axis['step_size'] * dVm  # carry out integration
        if i < np.size(nw.Vm, 1):
            nw.Vm[:, i] = Vm_now  # store result
        #i += 1  # increment counter

    Vm = nw.Vm
    return Vm


def legacy_interface_on_cleanup():
    from base.nmm_network import NeuralMassModelCleanupLegacy
    ###############################################################################
    # Initialize neural network
    ###############################################################################

    nw = NeuralMassModelCleanupLegacy(lattice, neurons, connectivity, external, time_axis, initstate, options)

    ###############################################################################
    # Compute synaptic kernels
    ###############################################################################

    kernel_length = 0.05  # 50 ms maximum kernel length
    max_samples = int(kernel_length / step_size)  # maximum samples in kernel
    nw.precompute_kernels(max_samples, np.arange(0, kernel_length, step_size))
    # plt.plot(np.transpose(nw.kernels))

    return nw.run_until_finished()


def test_legacy_interface_jrm():
    """This test ensures, that Thomas' original interface is still intact. May be changed later on, though.
    """
    legacy_vm = legacy_interface_jrm()
    new_legacy_vm = legacy_interface_on_subclass_jrm()
    assert legacy_vm.all() == new_legacy_vm.all()


def test_new_vs_old_interface_jrm():
    """This test ensures, that the new (not yet implemented) interface gives the same result as the original.
    """
    legacy_vm = legacy_interface_jrm()
    new_vm = new_interface_jrm()
    assert legacy_vm.all() == new_vm.all()


def test_cleaned_up_legacy_interface_jrm():
    """This test ensures, that Thomas' original interface is still intact. May be changed later on, though.
    """
    legacy_vm = legacy_interface_jrm()
    clean_legacy_vm = legacy_interface_on_cleanup()
    assert legacy_vm.all() == clean_legacy_vm.all()
    # import matplotlib.pyplot as plt
    # for i in range(clean_legacy_vm.shape[0]):
    #     plt.plot(clean_legacy_vm[i])
    # plt.show()


if __name__ == "__main__":
    test_cleaned_up_legacy_interface_jrm()
