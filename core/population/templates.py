"""
"""
from core.population import Population

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"


class JansenRitPyramidalCells(Population):
    """
    Pyramidal cell population as defined in Jansen & Rit(1995).

    :var synapses: list of synapse names.
    :var state_variables: list of state variable vectors collected over updates with the following entries:
            [0] - membrane potential [unit = V]
    :var input_firing_rate: list of input firing rates collected over updates [unit = 1/s]
    :var output_firing_rate: list of output firing rates collected over updates [unit = 1/s]
    :var store_state_variables: indicates whether all or only most recent state variables are stored
    :var store_input_firing_rate: indicates whether all or only last few inputs are stored
    :var store_output_firing_rate: indicates whether all or only most recent output firing rate is stored
    :var axon_plasticity: indicates whether axon plasticity mechanism is enabled or not
    :var tau_leak: time delay with which the population goes back to resting state potential [unit = s]
    :var resting_potential: resting-state membrane potential of the population [unit = V]

    """

    def __init__(self, init_state=(0., 0), step_size=5e-4, synaptic_kernel_length=100, tau_leak=0.016,
                 resting_potential=-0.075, membrane_capacitance=1e-12, synapse_params=None, axon_params=None,
                 store_state_variables=False, store_input_firing_rate=False, store_output_firing_rate=False):
        """
        Initializes a single pyramidal cell population as defined in Jansen & Rit (1995).

        :param init_state: vector of length 2, containing initial state of neural mass, i.e. membrane potential
               [unit = V] & firing rate [unit = 1/s] (default = (0,0)).
        :param step_size: scalar, size of the time step for which the population state will be updated according
               to euler formalism [unit = s] (default = 5e-4).
        :param synaptic_kernel_length: scalar that indicates number of bins the kernel should be evaluated for
               [unit = 1] (default = 100).
        :param tau_leak: scalar, time-scale with which the membrane potential of the population goes back to resting
               potential [unit = s] (default = 0.016).
        :param resting_potential: scalar, membrane potential at which no synaptic currents flow if no input arrives at
               population [unit = V] (default = -0.075).
        :param membrane_capacitance: scalar, determines average capacitance of the population cell membranes
               [unit = q/V] (default = 1e-12).
        :param synapse_params: list of dictionaries containing parameters for custom synapse type. For parameter
               explanation see synapse class (default = None).
        :param axon_params: dictionary containing parameters for custom axon type. For parameter explanation see
               axon class (default = None).
        :param store_state_variables: If false, old state variables will be erased after each state-update
               (default = False).
        :param store_input_firing_rate: If false, old input firing rates will only be kept for as much time-steps as
               necessary by synaptic kernel length (default = False).
        :param store_output_firing_rate: If false, old output firing rates will be erased after each state-update
               (default = False).

        """

        super(JansenRitPyramidalCells, self).__init__(synapses=['AMPA_current', 'GABAA_current'],
                                                      axon='JansenRit',
                                                      init_state=init_state,
                                                      step_size=step_size,
                                                      synaptic_kernel_length=synaptic_kernel_length,
                                                      tau_leak=tau_leak,
                                                      resting_potential=resting_potential,
                                                      membrane_capacitance=membrane_capacitance,
                                                      synapse_params=synapse_params,
                                                      axon_params=axon_params,
                                                      store_state_variables=store_state_variables,
                                                      store_input_firing_rate=store_input_firing_rate,
                                                      store_output_firing_rate=store_output_firing_rate)


class JansenRitExcitatoryInterneurons(Population):
    """
    Excitatory interneuron population as defined in Jansen & Rit(1995).

    :var synapses: list of synapse names.
    :var state_variables: list of state variable vectors collected over updates with the following entries:
            [0] - membrane potential [unit = V]
    :var input_firing_rate: list of input firing rates collected over updates [unit = 1/s]
    :var output_firing_rate: list of output firing rates collected over updates [unit = 1/s]
    :var store_state_variables: indicates whether all or only most recent state variables are stored
    :var store_input_firing_rate: indicates whether all or only last few inputs are stored
    :var store_output_firing_rate: indicates whether all or only most recent output firing rate is stored
    :var axon_plasticity: indicates whether axon plasticity mechanism is enabled or not
    :var tau_leak: time delay with which the population goes back to resting state potential [unit = s]
    :var resting_potential: resting-state membrane potential of the population [unit = V]

    """

    def __init__(self, init_state=(0., 0), step_size=5e-4, synaptic_kernel_length=100, tau_leak=0.016,
                 resting_potential=-0.075, membrane_capacitance=1e-12, synapse_params=None, axon_params=None,
                 store_state_variables=False, store_input_firing_rate=False, store_output_firing_rate=False):
        """
        Initializes a single excitatory interneuron population as defined in Jansen & Rit (1995).

        :param init_state: vector of length 2, containing initial state of neural mass, i.e. membrane potential
               [unit = V] & firing rate [unit = 1/s] (default = (0,0)).
        :param step_size: scalar, size of the time step for which the population state will be updated according
               to euler formalism [unit = s] (default = 5e-4).
        :param synaptic_kernel_length: scalar that indicates number of bins the kernel should be evaluated for
               [unit = 1] (default = 100).
        :param tau_leak: scalar, time-scale with which the membrane potential of the population goes back to resting
               potential [unit = s] (default = 0.016).
        :param resting_potential: scalar, membrane potential at which no synaptic currents flow if no input arrives at
               population [unit = V] (default = -0.075).
        :param membrane_capacitance: scalar, determines average capacitance of the population cell membranes
               [unit = q/V] (default = 1e-12).
        :param synapse_params: list of dictionaries containing parameters for custom synapse type. For parameter
               explanation see synapse class (default = None).
        :param axon_params: dictionary containing parameters for custom axon type. For parameter explanation see
               axon class (default = None).
        :param store_state_variables: If false, old state variables will be erased after each state-update
               (default = False).
        :param store_input_firing_rate: If false, old input firing rates will only be kept for as much time-steps as
               necessary by synaptic kernel length (default = False).
        :param store_output_firing_rate: If false, old output firing rates will be erased after each state-update
               (default = False).

        """

        super(JansenRitExcitatoryInterneurons, self).__init__(synapses=['AMPA_current'],
                                                              axon='JansenRit',
                                                              init_state=init_state,
                                                              step_size=step_size,
                                                              synaptic_kernel_length=synaptic_kernel_length,
                                                              tau_leak=tau_leak,
                                                              resting_potential=resting_potential,
                                                              membrane_capacitance=membrane_capacitance,
                                                              synapse_params=synapse_params,
                                                              axon_params=axon_params,
                                                              store_state_variables=store_state_variables,
                                                              store_input_firing_rate=store_input_firing_rate,
                                                              store_output_firing_rate=store_output_firing_rate)


class JansenRitInhibitoryInterneurons(Population):
    """
    Inhibitory interneuron population as defined in Jansen & Rit(1995).

    :var synapses: list of synapse names.
    :var state_variables: list of state variable vectors collected over updates with the following entries:
            [0] - membrane potential [unit = V]
    :var input_firing_rate: list of input firing rates collected over updates [unit = 1/s]
    :var output_firing_rate: list of output firing rates collected over updates [unit = 1/s]
    :var store_state_variables: indicates whether all or only most recent state variables are stored
    :var store_input_firing_rate: indicates whether all or only last few inputs are stored
    :var store_output_firing_rate: indicates whether all or only most recent output firing rate is stored
    :var axon_plasticity: indicates whether axon plasticity mechanism is enabled or not
    :var tau_leak: time delay with which the population goes back to resting state potential [unit = s]
    :var resting_potential: resting-state membrane potential of the population [unit = V]

    """

    def __init__(self, init_state=(0., 0), step_size=5e-4, synaptic_kernel_length=100, tau_leak=0.016,
                 resting_potential=-0.075, membrane_capacitance=1e-12, synapse_params=None, axon_params=None,
                 store_state_variables=False, store_input_firing_rate=False, store_output_firing_rate=False):
        """
        Initializes a single inhibitory interneuron population as defined in Jansen & Rit (1995).

        :param init_state: vector of length 2, containing initial state of neural mass, i.e. membrane potential
               [unit = V] & firing rate [unit = 1/s] (default = (0,0)).
        :param step_size: scalar, size of the time step for which the population state will be updated according
               to euler formalism [unit = s] (default = 5e-4).
        :param synaptic_kernel_length: scalar that indicates number of bins the kernel should be evaluated for
               [unit = 1] (default = 100).
        :param tau_leak: scalar, time-scale with which the membrane potential of the population goes back to resting
               potential [unit = s] (default = 0.016).
        :param resting_potential: scalar, membrane potential at which no synaptic currents flow if no input arrives at
               population [unit = V] (default = -0.075).
        :param membrane_capacitance: scalar, determines average capacitance of the population cell membranes
               [unit = q/V] (default = 1e-12).
        :param synapse_params: list of dictionaries containing parameters for custom synapse type. For parameter
               explanation see synapse class (default = None).
        :param axon_params: dictionary containing parameters for custom axon type. For parameter explanation see
               axon class (default = None).
        :param store_state_variables: If false, old state variables will be erased after each state-update
               (default = False).
        :param store_input_firing_rate: If false, old input firing rates will only be kept for as much time-steps as
               necessary by synaptic kernel length (default = False).
        :param store_output_firing_rate: If false, old output firing rates will be erased after each state-update
               (default = False).

        """

        super(JansenRitInhibitoryInterneurons, self).__init__(synapses=['GABAA_current'],
                                                              axon='JansenRit',
                                                              init_state=init_state,
                                                              step_size=step_size,
                                                              synaptic_kernel_length=synaptic_kernel_length,
                                                              tau_leak=tau_leak,
                                                              resting_potential=resting_potential,
                                                              membrane_capacitance=membrane_capacitance,
                                                              synapse_params=synapse_params,
                                                              axon_params=axon_params,
                                                              store_state_variables=store_state_variables,
                                                              store_input_firing_rate=store_input_firing_rate,
                                                              store_output_firing_rate=store_output_firing_rate)