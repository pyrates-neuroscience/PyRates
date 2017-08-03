"""
Definition of neural populations or neural mass that can have a number of properties.
"""

import synapses as syn
import axons as ax
import numpy as np

__author__ = "Richard Gast, Daniel Rose"
__status__ = "Development"

# TODO: Implement synaptic plasticity mechanism(s)


class Population(object):
    """
    Basic neural mass or population class. A population is defined via a number of synapses and an axon. Includes a
    method to update the state of the population.

    :var synapses: list of synapse names.
    :var state_variables: list of state variable vectors collected over updates with the following entries:
            [0] - membrane potential
    :var input_firing_rate: list of input firing rates collected over updates
    :var output_firing_rate: list of output firing rates collected over updates
    :var store_state_variables: indicates whether all or only most recent state variables are stored
    :var store_input_firing_rate: indicates whether all or only last few inputs are stored
    :var store_output_firing_rate: indicates whether all or only most recent output firing rate is stored
    :var axon_plasticity: indicates whether axon plasticity mechanism is enabled or not
    :var tau_leak: time delay with which the population goes back to resting state potential [unit = s]
    :var resting_potential: resting-state membrane potential of the population [unit = mV]

    """

    def __init__(self, synapses, axon='JansenRit', init_state=(0, 0), step_size=0.0001, kernel_length=100,
                 tau_leak=0.001, resting_potential=-70.0, synapse_params=None, axon_params=None,
                 store_state_variables=False, store_input_firing_rate=False, store_output_firing_rate=False):
        """
        Initializes a single neural mass.

        :param synapses: character string that indicates synapse type (see pre-implemented synapse sub-classes)
        :param axon: character string that indicates axon type (see pre-implemented axon sub-classes)
        :param init_state: vector of length 2, containing initial state of neural mass, i.e. membrane potential
               [unit = mV] & firing rate [unit = firing rate] (default = (0,0)).
        :param step_size: scalar, size of the time step for which the population state will be updated according
               to euler formalism [unit = s] (default = 0.1).
        :param kernel_length: scalar that indicates number of bins the kernel should be evaluated for
               [unit = time-steps] (default = 100).
        :param tau_leak: scalar, time-scale with which the membrane potential of the population goes back to resting
               potential [unit = s] (default = 0.001).
        :param resting_potential: scalar, membrane potential at which no synaptic currents flow if no input arrives at
               population [unit = mV] (default = -70.0).
        :param synapse_params: dictionary containing parameters for custom synapse type. For parameter explanation
               see synapse class (default = None).
        :param axon_params: dictionary containing parameters for custom axon type. For parameter explanation see
               axon class (default = None).
        :param store_state_variables: If false, old state variables will be erased after each state-update
               (default = False).
        :param store_input_firing_rate: If false, old input firing rates will only be kept for as much time-steps as
               necessary by synaptic kernel length (default = False).
        :param store_output_firing_rate: If false, old output firing rates will be erased after each state-update
               (default = False).

        """

        ##########################
        # check input parameters #
        ##########################

        assert len(init_state) == 2
        assert init_state[1] >= 0
        assert type(synapses) is list

        #############################
        # set population parameters #
        #############################

        self.synapses = list()
        self.state_variables = list()
        self.state_variables.append(init_state[0])
        self.input_firing_rate = list()
        self.input_firing_rate.append(np.zeros(len(synapses)))
        self.output_firing_rate = list()
        self.output_firing_rate.append(init_state[1])
        self.store_state_variables = store_state_variables
        self.store_input_firing_rate = store_input_firing_rate
        self.store_output_firing_rate = store_output_firing_rate
        self.tau_leak = tau_leak
        self.resting_potential = resting_potential

        ####################################
        # set axonal plasticity parameters #
        ####################################

        if axon_params:

            if 'tau' in axon_params:

                # if axon timescale tau is in axon parameters, set relevant parameters for axon plasticity
                self.axon_plasticity = True
                self.tau_axon = axon_params['tau']
                self.step_size = step_size
                if 'firing_rate_target' in axon_params:
                    self.firing_rate_target = axon_params['firing_rate_target']
                else:
                    self.firing_rate_target = 0.5

            else:

                self.axon_plasticity = False

        else:

            self.axon_plasticity = False

        ######################################
        # set synaptic plasticity parameters #
        ######################################

        # TODO: implement synaptic plasticity mechanism

        ###############
        # set synapse #
        ###############

        if not synapse_params:

            for i in range(len(synapses)):
                self.synapses.append(set_synapse(synapses[i], step_size, kernel_length))

        else:

            for i in range(len(synapse_params)):
                self.synapses.append(set_synapse(synapses[i], step_size, kernel_length, synapse_params[i]))

        ############
        # set axon #
        ############

        self.axon = set_axon(axon, axon_params)

    def state_update(self, synaptic_input, extrinsic_current=0.):
        """
        Updates state according to current synapse and axon parametrization.

        :param synaptic_input: vector, average firing rate arriving at each synapse [unit = firing rate].
        :param extrinsic_current: extrinsic current added to membrane potential change [unit = mA] (default = 0).

        """

        ##########################
        # check input parameters #
        ##########################

        assert all(synaptic_input) >= 0
        assert len(synaptic_input) == len(self.synapses)
        assert type(extrinsic_current)

        #############################################
        # get input firing rate and state variables #
        #############################################

        self.input_firing_rate.append(synaptic_input)
        inputs = np.asarray(self.input_firing_rate)
        states = self.state_variables[-1]

        #####################################
        # update average membrane potential #
        #####################################

        # calculate synaptic currents for each synapse
        synaptic_currents = np.zeros(len(self.synapses))
        for i in range(len(self.synapses)):

            if self.synapses[i].conductivity_based:
                synaptic_currents[i] = self.synapses[i].get_synaptic_current(inputs[:, i], states[0])
            else:
                synaptic_currents[i] = self.synapses[i].get_synaptic_current(inputs[:, i])

        synaptic_current = np.sum(synaptic_currents)

        # calculate leak current
        leak_current = (self.resting_potential - states[0]) / self.tau_leak

        # update membrane potential
        membrane_potential = states[0] + self.step_size * (synaptic_current + leak_current + extrinsic_current)

        ##############################
        # update average firing rate #
        ##############################

        firing_rate = self.axon.compute_firing_rate(states[0])

        #################################################
        # update state variables and output firing rate #
        #################################################

        self.state_variables.append(membrane_potential)
        self.output_firing_rate.append(firing_rate)

        ###################################
        # update axonal transfer function #
        ###################################

        if self.axon_plasticity:

            delta_membrane_potential_threshold = (firing_rate - self.firing_rate_target) / self.tau_axon
            self.axon.membrane_potential_threshold += self.step_size * delta_membrane_potential_threshold

        ###########################
        # update synaptic kernels #
        ###########################

        # TODO: implement differential equation that adapts synaptic efficiencies


def set_synapse(synapse, step_size, kernel_length, synapse_params=None):
    """
    Instantiates a synapse, including a method to calculate the synaptic current resulting from the average firing rate
    arriving at the population.

    :param synapse: character string that indicates synapse type (see pre-implemented synapse sub-classes)
    :param step_size: scalar, size of the time step for which the population state will be updated according
           to euler formalism [unit = ms] (default = 0.1).
    :param kernel_length: scalar that indicates number of bins the kernel should be evaluated for
           [unit = time-steps] (default = 100).
    :param synapse_params: dictionary containing parameters for custom synapse type. For parameter explanation see
           synapse class (default = None).

    :return synapse_instance: instance of synapse class, including method to compute synaptic current

    """

    ####################
    # check parameters #
    ####################

    assert (type(synapse) is str) or (not synapse)
    assert (type(synapse_params) is dict) or (not synapse_params)
    assert step_size > 0
    assert kernel_length > 0

    ###############################
    # initialize synapse instance #
    ###############################

    pre_defined_synapse = True

    if synapse == 'AMPA_current':

        syn_instance = syn.AMPACurrentSynapse(step_size, kernel_length)

    elif synapse == 'GABAA_current':

        syn_instance = syn.GABAACurrentSynapse(step_size, kernel_length)

    elif synapse == 'JansenRit_excitatory':

        syn_instance = syn.JansenRitExcitatorySynapse(step_size, kernel_length)

    elif synapse == 'JansenRit_inhibitory':

        syn_instance = syn.JansenRitInhibitorySynapse(step_size, kernel_length)

    elif synapse_params:

        syn_instance = syn.Synapse(synapse_params['efficiency'], synapse_params['tau_decay'],
                                   synapse_params['tau_rise'], step_size, kernel_length)
        pre_defined_synapse = False

    else:

        raise ValueError('Not a valid synapse type! Check synapses.py for all possible synapse types')

    #########################################################################
    # adjust pre-defined parametrization if relevant parameters were passed #
    #########################################################################

    if pre_defined_synapse and synapse_params:

        param_list = ['efficiency', 'tau_decay', 'tau_rise', 'conductivity_based', 'reversal_potential']

        for p in param_list:
            syn_instance = update_param(p, synapse_params, syn_instance)

    #########################
    # build synaptic kernel #
    #########################

    syn_instance.build_kernel()

    return syn_instance


def set_axon(axon, axon_params=None):
    """
    Instantiates an axon, including a method to calculate the average output firing rate of a population given its
    current membrane potential.

    :param axon: character string that indicates axon type (see pre-implemented axon sub-classes)
    :param axon_params: dictionary containing parameters for custom axon type. For parameter explanation see axon
           class (default = None).

    :return ax_instance: instance of axon class, including method to compute firing rate

    """

    ####################
    # check parameters #
    ####################

    assert (type(axon) is str) or (not axon)
    assert (type(axon_params) is dict) or (not axon_params)

    ############################
    # initialize axon instance #
    ############################

    pre_defined_axon = True

    if axon == 'Knoesche':

        ax_instance = ax.KnoescheAxon()

    elif axon == 'JansenRit':

        ax_instance = ax.JansenRitAxon()

    elif axon_params:

        ax_instance = ax.Axon(axon_params['max_firing_rate'], axon_params['membrane_potential_threshold'],
                              axon_params['sigmoid_steepness'])
        pre_defined_axon = False

    else:

        raise ValueError('Not a valid axon type!')

    #########################################################################
    # adjust pre-defined parametrization if relevant parameters were passed #
    #########################################################################

    if pre_defined_axon and axon_params:

        param_list = ['max_firing_rate', 'membrane_potential_threshold', 'sigmoid_steepness']

        for p in param_list:
            ax_instance = update_param(p, axon_params, ax_instance)

    return ax_instance


def update_param(param, param_dict, object_instance):
    """
    Checks whether param is a key in param_dict. If yes, the corresponding value in param_dict will be updated in
    object_instance.

    :param param: string, specifies parameter to check
    :param param_dict: dictionary, potentially contains param
    :param object_instance: object instance for which to update parameter

    :return: object
    """

    assert param in object_instance

    if param in param_dict:
        setattr(object_instance, param, param_dict[param])

    return object_instance
